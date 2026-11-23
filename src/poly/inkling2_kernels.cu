// src/poly/inkling2_kernels.cu - Inkling 2 Mini's own kernels, Phase 1: exact, unfused, FP32-accumulated.
//
// The reference for every formula is llama.cpp's inkling2 graph at the commit the plan pins
// (docs/models/reference/inkling2.cpp, kept beside this file).  Phase 1 deliberately does not fuse
// anything and does not use the tensor cores: each kernel is the plain arithmetic the graph describes, so
// that a mismatch against the reference is a bug in this code and not in an optimization.
#include "strata/poly/kernels.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::poly {

// The tensor-core attention is the default; `STRATA_INK_MMA=0` turns it off so that the two can be measured
// and compared on one binary, which is the only way to attribute a difference to it rather than to a rebuild.
// Read on every call rather than cached, so that one process can measure both paths over the same data.
static bool ink_mma_on() {
    const char* e = std::getenv("STRATA_INK_MMA");
    return !(e && e[0] == '0');
}

namespace {

constexpr int INK_THREADS = 256;

// The rows a visit's attention takes together (a block's rows share the sweep of the keys), and the keys it
// holds the logits of at once: two floats of shared memory a key (its logit and then its weight).
constexpr int INK_ATTEND_ROWS = 8;
constexpr int INK_ATTEND_TILE = 64;

inline cudaStream_t cs(void* stream) { return (cudaStream_t) stream; }
inline int grid(size_t n, int block) { return int((n + block - 1) / block); }

__device__ __forceinline__ float ink_warp_sum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o, 32);
    return x;
}

__device__ __forceinline__ float ink_warp_max(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o, 32));
    return x;
}

// One value per block summed over the block's threads.  blockDim.x is a multiple of 32 and `part` holds
// one float per warp.
__device__ __forceinline__ float ink_block_sum(float x, float* part) {
    x = ink_warp_sum(x);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float s = 0.0f;
    for (int k = 0; k < int(blockDim.x) / 32; ++k) s += part[k];
    __syncthreads();
    return s;
}

// The same shape, for a max: a softmax's stabiliser is the largest logit, not their sum.
__device__ __forceinline__ float ink_block_max(float x, float* part) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o, 32));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float m = -INFINITY;
    for (int k = 0; k < int(blockDim.x) / 32; ++k) m = fmaxf(m, part[k]);
    __syncthreads();
    return m;
}

// ggml's `clip`: scale each row (a head of `dim` values) down to an L2 norm of at most `threshold`,
// after the optional per-channel `scale`.  `scale` is `heads` rows of `dim`, as ggml_mul broadcasts a
// [dim, heads] tensor over a [dim, heads, tokens] one: the value of (channel i, head h) is at h*dim + i.
__global__ void ink_clip_l2_kernel(float* __restrict__ x, const float* __restrict__ scale, int dim, int heads,
                                  float threshold, int rows) {
    __shared__ float part[32];
    const int64_t row = blockIdx.x;
    const int h = blockIdx.y;
    if (row >= rows) return;
    float* v = x + (row * heads + h) * dim;
    if (scale) {
        const float* s = scale + (size_t) h * dim;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) v[i] *= s[i];
    }
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += v[i] * v[i];
    ss = ink_block_sum(ss, part);
    const float norm = sqrtf(ss) + 1e-6f;   // ggml_scale_bias(norm, 1.0, 1e-6)
    const float factor = fminf(fmaxf(threshold / norm, 0.0f), 1.0f);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) v[i] *= factor;
}

// situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
__global__ void ink_situ_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ out,
                               int n, float beta, float lb) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gate = beta * tanhf(g[i] / beta) / (1.0f + expf(-g[i]));
    const float up = lb > 0.0f ? lb * tanhf(u[i] / lb) : u[i];
    out[i] = gate * up;
}

// a row's gate and up scales; 1 (exact) where there is none.  The same as kernels.cu's row_scales().
__device__ __forceinline__ float2 ink_row_scales(const ExpertScales& s, int row) {
    if (!s.gate && !s.up) return make_float2(1.0f, 1.0f);
    const int e = s.ids ? s.ids[row] : s.e[row];
    return make_float2(s.gate ? s.gate[e] : 1.0f, s.up ? s.up[e] : 1.0f);
}

// situ for rows that each hold n gate values and then n up values, as swiglu_clamp_rows' layout.  Inkling 2's routed
// experts carry a global scale a gate and an up side (ffn_gate_up_exps.scale), which is applied before the activation.
__global__ void ink_situ_rows_kernel(const float* __restrict__ gate_up, float* __restrict__ out, int n, float beta,
                                    float lb, ExpertScales s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = gate_up + size_t(blockIdx.y) * 2 * n;
    const float2 sc = ink_row_scales(s, blockIdx.y);
    const float g = row[i] * sc.x;
    const float u = row[n + i] * sc.y;
    const float gate = beta * tanhf(g / beta) / (1.0f + expf(-g));
    const float up = lb > 0.0f ? lb * tanhf(u / lb) : u;
    out[size_t(blockIdx.y) * n + i] = gate * up;
}

// The causal short convolution of Inkling 2: out[t][c] = sum_i w[c][i] * x_ext[t + i][c], where x_ext's first
// `k - 1` rows are the state and the rest the call's own.  `dyn` (a full-attention layer's input-dependent
// taps) adds sum_i x_ext[t + i][c] * dyn[t][k - 1 - i][head(c)].  The result is
// (x[t][c] + out) / sqrt((dyn ? 2 : 1)*k + 1), as the reference's `convolve()` does.
__global__ void ink_conv_kernel(const float* __restrict__ x, float* __restrict__ state,
                               const float* __restrict__ w, const float* __restrict__ dyn,
                               float* __restrict__ out, int channels, int rows, int k, int n_head) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= channels || t >= rows) return;
    const float* tap = w + (size_t) c * k;
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        const int j = t + i - (k - 1);   // row of the call, or of the state when negative
        const float v = j < 0 ? state[(size_t) (j + k - 1) * channels + c]
                              : x[(size_t) j * channels + c];
        acc += tap[i] * v;
    }
    float total = acc;
    if (dyn) {
        // The projection's row for this token is n_conv taps per head, head after head - the reference
        // reshapes its [n_conv*n_heads, n_tokens] output to [n_conv, n_heads, n_tokens], so the tap of
        // (head h, lag l) sits at h*n_conv + l and not at l*n_heads + h.
        const int per = channels / n_head;
        const int h = c / per;
        const float* d = dyn + (size_t) t * k * n_head;
        float extra = 0.0f;
        for (int i = 0; i < k; ++i) {
            const int j = t + i - (k - 1);
            const float v = j < 0 ? state[(size_t) (j + k - 1) * channels + c]
                                  : x[(size_t) j * channels + c];
            extra += v * d[(size_t) h * k + (k - 1 - i)];
        }
        total += extra;
    }
    const float scale = 1.0f / sqrtf((dyn ? 2.0f : 1.0f) * (float) k + 1.0f);
    out[(size_t) t * channels + c] = (x[(size_t) t * channels + c] + total) * scale;
}

// Advance the convolution state: the last k - 1 rows of x_ext, oldest first.  A call of fewer than k - 1
// tokens leaves some of the old state in place, and its rows are read above the one being written, so a
// channel's thread can go up from 0 without meeting a row it has overwritten.
__global__ void ink_conv_state_kernel(const float* __restrict__ x, float* __restrict__ state, int channels,
                                     int rows, int k) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    for (int s = 0; s < k - 1; ++s) {
        const int j = rows + s - (k - 1);
        state[(size_t) s * channels + c] = j < 0 ? state[(size_t) (j + k - 1) * channels + c]
                                                : x[(size_t) j * channels + c];
    }
}

// The relative-position bias of one (token, head, distance): tau * sum_r bias[r][d] * rcur[token][head][r] - sink.
// `d` is the distance to the key, min(extent, pos_tok - pos_key) as llama-kv-cache's set_input_relative_pos()
// clamps it; row `extent` of the bias is the zero pad ggml_pad() adds, so a key further off weighs nothing.
__global__ void ink_rel_table_kernel(const float* __restrict__ bias, int n_rel, const float* __restrict__ rcur,
                                    const float* __restrict__ sinks, const float* __restrict__ tau,
                                    float* __restrict__ kqb, int rows, int n_head, int n_keys) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = (int64_t) rows * n_head * n_keys;
    if (i >= n) return;
    const int64_t d = i % n_keys;
    const int64_t h = (i / n_keys) % n_head;
    const int64_t row = i / ((int64_t) n_keys * n_head);
    const float* b = bias + (size_t) d * n_rel;
    const float* r = rcur + (row * n_head + h) * n_rel;
    float s = 0.0f;
    for (int e = 0; e < n_rel; ++e) s += b[e] * r[e];
    kqb[i] = (tau ? s * tau[row] : s) - (sinks ? sinks[row * n_head + h] : 0.0f);
}

// The attention with the relative-position bias: the keys a token may attend are its window's, and the bias is
// added before the softmax, with a zero-logit sink folded into the denominator.
//
// A block visits INK_ATTEND_ROWS rows of one head together, the keys a tile at a time with an online softmax:
// the largest logit and the denominator are carried across the tiles, and the running sums of the values are
// rescaled when the largest grows.  So the kernel needs no scores buffer for the whole sequence, each key's q.k is
// computed once for the whole block rather than once per output dimension, and a block's rows share the key tile's
// bytes rather than each reading them from memory.  Those are the three things that made a long prompt's attention
// quadratic in the head dim and in the sequence at once: at a 23,552-token prompt a full-context layer's 1,024 rows
// cost the one-row three-pass form 1,884 ms, 82 GFLOP/s and the card's own memory bandwidth.
//
// `kqb` holds the bias by distance, min(extent, p - key) as ink_rel_table() lays it out, and `tau` scales the
// q.k term of a full-attention layer (the bias is already tau-scaled).
__global__ void ink_attend_rel_kernel(const float* __restrict__ q, const half* __restrict__ kc,
                                     const half* __restrict__ vc, const float* __restrict__ kqb,
                                     const float* __restrict__ tau, long long pos_arg, int rows,
                                     float* __restrict__ out, int dim, int n_head, int kv_heads, float scale,
                                     int window, int n_keys, int ring, const long long* __restrict__ pos_at) {
    // a recorded launch replays with whatever position the card holds then, so it reads it rather than the argument
    const long long pos = pos_at ? *pos_at : pos_arg;
    __shared__ float qs[INK_ATTEND_ROWS][INK_THREADS];       // block's queries cached once (avoids re-reading Q per key)
    __shared__ float sc[INK_ATTEND_ROWS][INK_ATTEND_TILE];   // a tile's logits, then its weights
    __shared__ float wt[INK_ATTEND_ROWS][INK_ATTEND_TILE];
    __shared__ float corr_of[INK_ATTEND_ROWS];               // the rescale a row's running sums take this tile
    __shared__ float sum_of[INK_ATTEND_ROWS];                 // a row's running denominator
    const int h = blockIdx.x;
    const int row0 = blockIdx.y * INK_ATTEND_ROWS;
    const int nr = rows - row0 < INK_ATTEND_ROWS ? rows - row0 : INK_ATTEND_ROWS;
    if (nr <= 0) return;
    const int per = n_head / kv_heads;
    const int hkv = h / per;
    const size_t krow = (size_t) kv_heads * dim;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = int(blockDim.x) >> 5;

    for (int r = 0; r < nr; ++r) {
        const float* qr = q + ((size_t) (row0 + r) * n_head + h) * dim;
        if (threadIdx.x < dim) qs[r][threadIdx.x] = qr[threadIdx.x];
    }
    __syncthreads();

    // a row's last key; the rows of a block are consecutive, so the block's keys run from the first
    // row's first key to the last row's last one, and a key outside a row's own run is masked below
    int lo[INK_ATTEND_ROWS], hi[INK_ATTEND_ROWS];
#pragma unroll
    for (int r = 0; r < INK_ATTEND_ROWS; ++r) {
        if (r < nr) {
            const int p = int(pos) + row0 + r;
            lo[r] = window > 0 && p >= window ? p - window + 1 : 0;
            hi[r] = p;
        }
    }

    // the row of a cache the key at `j` was written to: a window layer's is a ring
    auto crow = [&](int j) { return (size_t) (ring > 0 ? j % ring : j); };

    // the bias of the key at `j` for the row at `r`: the keys past the extent read the pad row, which is zero
    auto bias_of = [&](int r, int j) {
        const int d = hi[r] - j;
        return kqb[((size_t) (row0 + r) * n_head + h) * n_keys + (size_t) (d < n_keys ? d : n_keys - 1)];
    };

    // The sink's logit is zero and it carries no value, so a row's running max starts there and its
    // denominator at the sink's weight of one: a tile whose logits are all below it still loses to it, as the
    // reference's fmaxf(.., 0) and its expf(-best) term say.
    float m[INK_ATTEND_ROWS], acc[INK_ATTEND_ROWS];
#pragma unroll
    for (int r = 0; r < INK_ATTEND_ROWS; ++r) {
        m[r] = 0.0f, acc[r] = 0.0f;
        sum_of[r] = 1.0f;   // the sink's own weight
    }

    const int last = hi[nr - 1];
    for (int64_t j0 = lo[0]; j0 <= last; j0 += INK_ATTEND_TILE) {
        const long long left = (long long) last + 1 - j0;
        const int jn = int(left < INK_ATTEND_TILE ? left : INK_ATTEND_TILE);

        // the tile's logits, a warp to a (row, key) pair: a warp reads the key's dims across its lanes
        // (coalesced), and its own reduction is the key's dot product.  A key outside a row's run is -inf,
        // which the online softmax drops without a branch, so the rows share one sweep of the keys.
        for (int e = warp; e < nr * jn; e += nwarps) {
            const int r = e / jn, k = e - r * jn, j = int(j0) + k;
            if (j < lo[r] || j > hi[r]) {
                if (lane == 0) sc[r][k] = -INFINITY;
                continue;
            }
            const half* kr = kc + crow(j) * krow + (size_t) hkv * dim;
            const float* qr = qs[r];
            float s = 0.0f;
            for (int d = lane; d < dim; d += 32) s += qr[d] * __half2float(kr[d]);
            s = ink_warp_sum(s);
            if (lane == 0) {
                const float t = tau ? tau[row0 + r] : 1.0f;
                sc[r][k] = s * scale * t + bias_of(r, j);
            }
        }
        __syncthreads();

        // a row's largest logit over the tile and the rescale its running sums take, a warp to a row
        for (int r = warp; r < nr; r += nwarps) {
            float tm = -INFINITY;
            for (int i = lane; i < jn; i += 32) tm = fmaxf(tm, sc[r][i]);
            tm = ink_warp_max(tm);
            const float mnew = fmaxf(m[r], tm);
            const float corr = expf(m[r] - mnew);
            float ts = 0.0f;
            for (int i = lane; i < jn; i += 32) {
                const float w = expf(sc[r][i] - mnew);
                wt[r][i] = w;
                ts += w;
            }
            ts = ink_warp_sum(ts);
            if (lane == 0) {
                sum_of[r] = sum_of[r] * corr + ts;
                corr_of[r] = corr;
            }
            m[r] = mnew;
        }
        __syncthreads();

        // the values: this thread's dimension of every row's tile sum, the weights read from the tile's
        // shared run, which is one float a key rather than one a key and dimension
        if (threadIdx.x < dim) {
#pragma unroll
            for (int r = 0; r < INK_ATTEND_ROWS; ++r) {
                if (r < nr) {
                    acc[r] *= corr_of[r];
                    const int i0 = lo[r] > int(j0) ? lo[r] - int(j0) : 0;
                    const int i1 = hi[r] - int(j0) < jn - 1 ? hi[r] - int(j0) : jn - 1;
#pragma unroll 4
                    for (int i = i0; i <= i1; ++i) {
                        const half* vr = vc + crow(int(j0) + i) * krow + (size_t) hkv * dim;
                        acc[r] += wt[r][i] * __half2float(vr[threadIdx.x]);
                    }
                }
            }
        }
        __syncthreads();   // the next tile overwrites sc with its logits, and wt with its weights
    }

#pragma unroll
    for (int r = 0; r < INK_ATTEND_ROWS; ++r) {
        if (r < nr) {
            float* oh = out + ((size_t) (row0 + r) * n_head + h) * dim;
            for (int e = threadIdx.x; e < dim; e += blockDim.x) oh[e] = acc[r] / sum_of[r];
        }
    }
}

// Same online-softmax relative-position attention, reading Q8_0 K/V caches (32 int8 codes + 1 fp16 scale per block).
// Quantizes the block's queries to int8 per 32-element block once into shared memory and uses STRATA_DP4A (4 int8
// MACs/instruction, 128-byte coalesced warp loads of K) for the Q.K sweep.
__global__ void ink_attend_rel_q8_kernel(const float* __restrict__ q, const int8_t* __restrict__ kc,
                                         const half* __restrict__ ks, const int8_t* __restrict__ vc,
                                         const half* __restrict__ vs, const float* __restrict__ kqb,
                                         const float* __restrict__ tau, long long pos_arg, int rows,
                                         float* __restrict__ out, int dim, int n_head, int kv_heads, float scale,
                                         int window, int n_keys, int ring, const long long* __restrict__ pos_at) {
    const long long pos = pos_at ? *pos_at : pos_arg;
    __shared__ int qq4[INK_ATTEND_ROWS][INK_THREADS / 4];
    __shared__ float qscale[INK_ATTEND_ROWS][INK_THREADS / 32];
    __shared__ float sc[INK_ATTEND_ROWS][INK_ATTEND_TILE];
    __shared__ float wt[INK_ATTEND_ROWS][INK_ATTEND_TILE];
    __shared__ float corr_of[INK_ATTEND_ROWS];
    __shared__ float sum_of[INK_ATTEND_ROWS];
    const int h = blockIdx.x;
    const int row0 = blockIdx.y * INK_ATTEND_ROWS;
    const int nr = rows - row0 < INK_ATTEND_ROWS ? rows - row0 : INK_ATTEND_ROWS;
    if (nr <= 0) return;
    const int per = n_head / kv_heads;
    const int hkv = h / per;
    const int blocks_per_head = dim >> 5;
    const size_t krow = (size_t) kv_heads * dim;
    const size_t srow = (size_t) kv_heads * blocks_per_head;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = int(blockDim.x) >> 5;

    // Quantize each 32-element block of this block's queries to int8 once before the key-tile loop.
    for (int rb = warp; rb < nr * blocks_per_head; rb += nwarps) {
        const int r = rb / blocks_per_head;
        const int b = rb - r * blocks_per_head;
        const int d = (b << 5) + lane;
        const float v = q[((size_t) (row0 + r) * n_head + h) * dim + d];
        const float amax = ink_warp_max(fabsf(v));
        const float qsc = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        int qv = __float2int_rn(v * inv);
        qv = qv > 127 ? 127 : (qv < -127 ? -127 : qv);
        ((int8_t*) qq4[r])[d] = (int8_t) qv;
        if (lane == 0) qscale[r][b] = qsc;
    }
    __syncthreads();

    int lo[INK_ATTEND_ROWS], hi[INK_ATTEND_ROWS];
#pragma unroll
    for (int r = 0; r < INK_ATTEND_ROWS; ++r) {
        if (r < nr) {
            const int p = int(pos) + row0 + r;
            lo[r] = window > 0 && p >= window ? p - window + 1 : 0;
            hi[r] = p;
        }
    }

    auto crow = [&](int j) { return (size_t) (ring > 0 ? j % ring : j); };
    auto bias_of = [&](int r, int j) {
        const int d = hi[r] - j;
        return kqb[((size_t) (row0 + r) * n_head + h) * n_keys + (size_t) (d < n_keys ? d : n_keys - 1)];
    };

    float m[INK_ATTEND_ROWS], acc[INK_ATTEND_ROWS];
#pragma unroll
    for (int r = 0; r < INK_ATTEND_ROWS; ++r) {
        m[r] = 0.0f, acc[r] = 0.0f;
        sum_of[r] = 1.0f;
    }

    const int words_per_head = dim >> 2;   // 32 int32 words for dim = 128 (1 iteration per warp)
    const int last = hi[nr - 1];
    for (int64_t j0 = lo[0]; j0 <= last; j0 += INK_ATTEND_TILE) {
        const long long left = (long long) last + 1 - j0;
        const int jn = int(left < INK_ATTEND_TILE ? left : INK_ATTEND_TILE);

        for (int e = warp; e < nr * jn; e += nwarps) {
            const int r = e / jn, k = e - r * jn, j = int(j0) + k;
            if (j < lo[r] || j > hi[r]) {
                if (lane == 0) sc[r][k] = -INFINITY;
                continue;
            }
            const size_t cr = crow(j);
            const int* kr4 = (const int*) (kc + cr * krow + (size_t) hkv * dim);
            const int* qr4 = qq4[r];
            const half* ksr = ks + cr * srow + (size_t) hkv * blocks_per_head;
            float s = 0.0f;
            for (int w = lane; w < words_per_head; w += 32) {
                const int b = w >> 3;   // 8 int32 words = 32 int8 elements = 1 Q8_0 block
                const int dp = STRATA_DP4A(qr4[w], kr4[w], 0);
                s += (float) dp * (qscale[r][b] * __half2float(ksr[b]));
            }
            s = ink_warp_sum(s);
            if (lane == 0) {
                const float t = tau ? tau[row0 + r] : 1.0f;
                sc[r][k] = s * scale * t + bias_of(r, j);
            }
        }
        __syncthreads();

        for (int r = warp; r < nr; r += nwarps) {
            float tm = -INFINITY;
            for (int i = lane; i < jn; i += 32) tm = fmaxf(tm, sc[r][i]);
            tm = ink_warp_max(tm);
            const float mnew = fmaxf(m[r], tm);
            const float corr = expf(m[r] - mnew);
            float ts = 0.0f;
            for (int i = lane; i < jn; i += 32) {
                const float w = expf(sc[r][i] - mnew);
                wt[r][i] = w;
                ts += w;
            }
            ts = ink_warp_sum(ts);
            if (lane == 0) {
                sum_of[r] = sum_of[r] * corr + ts;
                corr_of[r] = corr;
            }
            m[r] = mnew;
        }
        __syncthreads();

        if (threadIdx.x < dim) {
            const int b = threadIdx.x >> 5;
#pragma unroll
            for (int r = 0; r < INK_ATTEND_ROWS; ++r) {
                if (r < nr) {
                    acc[r] *= corr_of[r];
                    const int i0 = lo[r] > int(j0) ? lo[r] - int(j0) : 0;
                    const int i1 = hi[r] - int(j0) < jn - 1 ? hi[r] - int(j0) : jn - 1;
#pragma unroll 4
                    for (int i = i0; i <= i1; ++i) {
                        const size_t cr = crow(int(j0) + i);
                        const int8_t* vr = vc + cr * krow + (size_t) hkv * dim;
                        const half* vsr = vs + cr * srow + (size_t) hkv * blocks_per_head;
                        acc[r] += (wt[r][i] * __half2float(vsr[b])) * (float) vr[threadIdx.x];
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < INK_ATTEND_ROWS; ++r) {
        if (r < nr) {
            float* oh = out + ((size_t) (row0 + r) * n_head + h) * dim;
            for (int e = threadIdx.x; e < dim; e += blockDim.x) oh[e] = acc[r] / sum_of[r];
        }
    }
}

// Flash-decoding's split of one row's keys across blocks.  Decoding launches one block a head, which is
// 24 blocks for this model: a fraction of the card's SMs, each scanning the whole key range serially.  A
// full-context layer at 128K has 131,072 keys, so most of the card sits idle while each head walks them.
//
// Block (h, sp) scans keys [j_begin, j_end] of head h and leaves its unnormalized value sum in `part_out`
// and its (largest logit, denominator) in `part_ml`.  Only the first split carries the sink, whose logit is
// zero and whose weight is one; the others start at -inf and zero, so a split that saw no key is dropped by
// the reduce's rescale rather than adding a phantom sink to every head.
__global__ void ink_attend_rel_split_kernel(const float* __restrict__ q, const half* __restrict__ kc,
                                           const half* __restrict__ vc, const float* __restrict__ kqb,
                                           const float* __restrict__ tau, long long pos_arg, int splits,
                                           float* __restrict__ part_out, float* __restrict__ part_ml, int dim,
                                           int n_head, int kv_heads, float scale, int n_keys, int ring,
                                           const long long* __restrict__ pos_at) {
    const long long pos = pos_at ? *pos_at : pos_arg;
    __shared__ float qs[INK_THREADS];
    __shared__ float sc[INK_ATTEND_TILE];
    __shared__ float wt[INK_ATTEND_TILE];
    __shared__ float corr_s;
    const int h = blockIdx.x, sp = blockIdx.y;
    const int per = n_head / kv_heads;
    const int hkv = h / per;
    const size_t krow = (size_t) kv_heads * dim;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = int(blockDim.x) >> 5;
    float* acc = part_out + ((size_t) sp * n_head + h) * dim;

    // this split's keys: the [0, pos] range cut into `splits` chunks, and empty past the end
    const int last = int(pos);
    const int chunk = (last + 1 + splits - 1) / splits;
    const int j_begin = sp * chunk;
    const int j_end = (sp + 1) * chunk - 1 < last ? (sp + 1) * chunk - 1 : last;

    for (int e = threadIdx.x; e < dim; e += blockDim.x) {
        qs[e] = q[(size_t) h * dim + e];
        acc[e] = 0.0f;
    }
    float m = sp == 0 ? 0.0f : -INFINITY;   // the sink's logit, in the first split only
    float sum = sp == 0 ? 1.0f : 0.0f;     // the sink's weight, in the first split only
    if (j_begin > j_end) {
        if (threadIdx.x == 0) {
            part_ml[((size_t) sp * n_head + h) * 2] = m;
            part_ml[((size_t) sp * n_head + h) * 2 + 1] = sum;
        }
        return;
    }
    __syncthreads();

    auto crow = [&](int j) { return (size_t) (ring > 0 ? j % ring : j); };
    auto bias_of = [&](int j) {
        const int d = int(pos) - j;
        return kqb[(size_t) h * n_keys + (size_t) (d < n_keys ? d : n_keys - 1)];
    };

    float accv = 0.0f;
    for (int j0 = j_begin; j0 <= j_end; j0 += INK_ATTEND_TILE) {
        const int jn = j_end + 1 - j0 < INK_ATTEND_TILE ? j_end + 1 - j0 : INK_ATTEND_TILE;

        for (int e = warp; e < jn; e += nwarps) {
            const int k = e, j = j0 + k;
            const half* kr = kc + crow(j) * krow + (size_t) hkv * dim;
            float s = 0.0f;
            for (int d = lane; d < dim; d += 32) s += qs[d] * __half2float(kr[d]);
            s = ink_warp_sum(s);
            if (lane == 0) sc[k] = s * scale * (tau ? tau[0] : 1.0f) + bias_of(j);
        }
        __syncthreads();

        // one row, so its largest logit and its weights are warp 0's; the others wait at the barrier
        if (warp == 0) {
            float tm = -INFINITY;
            for (int i = lane; i < jn; i += 32) tm = fmaxf(tm, sc[i]);
            tm = ink_warp_max(tm);
            const float mnew = fmaxf(m, tm);
            // a split with no key yet takes the whole tile: expf(-inf - mnew) is zero anyway, but
            // -inf - -inf would be a NaN, and this keeps the first tile of every split clean
            const float corr = m == -INFINITY ? 0.0f : expf(m - mnew);
            float ts = 0.0f;
            for (int i = lane; i < jn; i += 32) {
                const float w = expf(sc[i] - mnew);
                wt[i] = w;
                ts += w;
            }
            ts = ink_warp_sum(ts);
            if (lane == 0) {
                sum = sum * corr + ts;
                corr_s = corr;
            }
            m = mnew;
        }
        __syncthreads();

        if (threadIdx.x < dim) {
            accv *= corr_s;
            for (int i = 0; i < jn; ++i) {
                const half* vr = vc + crow(j0 + i) * krow + (size_t) hkv * dim;
                accv += wt[i] * __half2float(vr[threadIdx.x]);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < dim) acc[threadIdx.x] = accv;
    if (threadIdx.x == 0) {
        part_ml[((size_t) sp * n_head + h) * 2] = m;
        part_ml[((size_t) sp * n_head + h) * 2 + 1] = sum;
    }
}

// The same split of one row's keys over Q8_0 K/V caches: the query's 32-element blocks are quantized
// once, and the Q.K sweep is STRATA_DP4A's, as the one-block kernel's is.
__global__ void ink_attend_rel_q8_split_kernel(const float* __restrict__ q, const int8_t* __restrict__ kc,
                                              const half* __restrict__ ks, const int8_t* __restrict__ vc,
                                              const half* __restrict__ vs, const float* __restrict__ kqb,
                                              const float* __restrict__ tau, long long pos_arg, int splits,
                                              float* __restrict__ part_out, float* __restrict__ part_ml, int dim,
                                              int n_head, int kv_heads, float scale, int n_keys, int ring,
                                              const long long* __restrict__ pos_at) {
    const long long pos = pos_at ? *pos_at : pos_arg;
    __shared__ int qq4[INK_THREADS / 4];
    __shared__ float qscale[INK_THREADS / 32];
    __shared__ float sc[INK_ATTEND_TILE];
    __shared__ float wt[INK_ATTEND_TILE];
    __shared__ float corr_s;
    const int h = blockIdx.x, sp = blockIdx.y;
    const int per = n_head / kv_heads;
    const int hkv = h / per;
    const int blocks_per_head = dim >> 5;
    const size_t krow = (size_t) kv_heads * dim;
    const size_t srow = (size_t) kv_heads * blocks_per_head;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = int(blockDim.x) >> 5;
    float* acc = part_out + ((size_t) sp * n_head + h) * dim;

    const int last = int(pos);
    const int chunk = (last + 1 + splits - 1) / splits;
    const int j_begin = sp * chunk;
    const int j_end = (sp + 1) * chunk - 1 < last ? (sp + 1) * chunk - 1 : last;

    // one 32-element block a warp quantizes, as the one-block kernel does
    if (warp < blocks_per_head) {
        const int b = warp, d = (b << 5) + lane;
        const float v = q[(size_t) h * dim + d];
        const float amax = ink_warp_max(fabsf(v));
        const float qsc = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        int qv = __float2int_rn(v * inv);
        qv = qv > 127 ? 127 : (qv < -127 ? -127 : qv);
        ((int8_t*) qq4)[d] = (int8_t) qv;
        if (lane == 0) qscale[b] = qsc;
    }
    for (int e = threadIdx.x; e < dim; e += blockDim.x) acc[e] = 0.0f;
    float m = sp == 0 ? 0.0f : -INFINITY;
    float sum = sp == 0 ? 1.0f : 0.0f;
    if (j_begin > j_end) {
        if (threadIdx.x == 0) {
            part_ml[((size_t) sp * n_head + h) * 2] = m;
            part_ml[((size_t) sp * n_head + h) * 2 + 1] = sum;
        }
        return;
    }
    __syncthreads();

    auto crow = [&](int j) { return (size_t) (ring > 0 ? j % ring : j); };
    auto bias_of = [&](int j) {
        const int d = int(pos) - j;
        return kqb[(size_t) h * n_keys + (size_t) (d < n_keys ? d : n_keys - 1)];
    };

    const int words_per_head = dim >> 2;
    float accv = 0.0f;
    for (int j0 = j_begin; j0 <= j_end; j0 += INK_ATTEND_TILE) {
        const int jn = j_end + 1 - j0 < INK_ATTEND_TILE ? j_end + 1 - j0 : INK_ATTEND_TILE;

        for (int e = warp; e < jn; e += nwarps) {
            const int k = e, j = j0 + k;
            const size_t cr = crow(j);
            const int* kr4 = (const int*) (kc + cr * krow + (size_t) hkv * dim);
            const half* ksr = ks + cr * srow + (size_t) hkv * blocks_per_head;
            float s = 0.0f;
            for (int w = lane; w < words_per_head; w += 32) {
                const int b = w >> 3;
                const int dp = STRATA_DP4A(qq4[w], kr4[w], 0);
                s += (float) dp * (qscale[b] * __half2float(ksr[b]));
            }
            s = ink_warp_sum(s);
            if (lane == 0) sc[k] = s * scale * (tau ? tau[0] : 1.0f) + bias_of(j);
        }
        __syncthreads();

        if (warp == 0) {
            float tm = -INFINITY;
            for (int i = lane; i < jn; i += 32) tm = fmaxf(tm, sc[i]);
            tm = ink_warp_max(tm);
            const float mnew = fmaxf(m, tm);
            const float corr = m == -INFINITY ? 0.0f : expf(m - mnew);
            float ts = 0.0f;
            for (int i = lane; i < jn; i += 32) {
                const float w = expf(sc[i] - mnew);
                wt[i] = w;
                ts += w;
            }
            ts = ink_warp_sum(ts);
            if (lane == 0) {
                sum = sum * corr + ts;
                corr_s = corr;
            }
            m = mnew;
        }
        __syncthreads();

        if (threadIdx.x < dim) {
            const int b = threadIdx.x >> 5;
            accv *= corr_s;
            for (int i = 0; i < jn; ++i) {
                const size_t cr = crow(j0 + i);
                const int8_t* vr = vc + cr * krow + (size_t) hkv * dim;
                const half* vsr = vs + cr * srow + (size_t) hkv * blocks_per_head;
                accv += (wt[i] * __half2float(vsr[b])) * (float) vr[threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < dim) acc[threadIdx.x] = accv;
    if (threadIdx.x == 0) {
        part_ml[((size_t) sp * n_head + h) * 2] = m;
        part_ml[((size_t) sp * n_head + h) * 2 + 1] = sum;
    }
}

// The reduce over a head's splits: the largest of their logits is the row's, and each split's value sum
// and denominator are rescaled by their distance from it before they are added, so the result is the one the
// unsplit kernel's online softmax reaches over the same keys.
__global__ void ink_attend_split_reduce_kernel(const float* __restrict__ part_out,
                                              const float* __restrict__ part_ml, float* __restrict__ out, int dim,
                                              int n_head, int splits) {
    const int h = blockIdx.x;
    float mall = -INFINITY;
    for (int sp = 0; sp < splits; ++sp) {
        const float ms = part_ml[((size_t) sp * n_head + h) * 2];
        if (ms > mall) mall = ms;
    }
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float a = 0.0f, s = 0.0f;
        for (int sp = 0; sp < splits; ++sp) {
            const float ms = part_ml[((size_t) sp * n_head + h) * 2];
            const float ss = part_ml[((size_t) sp * n_head + h) * 2 + 1];
            if (ss == 0.0f) continue;   // a split that saw no key carries no value and no weight
            const float w = expf(ms - mall);
            a += w * part_out[((size_t) sp * n_head + h) * dim + d];
            s += w * ss;
        }
        out[(size_t) h * dim + d] = a / s;
    }
}

// The attention over the depth (`mix()`): states[n_states - 1] is `cur` and states[0..] the bank's blocks,
// newest first, as the reference's concat leaves them; every state is RMS-normed without a gain first, and each
// of the n_q learned queries mixes the normed ones.
__global__ void ink_depth_mix_kernel(const float* __restrict__ states, const float* __restrict__ query,
                                    float* __restrict__ out, int n_embd, int n_states, int n_q, int rows) {
    __shared__ float part[32];
    __shared__ float scores[64];
    __shared__ float w[64];
    __shared__ float nrm[64];
    const int row = blockIdx.x;
    const int qi = blockIdx.y;
    if (row >= rows) return;
    const float inv = 1.0f / sqrtf((float) n_embd);

    // the norm each state is scaled by, which both the scores and the sum read
    for (int s = 0; s < n_states; ++s) {
        const float* st = states + ((size_t) row * n_states + s) * n_embd;
        float ss = 0.0f;
        for (int e = threadIdx.x; e < n_embd; e += blockDim.x) ss += st[e] * st[e];
        ss = ink_block_sum(ss, part);
        if (threadIdx.x == 0) nrm[s] = 1.0f / sqrtf(ss / (float) n_embd + 1e-6f);
        __syncthreads();
    }

    // scores[s] = rms_norm(states[s]) . query[qi] / sqrt(n_embd)
    for (int s = 0; s < n_states; ++s) {
        const float* st = states + ((size_t) row * n_states + s) * n_embd;
        const float* qv = query + (size_t) qi * n_embd;
        const float scale = nrm[s];
        float dot = 0.0f;
        for (int e = threadIdx.x; e < n_embd; e += blockDim.x) dot += st[e] * scale * qv[e];
        dot = ink_block_sum(dot, part);
        if (threadIdx.x == 0) scores[s] = dot * inv;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float best = -INFINITY;
        for (int s = 0; s < n_states; ++s) best = fmaxf(best, scores[s]);
        float sum = 0.0f;
        for (int s = 0; s < n_states; ++s) sum += expf(scores[s] - best);
        for (int s = 0; s < n_states; ++s) w[s] = expf(scores[s] - best) / sum;
    }
    __syncthreads();

    // out[query][token][embd]: a query's mix of every token is one contiguous run of rows * n_embd
    // floats, which is the block every caller hands in (mix_in_ holds n_q of them, msg_ four) and reads
    // as mix_in_ + qi * rows * n_embd.  A query of 1 (the output block, a window layer) has the
    // reference's [n_embd, 1, n_tok] layout either way round, so only the wider mixes move.
    float* o = out + ((size_t) qi * rows + row) * n_embd;
    for (int e = threadIdx.x; e < n_embd; e += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_states; ++s) acc += w[s] * states[((size_t) row * n_states + s) * n_embd + e] * nrm[s];
        o[e] = acc;
    }
}

// The states the depth attention mixes, a token at a time: the bank's blocks newest first and then `cur`,
// which the reference's concat leaves last.  `bank` holds max_bank blocks of `bank_stride` floats each, and
// only rows [first, first + rows) of them are read.
__global__ void ink_bank_states_kernel(const float* __restrict__ bank, const float* __restrict__ cur,
                                     float* __restrict__ out, int n_embd, int n_bank, int bank_stride,
                                     long long first, int rows) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = (int64_t) rows * (n_bank + 1) * n_embd;
    if (i >= n) return;
    const int64_t e = i % n_embd;
    const int64_t s = (i / n_embd) % (n_bank + 1);
    const int64_t row = i / ((int64_t) n_embd * (n_bank + 1));
    const int64_t at = (first + row) * n_embd + e;
    out[i] = s < n_bank ? bank[(size_t) (n_bank - 1 - s) * (size_t) bank_stride + at] : cur[at];
}

// The log-scaled position factor of a full-context layer: 1 + alpha*log(max(1, (pos + 1)/floor)), which
// multiplies both the queries and the relative projection (the reference's `tau`).
__global__ void ink_tau_kernel(float* __restrict__ tau, long long pos_arg, int floor, float alpha, int rows,
                              const long long* __restrict__ pos_at) {
    const long long pos = pos_at ? *pos_at : pos_arg;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float p = (float) (pos + row + 1);
    tau[row] = 1.0f + alpha * logf(fmaxf(1.0f, p / (float) floor));
}

// The looped core's control: (1 - forget)*x + gain*message, where gain and forget are the rank-`rank` taps of
// the two RMS-normed inputs.  `gate` is 2 * n_embd floats of scratch.
__global__ void ink_control_kernel(float* __restrict__ x, const float* __restrict__ msg,
                                 const float* __restrict__ down, const float* __restrict__ gain,
                                 const float* __restrict__ forget, int n_embd, int rank, int rows,
                                 float* __restrict__ gate) {
    __shared__ float part[32];
    const int row = blockIdx.x;
    if (row >= rows) return;
    float* xr = x + (size_t) row * n_embd;
    const float* mr = msg + (size_t) row * n_embd;
    float* gr = gate + (size_t) row * rank;
    float* cat = gate + (size_t) rows * rank;   // the concatenated normed inputs, after every row's gate

    // gate = concat(rms_norm(x), rms_norm(msg))
    float* cat_row = cat + (size_t) row * 2 * n_embd;
    for (int pass = 0; pass < 2; ++pass) {
        const float* src = pass == 0 ? xr : mr;
        float ss = 0.0f;
        for (int e = threadIdx.x; e < n_embd; e += blockDim.x) ss += src[e] * src[e];
        ss = ink_block_sum(ss, part);
        const float norm = 1.0f / sqrtf(ss / (float) n_embd + 1e-6f);
        for (int e = threadIdx.x; e < n_embd; e += blockDim.x) cat_row[pass * n_embd + e] = src[e] * norm;
        __syncthreads();
    }

    // the rank-`rank` gate, then the two taps of it
    for (int r = threadIdx.x; r < rank; r += blockDim.x) {
        float s = 0.0f;
        const float* dr = down + (size_t) r * 2 * n_embd;
        for (int e = 0; e < 2 * n_embd; ++e) s += dr[e] * cat_row[e];
        gr[r] = s;
    }
    __syncthreads();

    // x = x - tanh(forget . gate) * x + tanh(gain . gate) * message
    // The two tap matrices are {rank, n_embd}, as create_tensor() declares them, so the value of
    // (rank r, embd e) sits at e*rank + r and not at r*n_embd + e.  ctrl_down is {2*n_embd, rank}
    // and is read above the other way round, which is why the two must be indexed differently.
    for (int e = threadIdx.x; e < n_embd; e += blockDim.x) {
        float g = 0.0f, f = 0.0f;
        const size_t at = (size_t) e * rank;
        for (int r = 0; r < rank; ++r) {
            const float gr_ = gr[r];
            g += gain[at + r] * gr_;
            f += forget[at + r] * gr_;
        }
        xr[e] = xr[e] - tanhf(f) * xr[e] + tanhf(g) * mr[e];
    }
}

// The final softcap and the padded vocabulary the reference drops: the rows keep their width, and the
// columns past the unpadded vocabulary become -inf so no sampler can pick one.
__global__ void ink_finish_logits_kernel(float* __restrict__ logits, int n_vocab, int n_unpadded, float cap,
                                       int rows) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = (int64_t) rows * n_vocab;
    if (i >= n) return;
    if ((int) (i % n_vocab) >= n_unpadded) {
        logits[i] = -INFINITY;
        return;
    }
    if (cap > 0.0f) logits[i] = cap * tanhf(logits[i] * (1.0f / cap));
}

}  // namespace

void ink_clip_l2(float* x, const float* scale, int dim, int heads, int rows, float threshold, void* stream) {
    if (rows <= 0) return;
    ink_clip_l2_kernel<<<dim3((unsigned) rows, (unsigned) heads), INK_THREADS, 0, cs(stream)>>>(x, scale, dim,
                                                                                         heads, threshold, rows);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_clip_l2: ") + cudaGetErrorString(e));
}

void ink_situ(const float* gate, const float* up, float* out, int n, int rows, float beta, float linear_beta,
              void* stream) {
    const int64_t all = (int64_t) n * rows;
    if (all <= 0) return;
    ink_situ_kernel<<<grid((size_t) all, INK_THREADS), INK_THREADS, 0, cs(stream)>>>(gate, up, out, n, beta,
                                                                                  linear_beta);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_situ: ") + cudaGetErrorString(e));
}

void ink_situ_rows(const float* gate_up, float* out, int n, int rows, float beta, float linear_beta, void* stream,
                   const ExpertScales* scales) {
    if (rows <= 0) return;
    ink_situ_rows_kernel<<<dim3(grid((size_t) n, 256), (unsigned) rows), 256, 0, cs(stream)>>>(
        gate_up, out, n, beta, linear_beta, scales ? *scales : ExpertScales{});
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_situ_rows: ") + cudaGetErrorString(e));
}

void ink_conv(const float* x, float* state, const float* w, const float* dyn, float* out, int channels,
              int rows, int k, int n_head, void* stream) {
    if (rows <= 0) return;
    ink_conv_kernel<<<dim3((unsigned) grid((size_t) channels, INK_THREADS), (unsigned) rows), INK_THREADS, 0, cs(stream)>>>(
        x, state, w, dyn, out, channels, rows, k, n_head);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_conv: ") + cudaGetErrorString(e));
    ink_conv_state_kernel<<<grid((size_t) channels, INK_THREADS), INK_THREADS, 0, cs(stream)>>>(x, state, channels, rows, k);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_conv state: ") + cudaGetErrorString(e));
}

// ---- The tensor cores' prefill attention -------------------------------------------------------------
// A prompt's rows are many and its keys are all of them, so Q.K and P.V are where a long prompt's reading goes:
// the scalar kernels above spend a full FMA per (row, key, dimension) of both products, which is the quadratic wall:
// removing it is worth 2.10x of an 8K prefill's time, 3.46x of a 32K one and 6.97x of a 128K one (§4.6 of the
// progress report).  This kernel runs both products on the tensor cores'
// m16n8k16 step, as attn_flash128_kernel (kernels.cu) does for Flint, and keeps everything else exactly as
// ink_attend_rel_kernel has it: the bias by distance, the per-row tau, the zero-logit sink, the window and the ring.
//
// A block takes one head and INK_MMA_ROWS consecutive rows of it, a warp 16 of those rows, and walks the keys
// INK_MMA_KEYS at a time with the online softmax in the registers, in log2 units.  The fragment layouts are PTX's
// for mma.m16n8k16: a thread holds rows gid and gid + 8, columns 2 tig and 2 tig + 1 of each block of 8.
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750)
#define INK_MMA 0
#else
#define INK_MMA 1
#endif

constexpr int INK_MMA_ROWS = 64;               // the rows a block takes
constexpr int INK_MMA_KEYS = 64;               // the keys a tile walks
constexpr int INK_MMA_DIM = 128;               // the head dim it is for
constexpr int INK_MMA_ROW = INK_MMA_DIM + 8;   // a shared row of halves, 16-byte aligned
constexpr int INK_MMA_THREADS = 4 * 32;
constexpr int INK_MMA_MIN = 32;                // fewer rows than this keep the scalar kernels
// the keys and the values, two buffers each: one is worked on while the next is copied in
constexpr int INK_MMA_SMEM = 2 * 2 * INK_MMA_KEYS * INK_MMA_ROW * (int) sizeof(half);

namespace {

#if INK_MMA
__device__ __forceinline__ void ink_mma(float* c, const uint32_t* a, const uint32_t* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ void ink_ldmatrix(uint32_t* r, const half* p, bool trans) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    if (trans)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ uint32_t ink_half2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// 16 bytes from global to shared, or zeros where `whole` is false: asynchronous from Ampere on (a thread's copies
// are a group from ink_commit() on, waited for by ink_wait()), a plain load and store on Turing.
__device__ __forceinline__ void ink_copy(half* dst, const half* src, bool whole) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src), "r"(whole ? 16 : 0));
#else
    *reinterpret_cast<uint4*>(dst) = whole ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
#endif
}

__device__ __forceinline__ void ink_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int ahead> __device__ __forceinline__ void ink_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(ahead));
#endif
}

// A key or value tile into the shared buffer: 8 halves a thread, the Q8_0 codes dequantized with their block's
// scale.  `at` is the element offset of the first of the eight, so the two caches' element sizes do not matter here.
template <bool Q8>
__device__ __forceinline__ void ink_tile(half* dst, const void* base, size_t at, const half* sc, bool whole) {
    if (!whole) {
        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        return;
    }
    if constexpr (Q8) {
        const int8_t* s8 = reinterpret_cast<const int8_t*>(base) + at;
        const float k = __half2float(*sc);
        for (int e = 0; e < 8; ++e) dst[e] = __float2half_rn((float) s8[e] * k);
    } else {
        ink_copy(dst, reinterpret_cast<const half*>(base) + at, whole);
    }
}

// The Q.K and P.V products on the tensor cores, the softmax and the bias as ink_attend_rel_kernel has them.
template <bool Q8>
__global__ void __launch_bounds__(INK_MMA_THREADS)
ink_attend_rel_mma_kernel(const float* __restrict__ q, const void* __restrict__ kc,
                          const uint16_t* __restrict__ ks, const void* __restrict__ vc,
                          const uint16_t* __restrict__ vs, const float* __restrict__ kqb,
                          const float* __restrict__ tau, long long pos_arg, int rows, float* __restrict__ out,
                          int n_head, int kv_heads, float scale, int window, int n_keys, int ring,
                          const long long* __restrict__ pos_at) {
    const long long pos = pos_at ? *pos_at : pos_arg;
#if INK_MMA
    constexpr int DIM = INK_MMA_DIM, KEYS = INK_MMA_KEYS, ROW = INK_MMA_ROW;
    extern __shared__ __align__(16) half sm[];
    half(*kb)[KEYS][ROW] = reinterpret_cast<half(*)[KEYS][ROW]>(sm);
    half(*vb)[KEYS][ROW] = reinterpret_cast<half(*)[KEYS][ROW]>(sm + 2 * KEYS * ROW);
    const int h = blockIdx.x, row0 = blockIdx.y * INK_MMA_ROWS;
    const int nr = rows - row0 < INK_MMA_ROWS ? rows - row0 : INK_MMA_ROWS;
    if (nr <= 0) return;
    const int per = n_head / kv_heads, hkv = h / per;
    const size_t krow = (size_t) kv_heads * DIM, srow = (size_t) kv_heads * (DIM >> 5);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const int r0 = row0 + w * 16;          // the warp's first row
    const int ra = r0 + gid, rb = r0 + gid + 8;   // the two rows a thread holds
    const float sl = scale * 1.44269504f;    // the scores' base, in log2 units

    // the queries: rounded to halves once, each row scaled by its own tau
    uint32_t qa[8][4];
    for (int k = 0; k < 8; ++k)
        for (int i = 0; i < 2; ++i) {
            const int r = i == 0 ? ra : rb;
            const float f = r < rows && tau ? tau[r] * sl : sl;
            for (int hi = 0; hi < 2; ++hi) {
                float2 x = make_float2(0.0f, 0.0f);
                if (r < rows)
                    x = *reinterpret_cast<const float2*>(q + ((size_t) r * n_head + h) * DIM + 16 * k + 8 * hi + 2 * tig);
                qa[k][i + 2 * hi] = ink_half2(x.x * f, x.y * f);
            }
        }

    // The block's keys: from its first row's window to its last row's own position.  Both come from the
    // block and not from this warp's rows, so that a warp whose rows run past `rows` (a block whose last
    // rows are short) reads the block's range and not one beyond it -- the mask drops its own logits.
    const int nb = rows - row0 < INK_MMA_ROWS ? rows - row0 : INK_MMA_ROWS;
    const long long hi = pos + row0 + nb;
    const long long lo = window > 0 ? (pos + row0 - window + 1 > 0 ? pos + row0 - window + 1 : 0) : 0;
    const auto crow = [&](long long j) { return (size_t) (ring > 0 ? j % ring : j); };

    const auto fetch = [&](int buf, long long k0) {
        for (int i = threadIdx.x; i < KEYS * DIM / 8; i += INK_MMA_THREADS) {
            const int r = i / (DIM / 8), c = 8 * (i % (DIM / 8));
            const long long key = k0 + r;
            const bool whole = key >= lo && key < hi;   // the others are zeros, which no weight reads
            const size_t cr = crow(key), at = cr * krow + (size_t) hkv * DIM + c;
            const size_t sb = cr * srow + (size_t) hkv * (DIM >> 5) + (c >> 5);
            const half* sp = Q8 ? reinterpret_cast<const half*>(ks) + sb : nullptr;
            const half* vp = Q8 ? reinterpret_cast<const half*>(vs) + sb : nullptr;
            ink_tile<Q8>(&kb[buf][r][c], kc, at, sp, whole);
            ink_tile<Q8>(&vb[buf][r][c], vc, at, vp, whole);
        }
        ink_commit();
    };
    const long long first = lo / KEYS * KEYS;
    fetch(0, first);
    int buf = 0;
    // the sink's logit is zero and it carries no value, so a row starts there: largest 0, denominator 1
    float m[2] = {0.0f, 0.0f}, l[2] = {1.0f, 1.0f}, o[16][4];
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    for (long long k0 = first; k0 < hi; k0 += KEYS, buf ^= 1) {
        __syncthreads();   // the buffer the last tile was worked on in is free
        if (k0 + KEYS < hi) {
            fetch(buf ^ 1, k0 + KEYS);
            ink_wait<1>();
        } else {
            ink_wait<0>();
        }
        __syncthreads();
        float s[8][4];
        for (int n = 0; n < 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            for (int k = 0; k < 8; k += 2) {
                uint32_t b[4];   // keys 8 n.., values 16 k.. and 16 (k + 1)..
                ink_ldmatrix(b, &kb[buf][8 * n + lane % 8][16 * k + 8 * (lane / 8)], false);
                ink_mma(s[n], qa[k], b);
                ink_mma(s[n], qa[k + 1], b + 2);
            }
        }
        // the bias by distance and the mask, a row's own position being the one its keys end at
        for (int i = 0; i < 2; ++i) {
            const int r = i == 0 ? ra : rb;
            const long long own = pos + r;
            float mx = -INFINITY;
            for (int n = 0; n < 8; ++n)
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    const long long key = k0 + 8 * n + 2 * tig + j % 2;
                    if (r >= rows || key > own || (window > 0 && key <= own - window)) {
                        s[n][j] = -INFINITY;
                    } else {
                        const long long d = own - key;
                        s[n][j] += 1.44269504f *
                                   kqb[((size_t) r * n_head + h) * n_keys + (size_t) (d < n_keys ? d : n_keys - 1)];
                    }
                    mx = fmaxf(mx, s[n][j]);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float now = fmaxf(m[i], mx), base = now == -INFINITY ? 0.0f : now;
            const float f = exp2f(m[i] - base);
            m[i] = now;
            l[i] *= f;
            for (int n = 0; n < 16; ++n) {
                o[n][2 * i] *= f;
                o[n][2 * i + 1] *= f;
            }
            for (int n = 0; n < 8; ++n)
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    s[n][j] = exp2f(s[n][j] - base);
                    l[i] += s[n][j];
                }
        }
        for (int k = 0; k < 4; ++k) {   // keys 16 k..: the weights as an A fragment
            const uint32_t pa[4] = {ink_half2(s[2 * k][0], s[2 * k][1]), ink_half2(s[2 * k][2], s[2 * k][3]),
                                    ink_half2(s[2 * k + 1][0], s[2 * k + 1][1]),
                                    ink_half2(s[2 * k + 1][2], s[2 * k + 1][3])};
            for (int n = 0; n < 16; n += 2) {
                uint32_t b[4];   // keys 16 k.., values 8 n.. and 8 (n + 1)..
                ink_ldmatrix(b, &vb[buf][16 * k + lane % 8 + 8 * (lane / 8 % 2)][8 * n + 8 * (lane / 16)], true);
                ink_mma(o[n], pa, b);
                ink_mma(o[n + 1], pa, b + 2);
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        const int r = i == 0 ? ra : rb;
        if (r >= rows) continue;
        float* y = out + ((size_t) r * n_head + h) * DIM + 2 * tig;
        for (int n = 0; n < 16; ++n)
            *reinterpret_cast<float2*>(y + 8 * n) = make_float2(o[n][2 * i] / l[i], o[n][2 * i + 1] / l[i]);
    }
#else
    (void) q, (void) kc, (void) ks, (void) vc, (void) vs, (void) kqb, (void) tau, (void) pos, (void) rows,
        (void) out, (void) n_head, (void) kv_heads, (void) scale, (void) window, (void) n_keys, (void) ring;
#endif
}
#endif

}  // namespace

template <bool Q8>
void ink_attend_rel_mma_launch(const float* q, const void* kc, const uint16_t* ks, const void* vc,
                               const uint16_t* vs, const float* kqb, const float* tau, int64_t pos, int rows,
                               float* out, int n_head, int kv_heads, float scale, int window, int n_keys, int ring,
                               const long long* pos_at, void* stream) {
    if (rows <= 0) return;
#if INK_MMA
    static bool room = false;
    if (!room) {
        cudaFuncSetAttribute((const void*) ink_attend_rel_mma_kernel<Q8>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, INK_MMA_SMEM);
        room = true;
    }
    const unsigned blocks = (unsigned) ((rows + INK_MMA_ROWS - 1) / INK_MMA_ROWS);
    ink_attend_rel_mma_kernel<Q8><<<dim3((unsigned) n_head, blocks), INK_MMA_THREADS, INK_MMA_SMEM, cs(stream)>>>(
        q, kc, ks, vc, vs, kqb, tau, pos, rows, out, n_head, kv_heads, scale, window, n_keys, ring, pos_at);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_attend_rel mma: ") + cudaGetErrorString(e));
#else
    (void) q, (void) kc, (void) ks, (void) vc, (void) vs, (void) kqb, (void) tau, (void) pos, (void) rows,
        (void) out, (void) n_head, (void) kv_heads, (void) scale, (void) window, (void) n_keys, (void) ring,
        (void) pos_at, (void) stream;   // a card without the tensor cores' steps never launches it
#endif
}

void ink_rel_table(const float* bias, int n_rel, const float* rcur, const float* sinks, const float* tau,
                   float* kqb, int rows, int n_head, int n_keys, void* stream) {
    const int64_t n = (int64_t) rows * n_head * n_keys;
    if (n <= 0) return;
    ink_rel_table_kernel<<<grid((size_t) n, INK_THREADS), INK_THREADS, 0, cs(stream)>>>(bias, n_rel, rcur, sinks, tau,
                                                                                   kqb, rows, n_head, n_keys);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_rel_table: ") + cudaGetErrorString(e));
}

void ink_attend_rel(const float* q, const uint16_t* kc, const uint16_t* vc, const float* kqb, const float* tau,
                    int64_t pos, int rows, float* out, int dim, int n_head, int kv_heads, float scale,
                    int window, int n_keys, int ring, void* stream) {
    if (rows <= 0) return;
    // A prompt's rows are many: the tensor cores carry both products, and the scalar kernel the rest.
    if (dim == INK_MMA_DIM && rows >= INK_MMA_MIN && ink_mma_on()) {
        ink_attend_rel_mma_launch<false>(q, kc, nullptr, vc, nullptr, kqb, tau, pos, rows, out, n_head, kv_heads,
                                          scale, window, n_keys, ring, nullptr, stream);
        return;
    }
    // one accumulator a thread holds its dimension of the running sum in, so a dimension takes a thread
    if (dim <= 0 || dim > INK_THREADS)
        throw std::runtime_error("ink_attend_rel: the head dim must be 1 to " + std::to_string(INK_THREADS));
    ink_attend_rel_kernel<<<dim3((unsigned) n_head, (unsigned) ((rows + INK_ATTEND_ROWS - 1) / INK_ATTEND_ROWS)),
                              INK_THREADS, 0, cs(stream)>>>(
        q, (const half*) kc, (const half*) vc, kqb, tau, pos, rows, out, dim, n_head, kv_heads, scale, window, n_keys,
        ring, nullptr);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_attend_rel: ") + cudaGetErrorString(e));
}

void ink_attend_rel_at(const float* q, const uint16_t* kc, const uint16_t* vc, const float* kqb, const float* tau,
                        const long long* pos_at, int rows, float* out, int dim, int n_head, int kv_heads, float scale,
                        int window, int n_keys, int ring, void* stream) {
    if (rows <= 0) return;
    if (dim == INK_MMA_DIM && rows >= INK_MMA_MIN && ink_mma_on()) {
        ink_attend_rel_mma_launch<false>(q, kc, nullptr, vc, nullptr, kqb, tau, 0, rows, out, n_head, kv_heads,
                                          scale, window, n_keys, ring, pos_at, stream);
        return;
    }
    if (dim <= 0 || dim > INK_THREADS)
        throw std::runtime_error("ink_attend_rel_at: the head dim must be 1 to " + std::to_string(INK_THREADS));
    ink_attend_rel_kernel<<<dim3((unsigned) n_head, (unsigned) ((rows + INK_ATTEND_ROWS - 1) / INK_ATTEND_ROWS)),
                              INK_THREADS, 0, cs(stream)>>>(
        q, (const half*) kc, (const half*) vc, kqb, tau, 0, rows, out, dim, n_head, kv_heads, scale, window, n_keys,
        ring, pos_at);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_attend_rel_at: ") + cudaGetErrorString(e));
}

void ink_attend_rel_q8(const float* q, const int8_t* kc, const uint16_t* ks, const int8_t* vc, const uint16_t* vs,
                       const float* kqb, const float* tau, int64_t pos, int rows, float* out, int dim, int n_head,
                       int kv_heads, float scale, int window, int n_keys, int ring, void* stream) {
    if (rows <= 0) return;
    if (dim == INK_MMA_DIM && rows >= INK_MMA_MIN && ink_mma_on()) {
        ink_attend_rel_mma_launch<true>(q, kc, ks, vc, vs, kqb, tau, pos, rows, out, n_head, kv_heads, scale,
                                         window, n_keys, ring, nullptr, stream);
        return;
    }
    if (dim <= 0 || dim > INK_THREADS || (dim % 32) != 0)
        throw std::runtime_error("ink_attend_rel_q8: head dim must be a multiple of 32 in 32.." +
                                 std::to_string(INK_THREADS));
    ink_attend_rel_q8_kernel<<<dim3((unsigned) n_head, (unsigned) ((rows + INK_ATTEND_ROWS - 1) / INK_ATTEND_ROWS)),
                                 INK_THREADS, 0, cs(stream)>>>(
        q, kc, (const half*) ks, vc, (const half*) vs, kqb, tau, pos, rows, out, dim, n_head, kv_heads, scale, window,
        n_keys, ring, nullptr);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_attend_rel_q8: ") + cudaGetErrorString(e));
}

void ink_attend_rel_q8_at(const float* q, const int8_t* kc, const uint16_t* ks, const int8_t* vc,
                           const uint16_t* vs, const float* kqb, const float* tau, const long long* pos_at, int rows,
                           float* out, int dim, int n_head, int kv_heads, float scale, int window, int n_keys,
                           int ring, void* stream) {
    if (rows <= 0) return;
    if (dim == INK_MMA_DIM && rows >= INK_MMA_MIN && ink_mma_on()) {
        ink_attend_rel_mma_launch<true>(q, kc, ks, vc, vs, kqb, tau, 0, rows, out, n_head, kv_heads, scale,
                                         window, n_keys, ring, pos_at, stream);
        return;
    }
    if (dim <= 0 || dim > INK_THREADS || (dim % 32) != 0)
        throw std::runtime_error("ink_attend_rel_q8_at: head dim must be a multiple of 32 in 32.." +
                                 std::to_string(INK_THREADS));
    ink_attend_rel_q8_kernel<<<dim3((unsigned) n_head, (unsigned) ((rows + INK_ATTEND_ROWS - 1) / INK_ATTEND_ROWS)),
                                 INK_THREADS, 0, cs(stream)>>>(
        q, kc, (const half*) ks, vc, (const half*) vs, kqb, tau, 0, rows, out, dim, n_head, kv_heads, scale, window,
        n_keys, ring, pos_at);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_attend_rel_q8_at: ") + cudaGetErrorString(e));
}

void ink_attend_rel_split(const float* q, const uint16_t* kc, const uint16_t* vc, const float* kqb, const float* tau,
                          int64_t pos, float* out, float* part_out, float* part_ml, int dim, int n_head, int kv_heads,
                          float scale, int window, int n_keys, int ring, int splits, void* stream) {
    if (dim <= 0 || dim > INK_THREADS) throw std::runtime_error("ink_attend_rel_split: the head dim must be 1 to 256");
    if (splits <= 0) throw std::runtime_error("ink_attend_rel_split: splits must be positive");
    ink_attend_rel_split_kernel<<<dim3((unsigned) n_head, (unsigned) splits), INK_THREADS, 0, cs(stream)>>>(
        q, (const half*) kc, (const half*) vc, kqb, tau, pos, splits, part_out, part_ml, dim, n_head, kv_heads, scale,
        n_keys, ring, nullptr);
    ink_attend_split_reduce_kernel<<<dim3((unsigned) n_head, 1u), 128, 0, cs(stream)>>>(part_out, part_ml, out, dim,
                                                                                     n_head, splits);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_attend_rel_split: ") + cudaGetErrorString(e));
}

void ink_attend_rel_split_at(const float* q, const uint16_t* kc, const uint16_t* vc, const float* kqb,
                              const float* tau, const long long* pos_at, float* out, float* part_out,
                              float* part_ml, int dim, int n_head, int kv_heads, float scale, int window,
                              int n_keys, int ring, int splits, void* stream) {
    if (dim <= 0 || dim > INK_THREADS)
        throw std::runtime_error("ink_attend_rel_split_at: the head dim must be 1 to 256");
    if (splits <= 0) throw std::runtime_error("ink_attend_rel_split_at: splits must be positive");
    ink_attend_rel_split_kernel<<<dim3((unsigned) n_head, (unsigned) splits), INK_THREADS, 0, cs(stream)>>>(
        q, (const half*) kc, (const half*) vc, kqb, tau, 0, splits, part_out, part_ml, dim, n_head, kv_heads, scale,
        n_keys, ring, pos_at);
    ink_attend_split_reduce_kernel<<<dim3((unsigned) n_head, 1u), 128, 0, cs(stream)>>>(part_out, part_ml, out, dim,
                                                                                     n_head, splits);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_attend_rel_split_at: ") + cudaGetErrorString(e));
}

void ink_attend_rel_q8_split(const float* q, const int8_t* kc, const uint16_t* ks, const int8_t* vc,
                             const uint16_t* vs, const float* kqb, const float* tau, int64_t pos, float* out,
                             float* part_out, float* part_ml, int dim, int n_head, int kv_heads, float scale,
                             int window, int n_keys, int ring, int splits, void* stream) {
    if (dim <= 0 || dim > INK_THREADS || (dim % 32) != 0)
        throw std::runtime_error("ink_attend_rel_q8_split: head dim must be a multiple of 32 in 32..256");
    if (splits <= 0) throw std::runtime_error("ink_attend_rel_q8_split: splits must be positive");
    ink_attend_rel_q8_split_kernel<<<dim3((unsigned) n_head, (unsigned) splits), INK_THREADS, 0, cs(stream)>>>(
        q, kc, (const half*) ks, vc, (const half*) vs, kqb, tau, pos, splits, part_out, part_ml, dim, n_head, kv_heads,
        scale, n_keys, ring, nullptr);
    ink_attend_split_reduce_kernel<<<dim3((unsigned) n_head, 1u), 128, 0, cs(stream)>>>(part_out, part_ml, out, dim,
                                                                                     n_head, splits);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_attend_rel_q8_split: ") + cudaGetErrorString(e));
}

void ink_attend_rel_q8_split_at(const float* q, const int8_t* kc, const uint16_t* ks, const int8_t* vc,
                                 const uint16_t* vs, const float* kqb, const float* tau,
                                 const long long* pos_at, float* out, float* part_out, float* part_ml, int dim,
                                 int n_head, int kv_heads, float scale, int window, int n_keys, int ring,
                                 int splits, void* stream) {
    if (dim <= 0 || dim > INK_THREADS || (dim % 32) != 0)
        throw std::runtime_error("ink_attend_rel_q8_split_at: head dim must be a multiple of 32 in 32..256");
    if (splits <= 0) throw std::runtime_error("ink_attend_rel_q8_split_at: splits must be positive");
    ink_attend_rel_q8_split_kernel<<<dim3((unsigned) n_head, (unsigned) splits), INK_THREADS, 0, cs(stream)>>>(
        q, kc, (const half*) ks, vc, (const half*) vs, kqb, tau, 0, splits, part_out, part_ml, dim, n_head, kv_heads,
        scale, n_keys, ring, pos_at);
    ink_attend_split_reduce_kernel<<<dim3((unsigned) n_head, 1u), 128, 0, cs(stream)>>>(part_out, part_ml, out, dim,
                                                                                     n_head, splits);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_attend_rel_q8_split_at: ") + cudaGetErrorString(e));
}



void ink_depth_mix(const float* states, const float* query, float* out, int n_embd, int n_states, int n_q,
                   int rows, void* stream) {
    if (rows <= 0 || n_states <= 0) return;
    ink_depth_mix_kernel<<<dim3((unsigned) rows, (unsigned) n_q), INK_THREADS, 0, cs(stream)>>>(
        states, query, out, n_embd, n_states, n_q, rows);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_depth_mix: ") + cudaGetErrorString(e));
}

void ink_control(float* x, const float* msg, const float* down, const float* gain, const float* forget, int n_embd,
                 int rank, int rows, float* gate, void* stream) {
    if (rows <= 0) return;
    ink_control_kernel<<<rows, INK_THREADS, 0, cs(stream)>>>(x, msg, down, gain, forget, n_embd, rank, rows, gate);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_control: ") + cudaGetErrorString(e));
}

void ink_bank_states(const float* bank, const float* cur, float* out, int n_embd, int n_bank, int bank_stride,
                     long long first, int rows, void* stream) {
    const int64_t n = (int64_t) rows * (n_bank + 1) * n_embd;
    if (n <= 0) return;
    ink_bank_states_kernel<<<grid((size_t) n, INK_THREADS), INK_THREADS, 0, cs(stream)>>>(
        bank, cur, out, n_embd, n_bank, bank_stride, first, rows);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_bank_states: ") + cudaGetErrorString(e));
}

void ink_tau(float* tau, int64_t pos, int floor, float alpha, int rows, void* stream) {
    if (rows <= 0) return;
    ink_tau_kernel<<<grid((size_t) rows, INK_THREADS), INK_THREADS, 0, cs(stream)>>>(tau, pos, floor, alpha, rows,
                                                                                 nullptr);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_tau: ") + cudaGetErrorString(e));
}

void ink_tau_at(float* tau, const long long* pos_at, int floor, float alpha, int rows, void* stream) {
    if (rows <= 0) return;
    ink_tau_kernel<<<grid((size_t) rows, INK_THREADS), INK_THREADS, 0, cs(stream)>>>(tau, 0, floor, alpha, rows,
                                                                                 pos_at);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_tau_at: ") + cudaGetErrorString(e));
}

void ink_finish_logits(float* logits, int n_vocab, int n_unpadded, float cap, int rows, void* stream) {
    const int64_t n = (int64_t) rows * n_vocab;
    if (n <= 0) return;
    ink_finish_logits_kernel<<<grid((size_t) n, INK_THREADS), INK_THREADS, 0, cs(stream)>>>(logits, n_vocab, n_unpadded,
                                                                                          cap, rows);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("ink_finish_logits: ") + cudaGetErrorString(e));
}

}  // namespace strata::poly
