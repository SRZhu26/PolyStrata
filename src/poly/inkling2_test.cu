// src/poly/inkling2_test.cu - Inkling 2 Mini's kernels against a double-precision reference of the formulas
// in docs/models/reference/inkling2.cpp: the clip, situ, the short convolution and its state, the relative-position
// bias, the attention it feeds, the depth attention, the bank gather, the log-scaled position and the loop control.
//
// usage: poly_inkling2_test   (exit 0: every kernel within 1e-4 of the reference, relative to its size)
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace strata::poly;

constexpr int kDim = 32, kHead = 8, kKv = 2, kRel = 16, kK = 4, kRank = 8;
constexpr int kRows = 6, kKeys = 24, kWindow = 16, kRing = 20, kStates = 5;

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> download(const float* d, size_t n) {
    std::vector<float> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

std::vector<uint16_t> halves(const std::vector<float>& f) {
    std::vector<uint16_t> h(f.size());
    for (size_t i = 0; i < f.size(); ++i) h[i] = __half_as_ushort(__float2half(f[i]));
    return h;
}

int bad = 0;

// The largest difference of the kernel's result from the reference, relative to the reference's largest value.
void report(const char* what, const std::vector<float>& got, const std::vector<float>& want) {
    float most = 0.f, big = 0.f;
    for (size_t i = 0; i < got.size(); ++i) {
        most = std::max(most, std::fabs(got[i] - want[i]));
        big = std::max(big, std::fabs(want[i]));
    }
    const bool ok = most <= 1e-4f * std::max(1.f, big);
    if (!ok) ++bad;
    std::printf("%s %-12s %.3e of %.3f\n", ok ? "ok  " : "FAIL", what, most, big);
}

float rms(const std::vector<float>& x, size_t at, int n) {
    double ss = 0;
    for (int e = 0; e < n; ++e) ss += (double) x[at + e] * x[at + e];
    return (float) (1.0 / std::sqrt(ss / n + 1e-6));
}

// ggml's clip: the per-channel scale, then an L2 norm of at most `threshold`
void check_clip(const std::vector<float>& x0, const std::vector<float>* scale, int heads, int rows) {
    const float threshold = 22.6274166f;
    std::vector<float> x = x0, want(x0.size());
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < heads; ++h) {
            const size_t o = (size_t(t) * heads + h) * kDim;
            auto s = [&](int i) { return scale ? (*scale)[size_t(h) * kDim + i] : 1.f; };
            double ss = 0;
            for (int i = 0; i < kDim; ++i) ss += (double) (x[o + i] * s(i)) * (x[o + i] * s(i));
            const double norm = std::sqrt(ss) + 1e-6;
            const double f = std::min(std::max(threshold / norm, 0.0), 1.0);
            for (int i = 0; i < kDim; ++i) want[o + i] = (float) (x[o + i] * s(i) * f);
        }
    float* dx = upload(x0);
    const float* ds = scale ? upload(*scale) : nullptr;
    ink_clip_l2(dx, ds, kDim, heads, rows, threshold, nullptr);
    report("clip_l2", download(dx, dx ? x0.size() : 0), want);
    cudaFree(dx);
    if (ds) cudaFree((void*) ds);
}

void check_situ(int n) {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 3.f);
    std::vector<float> g(n), u(n);
    for (float& v : g) v = 4.f * nd(rng);
    for (float& v : u) v = 8.f * nd(rng);
    std::vector<float> want(n);
    for (int i = 0; i < n; ++i) {
        const double gate = 4.0 * std::tanh(g[i] / 4.0) / (1.0 + std::exp(-(double) g[i]));
        want[i] = (float) (gate * (25.0 * std::tanh(u[i] / 25.0)));
    }
    float* dg = upload(g);
    float* du = upload(u);
    float* do_ = nullptr;
    cudaMalloc(&do_, size_t(n) * sizeof(float));
    ink_situ(dg, du, do_, n, 1, 4.f, 25.f, nullptr);
    report("situ", download(do_, n), want);
    cudaFree(dg), cudaFree(du), cudaFree(do_);
}

// The short convolution, its state and the input-dependent taps of a full-attention layer
void check_conv(int channels, int rows, bool dyn, int n_head) {
    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int past = kK - 1;
    std::vector<float> x(size_t(rows) * channels), st(size_t(past) * channels);
    for (float& v : x) v = nd(rng);
    for (float& v : st) v = nd(rng);
    std::vector<float> w(size_t(channels) * kK), dy(dyn ? size_t(rows) * kK * n_head : 0);
    for (float& v : w) v = 0.3f * nd(rng);
    for (float& v : dy) v = 0.3f * nd(rng);

    std::vector<float> want(size_t(rows) * channels);
    const double scale = 1.0 / std::sqrt((dyn ? 2.0 : 1.0) * kK + 1.0);
    auto ext = [&](int j, int c) -> double {   // row of the call, or of the state when negative
        return j < 0 ? st[size_t(j + past) * channels + c] : x[size_t(j) * channels + c];
    };
    for (int t = 0; t < rows; ++t)
        for (int c = 0; c < channels; ++c) {
            double acc = 0;
            for (int i = 0; i < kK; ++i) acc += (double) w[size_t(c) * kK + i] * ext(t + i - past, c);
            if (dyn) {
                // The projection's row for a token is n_conv taps per head, head after head: the reference
                // reshapes its [n_conv*n_heads, n_tokens] output to [n_conv, n_heads, n_tokens], so the tap
                // of (head h, lag l) sits at h*n_conv + l.
                const int h = c / (channels / n_head);
                for (int i = 0; i < kK; ++i)
                    acc += ext(t + i - past, c) * dy[(size_t(t) * n_head + h) * kK + (kK - 1 - i)];
            }
            want[size_t(t) * channels + c] = (float) (((double) x[size_t(t) * channels + c] + acc) * scale);
        }
    std::vector<float> st_want(size_t(past) * channels);
    for (int s = 0; s < past; ++s)
        for (int c = 0; c < channels; ++c) {
            const int j = rows + s - past;
            st_want[size_t(s) * channels + c] = j < 0 ? st[size_t(j + past) * channels + c] : x[size_t(j) * channels + c];
        }

    float* dx = upload(x);
    float* dst = upload(st);
    const float* dw = upload(w);
    const float* dd = dyn ? upload(dy) : nullptr;
    float* do_ = nullptr;
    cudaMalloc(&do_, want.size() * sizeof(float));
    ink_conv(dx, dst, dw, dd, do_, channels, rows, kK, n_head, nullptr);
    report(dyn ? "conv dyn" : "conv", download(do_, want.size()), want);
    report("conv state", download(dst, st_want.size()), st_want);
    cudaFree(dx), cudaFree(dst), cudaFree((void*) dw), cudaFree(do_);
    if (dd) cudaFree((void*) dd);
}

// tau * sum_r bias[r][d] * rcur[token][head][r] - sink
void check_rel_table() {
    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> bias(size_t(kKeys) * kRel), rcur(size_t(kRows) * kHead * kRel), sinks(size_t(kRows) * kHead),
        tau(kRows);
    for (float& v : bias) v = 0.1f * nd(rng);
    for (float& v : rcur) v = nd(rng);
    for (float& v : sinks) v = nd(rng);
    for (float& v : tau) v = 1.f + 0.1f * std::fabs(nd(rng));
    std::vector<float> want(size_t(kRows) * kHead * kKeys);
    for (int t = 0; t < kRows; ++t)
        for (int h = 0; h < kHead; ++h)
            for (int d = 0; d < kKeys; ++d) {
                double s = 0;
                for (int r = 0; r < kRel; ++r) s += (double) bias[size_t(d) * kRel + r] * rcur[(size_t(t) * kHead + h) * kRel + r];
                want[(size_t(t) * kHead + h) * kKeys + d] = (float) (s * tau[t] - sinks[size_t(t) * kHead + h]);
            }
    const float* db = upload(bias);
    const float* dr = upload(rcur);
    const float* dk = upload(sinks);
    const float* dt = upload(tau);
    float* dq = nullptr;
    cudaMalloc(&dq, want.size() * sizeof(float));
    ink_rel_table(db, kRel, dr, dk, dt, dq, kRows, kHead, kKeys, nullptr);
    report("rel_table", download(dq, want.size()), want);
    cudaFree((void*) db), cudaFree((void*) dr), cudaFree((void*) dk), cudaFree((void*) dt), cudaFree(dq);
}

// The attention the bias feeds, over a full cache and over a window layer's ring
void check_attend(bool ring) {
    std::mt19937 rng(19);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int nh = kHead, hd = kDim, nkv = kKv, chan = nkv * hd;
    const long long pos = 40;
    const int window = ring ? kWindow : 0, n_keys = ring ? kWindow + 1 : pos + kRows + 1;
    std::vector<float> q(size_t(kRows) * nh * hd), kqb(size_t(kRows) * nh * n_keys), tau(kRows);
    for (float& v : q) v = 2.f * nd(rng);
    for (float& v : kqb) v = 0.3f * nd(rng);
    for (float& v : tau) v = 1.f + 0.1f * std::fabs(nd(rng));
    std::vector<float> k(size_t(pos + kRows) * chan), v(size_t(pos + kRows) * chan);
    for (float& x : k) x = nd(rng);
    for (float& x : v) x = nd(rng);
    const std::vector<uint16_t> kh = halves(k), vh = halves(v);
    // the cache holds fp16, so the reference must read the values the kernel actually sees, not the fp32 originals
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = __half2float(__ushort_as_half(kh[i]));
        v[i] = __half2float(__ushort_as_half(vh[i]));
    }
    const float scale = 1.f / std::sqrt((float) hd);
    std::vector<float> want(size_t(kRows) * nh * hd);
    auto row_of = [&](long long j) -> size_t { return size_t(ring ? j % kRing : j); };
    auto bias_of = [&](int t, int h, long long j) -> double {
        const long long d = pos + t - j;
        return kqb[(size_t(t) * nh + h) * n_keys + size_t(std::min<long long>(d, n_keys - 1))];
    };
    for (int t = 0; t < kRows; ++t) {
        const long long p = pos + t;
        const long long lo = window > 0 && p >= window ? p - window + 1 : 0;
        for (int h = 0; h < nh; ++h) {
            const int hkv = h / (nh / nkv);
            std::vector<double> logit;
            for (long long j = lo; j <= p; ++j) {
                double s = 0;
                for (int e = 0; e < hd; ++e) s += (double) q[(size_t(t) * nh + h) * hd + e] * k[row_of(j) * chan + size_t(hkv) * hd + e];
                logit.push_back(s * scale * tau[t] + bias_of(t, h, j));
            }
            double best = 0.0;   // the sink's own logit is zero
            for (double s : logit) best = std::max(best, s);
            double sum = std::exp(-best);
            for (double s : logit) sum += std::exp(s - best);
            for (int e = 0; e < hd; ++e) {
                double acc = 0;
                for (size_t j = 0; j < logit.size(); ++j)
                    acc += std::exp(logit[j] - best) / sum * v[row_of(lo + (long long) j) * chan + size_t(hkv) * hd + e];
                want[(size_t(t) * nh + h) * hd + e] = (float) acc;
            }
        }
    }
    const float* dq = upload(q);
    const uint16_t* dk = upload(kh);
    const uint16_t* dv = upload(vh);
    const float* db = upload(kqb);
    const float* dt = upload(tau);
    float* do_ = nullptr;
    cudaMalloc(&do_, want.size() * sizeof(float));
    ink_attend_rel(dq, dk, dv, db, dt, pos, kRows, do_, hd, nh, nkv, scale, window, n_keys, ring ? kRing : 0,
                    nullptr);
    report(ring ? "attend ring" : "attend", download(do_, want.size()), want);
    cudaFree((void*) dq), cudaFree((void*) dk), cudaFree((void*) dv), cudaFree((void*) db), cudaFree((void*) dt),
        cudaFree(do_);
}

// The depth attention: the bank newest first, then cur last, every state RMS-normed without a gain
void check_depth_mix(int n_states, int n_q) {
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> states(size_t(kRows) * n_states * kDim), query(size_t(n_q) * kDim);
    for (float& v : states) v = nd(rng);
    for (float& v : query) v = 0.5f * nd(rng);
    // query-major: a query's mix of every token is one contiguous run, which is the block every caller
    // hands in and reads back (mix_in_ holds n_q of them, msg_ four); a query of 1, the output block and
    // every window layer, has the reference's [n_embd, 1, n_tok] layout either way round
    std::vector<float> want(size_t(n_q) * kRows * kDim);
    const float inv = 1.f / std::sqrt((float) kDim);
    for (int t = 0; t < kRows; ++t) {
        std::vector<double> nrm(static_cast<size_t>(n_states)), score(static_cast<size_t>(n_states));
        for (int s = 0; s < n_states; ++s) {
            double ss = 0;
            for (int e = 0; e < kDim; ++e) ss += (double) states[(size_t(t) * n_states + s) * kDim + e] * states[(size_t(t) * n_states + s) * kDim + e];
            nrm[size_t(s)] = 1.0 / std::sqrt(ss / kDim + 1e-6);
        }
        for (int qi = 0; qi < n_q; ++qi) {
            for (int s = 0; s < n_states; ++s) {
                double dot = 0;
                for (int e = 0; e < kDim; ++e)
                    dot += (double) states[(size_t(t) * n_states + s) * kDim + e] * nrm[size_t(s)] * query[size_t(qi) * kDim + e];
                score[size_t(s)] = dot * inv;
            }
            double best = -1e300;
            for (double s : score) best = std::max(best, s);
            double sum = 0;
            for (double s : score) sum += std::exp(s - best);
            for (int e = 0; e < kDim; ++e) {
                double acc = 0;
                for (int s = 0; s < n_states; ++s)
                    acc += std::exp(score[size_t(s)] - best) / sum * states[(size_t(t) * n_states + s) * kDim + e] * nrm[size_t(s)];
                want[(size_t(qi) * kRows + t) * kDim + e] = (float) acc;
            }
        }
    }
    const float* ds = upload(states);
    const float* dq = upload(query);
    float* do_ = nullptr;
    cudaMalloc(&do_, want.size() * sizeof(float));
    ink_depth_mix(ds, dq, do_, kDim, n_states, n_q, kRows, nullptr);
    report(n_q > 1 ? "depth_mix 3" : "depth_mix", download(do_, want.size()), want);
    cudaFree((void*) ds), cudaFree((void*) dq), cudaFree(do_);
}

// The bank's blocks newest first and then cur last, which the reference's concat leaves last
void check_bank_states() {
    const int n_bank = 4, first = 1, rows = kRows - 1;
    const int stride = kRows * kDim;
    std::mt19937 rng(29);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> bank(size_t(n_bank) * stride), cur(size_t(kRows) * kDim);
    for (float& v : bank) v = nd(rng);
    for (float& v : cur) v = nd(rng);
    std::vector<float> want(size_t(rows) * (n_bank + 1) * kDim);
    for (int t = 0; t < rows; ++t)
        for (int s = 0; s <= n_bank; ++s)
            for (int e = 0; e < kDim; ++e) {
                const size_t at = size_t(first + t) * kDim + e;
                want[(size_t(t) * (n_bank + 1) + s) * kDim + e] =
                    s < n_bank ? bank[size_t(n_bank - 1 - s) * stride + at] : cur[at];
            }
    const float* db = upload(bank);
    const float* dc = upload(cur);
    float* do_ = nullptr;
    cudaMalloc(&do_, want.size() * sizeof(float));
    ink_bank_states(db, dc, do_, kDim, n_bank, stride, first, rows, nullptr);
    report("bank_states", download(do_, want.size()), want);
    cudaFree((void*) db), cudaFree((void*) dc), cudaFree(do_);
}

// 1 + alpha*log(max(1, (pos + row + 1)/floor))
void check_tau() {
    const long long pos = 9000;
    const int floor = 8192;
    const float alpha = 0.1f;
    std::vector<float> want(kRows);
    for (int r = 0; r < kRows; ++r)
        want[size_t(r)] = (float) (1.0 + alpha * std::log(std::max(1.0, (double) (pos + r + 1) / floor)));
    float* d = nullptr;
    cudaMalloc(&d, size_t(kRows) * sizeof(float));
    ink_tau(d, pos, floor, alpha, kRows, nullptr);
    report("tau", download(d, kRows), want);
    cudaFree(d);
}

// x = (1 - tanh(forget . gate)) x + tanh(gain . gate) message
void check_control() {
    const int n_embd = kDim;
    std::mt19937 rng(31);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(size_t(kRows) * n_embd), msg(x.size()), down(size_t(kRank) * 2 * n_embd),
        gain(size_t(n_embd) * kRank), forget(gain.size());
    for (float& v : x) v = nd(rng);
    for (float& v : msg) v = nd(rng);
    for (float& v : down) v = 0.1f * nd(rng);
    // gain and forget are {rank, n_embd} in the file, as create_tensor() declares them, so the value of
    // (rank r, embd e) sits at e*rank + r and not at r*n_embd + e.  ctrl_down is {2*n_embd, rank} and
    // keeps the other layout, which is why the two are indexed differently.
    for (int e = 0; e < n_embd; ++e)
        for (int r = 0; r < kRank; ++r) {
            gain[size_t(e) * kRank + r] = 0.1f * nd(rng);
            forget[size_t(e) * kRank + r] = 0.1f * nd(rng);
        }
    std::vector<float> want(x);
    for (int t = 0; t < kRows; ++t) {
        std::vector<double> cat(static_cast<size_t>(2 * n_embd)), g(static_cast<size_t>(kRank));
        for (int pass = 0; pass < 2; ++pass) {
            const std::vector<float>& src = pass == 0 ? x : msg;
            const float f = rms(src, size_t(t) * n_embd, n_embd);
            for (int e = 0; e < n_embd; ++e) cat[size_t(pass) * n_embd + e] = src[size_t(t) * n_embd + e] * f;
        }
        for (int r = 0; r < kRank; ++r) {
            double s = 0;
            for (int e = 0; e < 2 * n_embd; ++e) s += (double) down[size_t(r) * 2 * n_embd + e] * cat[size_t(e)];
            g[size_t(r)] = s;
        }
        for (int e = 0; e < n_embd; ++e) {
            double gi = 0, fi = 0;
            for (int r = 0; r < kRank; ++r) {
                gi += (double) gain[size_t(e) * kRank + r] * g[size_t(r)];
                fi += (double) forget[size_t(e) * kRank + r] * g[size_t(r)];
            }
            want[size_t(t) * n_embd + e] = (float) ((double) x[size_t(t) * n_embd + e] -
                                                   std::tanh(fi) * x[size_t(t) * n_embd + e] +
                                                   std::tanh(gi) * msg[size_t(t) * n_embd + e]);
        }
    }
    float* dx = upload(x);
    const float* dm = upload(msg);
    const float* dd = upload(down);
    const float* dg = upload(gain);
    const float* df = upload(forget);
    float* scratch = nullptr;
    cudaMalloc(&scratch, size_t(kRows) * (kRank + 2 * n_embd) * sizeof(float));
    ink_control(dx, dm, dd, dg, df, n_embd, kRank, kRows, scratch, nullptr);
    report("control", download(dx, want.size()), want);
    cudaFree(dx), cudaFree((void*) dm), cudaFree((void*) dd), cudaFree((void*) dg), cudaFree((void*) df),
        cudaFree(scratch);
}

}  // namespace

// A timing of the attention at the shape a 23,552-token prompt gives it: a batch of 1,024 rows, 24 heads
// over 8 kv heads, 128 values a head, the bias over the model's 1,024-position extent.  The window layers read
// their 1,024 keys, the full-context layers the whole extent.  `poly_inkling2_test bench` runs only this.
int bench() {
    const long long pos = 23552;
    const int rows = 1024, nh = 24, nkv = 8, hd = 128, chan = nkv * hd, n_keys = 1024;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q(size_t(rows) * nh * hd), tau(rows), kqb(size_t(rows) * nh * n_keys);
    for (float& v : q) v = 2.f * nd(rng);
    for (float& v : tau) v = 1.f + 0.1f * std::fabs(nd(rng));
    for (float& v : kqb) v = 0.3f * nd(rng);
    const int keys = int(pos) + rows;
    std::vector<float> k(size_t(keys) * chan), v(size_t(keys) * chan);
    for (float& x : k) x = nd(rng);
    for (float& x : v) x = nd(rng);
    const std::vector<uint16_t> kh = halves(k), vh = halves(v);
    std::vector<float> out(size_t(rows) * nh * hd);
    const float* dq = upload(q);
    const uint16_t* dk = upload(kh);
    const uint16_t* dv = upload(vh);
    const float* db = upload(kqb);
    const float* dt = upload(tau);
    float* do_ = nullptr;
    cudaMalloc(&do_, out.size() * sizeof(float));
    const float scale = 1.f / std::sqrt((float) hd);
    // a full-context layer attends every position before the row, a window layer the last 1,024 (its ring)
    for (int window : {0, 1024}) {
        const int ring = window ? 1024 : 0;
        for (int i = 0; i < 3; ++i)
            strata::poly::ink_attend_rel(dq, dk, dv, db, dt, pos, rows, do_, hd, nh, nkv, scale, window, n_keys,
                                        ring, nullptr);
        cudaDeviceSynchronize();
        cudaEvent_t a = nullptr, b = nullptr;
        cudaEventCreate(&a), cudaEventCreate(&b);
        cudaEventRecord(a);
        const int rep = 10;
        for (int i = 0; i < rep; ++i)
            strata::poly::ink_attend_rel(dq, dk, dv, db, dt, pos, rows, do_, hd, nh, nkv, scale, window, n_keys,
                                        ring, nullptr);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        ms /= rep;
        // the products the visit makes: every row's keys, and their two per value and key
        const double pairs = (double) rows * nh * (window ? window : (double) pos / 2 + rows / 2);
        std::printf("%-6s %8.2f ms  %6.1f GFLOP/s  %6.1f GMAC/s\n", window ? "window" : "full", ms,
                    2.0 * pairs * hd * 2.0 / (ms * 1e6), pairs * hd * 2.0 / (ms * 1e6));
        cudaEventDestroy(a), cudaEventDestroy(b);
    }
    cudaFree((void*) dq), cudaFree((void*) dk), cudaFree((void*) dv), cudaFree((void*) db), cudaFree((void*) dt),
        cudaFree(do_);
    return 0;
}

int main() {
    if (const char* m = std::getenv("STRATA_INK_BENCH")) {
        (void) m;
        return bench();
    }
    std::vector<float> scale(size_t(kHead) * kDim);
    for (size_t i = 0; i < scale.size(); ++i) scale[i] = 0.9f + 0.2f * float(i % 7) / 7.f;
    std::vector<float> x(size_t(kRows) * kHead * kDim);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 2.f);
    for (float& v : x) v = nd(rng);
    check_clip(x, &scale, kHead, kRows);
    check_clip(x, nullptr, kHead, kRows);   // no per-channel scale, as the keys take none

    check_situ(kRows * 8);
    check_conv(kKv * kDim, kRows, false, kKv);
    check_conv(kKv * kDim, kRows, true, kKv);
    check_conv(16, kRows, false, 1);
    check_rel_table();
    check_attend(false);
    check_attend(true);
    check_depth_mix(kStates, 1);
    check_depth_mix(kStates, 3);
    check_bank_states();
    check_tau();
    check_control();

    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
