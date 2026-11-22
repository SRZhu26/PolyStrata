# Reference copies

The two files beside this one are **not** part of PolyStrata. They are verbatim copies of llama.cpp's
Inkling 2 graph, kept here so that `src/poly/inkling2_kernels.cu`, `src/poly/inkling2.cpp` and
`tools/ref_core.py` can name a single, stable source for every formula they implement — the clip, situ, the
short convolution, the relative-position bias and the attention it feeds, the depth attention, the log-scaled
position, the loop control and the softcapped logits.

| file | upstream |
|---|---|
| `inkling2.cpp` | `ggml-org/llama.cpp` — `src/models/inkling2.cpp` |
| `inkling2-loop.cpp` | `ggml-org/llama.cpp` — `src/models/inkling2-loop.cpp` |

They are unmodified copies, used only as documentation of the arithmetic. PolyStrata does not compile them
and does not link against them.

## Licence

llama.cpp is MIT licensed, and these copies are redistributed under the same terms:

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
