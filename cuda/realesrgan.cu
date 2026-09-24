// Real-ESRGAN (RRDBNet): this project's own kernels.
//
// The production 3x3 convolution is NOT here - it is the toolkit's
// `lg_conv3x3_winograd`. What remains is the direct conv FALLBACK, which covers
// the shapes that op cannot take: a non-3x3 or non-stride-1 convolution, and a
// tensor smaller than one Winograd CTA (w < 16 or h < 8).
//
// Layout convention: NCHW, f32, contiguous. A conv weight is [c_out][c_in][3][3]
// exactly as PyTorch stores it, so no transposition happens on the load path.
//
// ACCUMULATION ORDER is a contract, not an implementation detail: the CPU twin
// in src/net.rs sums in the order ky, kx, ci, and this kernel keeps that order so
// the backends agree bit-for-bit on short reductions rather than merely to
// within rounding. There is no residual operand: the engine's residual is the
// whole-plane `lg_add_scaled`.

#include <cuda_runtime.h>

#define LA_DEVI static __device__ __forceinline__

// The network's activation slope. A local helper because the toolkit's `lg_lrelu`
// is a whole-plane kernel and cannot be called from device code.
LA_DEVI float la_lrelu(float v) {
    return v >= 0.0f ? v : 0.2f * v;
}

// 3x3, stride 1, pad 1, plus bias and optional LeakyReLU. `grid = (ceil(w/32),
// ceil(h/8))`, `block = (32,8,1)`; one thread walks all c_out for one pixel.
extern "C" __global__ void lg_conv3x3_res(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int act)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= wd || y >= h) return;
    const size_t plane = (size_t)h * wd;
    const size_t off = (size_t)y * wd + x;

    for (int oc = 0; oc < c_out; ++oc) {
        float acc = bias ? bias[oc] : 0.0f;
        const float *wp = w + (size_t)oc * c_in * 9;
        for (int ky = 0; ky < 3; ++ky) {
            const int iy = y + ky - 1;
            if (iy < 0 || iy >= h) continue;
            for (int kx = 0; kx < 3; ++kx) {
                const int ix = x + kx - 1;
                if (ix < 0 || ix >= wd) continue;
                const size_t ioff = (size_t)iy * wd + ix;
                const float *ipp = in + ioff;
                for (int ci = 0; ci < c_in; ++ci) {
                    acc += wp[(size_t)ci * 9 + ky * 3 + kx] * ipp[(size_t)ci * plane];
                }
            }
        }
        if (act) acc = la_lrelu(acc);
        out[(size_t)oc * plane + off] = acc;
    }
}


