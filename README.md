# realesrgan

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).
The family also includes [rmbg-rs](https://github.com/jacobsparts/rmbg-rs),
[locate-anything-rs](https://github.com/jacobsparts/locate-anything-rs) and
[lama-inpaint-rs](https://github.com/jacobsparts/lama-inpaint-rs); they
share the [lightgpu toolkit](https://github.com/jacobsparts/lightgpu).
[pixeldeck](https://github.com/jacobsparts/pixeldeck) is a local web app for
cleaning up product photos that drives all of these engines.

Real-ESRGAN (RRDBNet) upscaling as a single self-contained binary. No Python at
inference, no PyTorch, no ONNX Runtime, no CUDA toolkit needed to run it.

```
realesrgan --model RealESRGAN_x4plus.safetensors -i photo.png -o photo_4x.png
```

## Why this exists

The reference Real-ESRGAN implementation is a Python program. Getting an upscale
out of it means a Python environment, a PyTorch install of the right version for
the right CUDA, and a ~64 MB `.pth` pickle that only `torch.load` can read.

This is the same network, rewritten as an engine:

* **One self-contained binary**: 1.4 MB with the CUDA backend (0.94 MB with it
  compiled out), and 30 crates in the whole dependency tree. The direct
  dependencies are three small crates - `png`, `rayon` and `libc` - plus the
  toolkit below. The GPU backend `dlopen`s `libcuda.so.1` at run time, so nothing
  links against CUDA and no NVIDIA toolchain is needed to *run* it: the toolkit
  and `nvcc` are needed only to build the GPU backend.
* **Composes in a pipeline.** Input and output default to stdin and stdout, and
  `-` names either stream explicitly, so
  `convert in.png -resize 200% png:- | realesrgan -m RealESRGAN_x4plus.safetensors | display -`
  works without temporary files. All progress goes to stderr.
* **Weights are memory-mapped.** The checkpoint is converted once into the
  standard `.safetensors` container and read straight out of the mapping by the
  shared [`lightgpu`](https://github.com/jacobsparts/lightgpu) reader - the same
  one every engine in the family uses. No pickle, no deserialisation, no per-run
  parse: the architecture constants come from the file header and every tensor is
  shape-checked against them at load.
* **A CPU backend that is actually tuned.** `--device cpu` is the fallback for a
  machine with no GPU or driver - the same graph in plain Rust (no CUDA, no
  driver): avx2 kernels with output-channel blocking, 24 threads, about 6x faster
  than the reference on CPU.
* **Tiling for images that do not fit in VRAM**, with the fidelity cost
  documented rather than hidden (see below).
* **Byte-level agreement with the reference.** On the same input, both models
  reproduce PyTorch within a single 8-bit level - a handful of values out of
  thousands, each off by exactly one level and none by more.

## What makes it fast: lightgpu

Both backends sit on [lightgpu](https://github.com/jacobsparts/lightgpu), a CUDA toolkit written for this
family of engines. It is not a wrapper around cuDNN or cuBLAS - it is a small
library of hand-written kernels plus the driver plumbing to launch them, and it
is what this project contributes back to.

* **The 3x3 convolution is F(4x4,3x3) Winograd**, which computes the same result
  with 4/9ths of the multiplies. It was written here, measured against the
  fastest direct convolution the hardware can run, and promoted into the
  toolkit's shared kernel set, where any engine can call it.
* **Kernels are compiled per consumer.** A binary embeds only the kernels it
  calls: this engine's fatbins carry 10 of the toolkit's 48 kernels plus one of
  its own, in two separately loaded modules, so a name collision or a mis-filed
  kernel fails at build time rather than mid-inference.
* **Every kernel has a CPU twin** it is checked against on random data
  (`--cuda-selftest`), which is a development-time check of the arithmetic, not
  the correctness record: that is the golden fixtures and the PyTorch comparison
  under Accuracy.
* **Nothing to install to run it.** The driver bindings are `dlopen`ed at run
  time, so the binary links no CUDA library and the GPU backend works on any
  machine with a driver - no toolkit, no headers. `--no-default-features` builds
  a CPU-only binary with no CUDA dependency at build time either.

## Performance

The same job end to end: a fresh process reads a PNG from disk, upscales it, and
writes a PNG. That is the honest comparison, because it is what a user waits
for - process start, checkpoint load, image decode, inference, encode. The
reference is timed the same way, through its own Python entry point.

RealESRGAN_x4plus, 256x256 -> 1024x1024, three interleaved rounds:

| | GPU | CPU |
| --- | --- | --- |
| **this engine** | **0.88-0.90 s** | **6.8 s** (24 threads) |
| torch / Real-ESRGAN | 1.02-1.08 s | 41-44 s (24 threads) |

Both engines produce the same image (max difference one 8-bit level, none off by
more than one). The reference numbers include its Python start-up and checkpoint
load, which is most of its GPU time at this size; at larger images the gap in
inference throughput is the part that matters, and the Winograd convolution is
where it comes from.

## Build

Prebuilt binaries and converted models are attached to the
[GitHub release](https://github.com/jacobsparts/realesrgan-rs/releases/latest):

* `realesrgan-linux-x86_64` — CUDA-enabled binary; needs an NVIDIA driver to use
  the GPU and also supports `--device cpu`.
* `realesrgan-linux-x86_64-cpu-only` — CPU-only binary; no CUDA toolkit or NVIDIA
  driver is needed.
* `RealESRGAN_x4plus.safetensors`, `RealESRGAN_x2plus.safetensors`,
  `RealESRNet_x4plus.safetensors`, and `4x_RealisticRescaler_100000_G.safetensors`
  — converted model files ready for the binaries.

The model files are about 64 MiB each. They are converted from the official
upstream checkpoints and contain no Python pickle data; the engine reads them
as standard safetensors files.

To build from source:

```sh
cargo build --release                          # both backends (needs nvcc on PATH)
cargo build --release --no-default-features    # CPU only, no CUDA toolchain
```

The GPU backend needs `nvcc` to compile the kernels; the CPU-only build needs
nothing but a Rust toolchain.

## Get the weights

The release includes converted copies of the tested official models:

* [`RealESRGAN_x4plus.safetensors`](https://github.com/jacobsparts/realesrgan-rs/releases/latest/download/RealESRGAN_x4plus.safetensors)
* [`RealESRGAN_x2plus.safetensors`](https://github.com/jacobsparts/realesrgan-rs/releases/latest/download/RealESRGAN_x2plus.safetensors)
* [`RealESRNet_x4plus.safetensors`](https://github.com/jacobsparts/realesrgan-rs/releases/latest/download/RealESRNet_x4plus.safetensors)
* [`4x_RealisticRescaler_100000_G.safetensors`](https://github.com/jacobsparts/realesrgan-rs/releases/latest/download/4x_RealisticRescaler_100000_G.safetensors)

They were converted from the official
[`RealESRGAN_x4plus.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
and
[`RealESRGAN_x2plus.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth),
and
[`RealESRNet_x4plus.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth)
checkpoints. See the upstream
[model zoo](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md)
for other models.

The converted checkpoints remain subject to the upstream Real-ESRGAN BSD
3-Clause license; see [`MODEL_LICENSE-Real-ESRGAN.txt`](MODEL_LICENSE-Real-ESRGAN.txt).

The community-trained [4x RealisticRescaler](https://openmodeldb.info/models/4x-RealisticRescaler)
by Mutin Choler is a useful alternative with a different character. It was
trained for realistic low-resolution textures degraded by JPEG or BC1 and can
also work well on photographs, particularly outdoor scenes. It is licensed
under WTFPL; see
[`MODEL_LICENSE-RealisticRescaler.txt`](MODEL_LICENSE-RealisticRescaler.txt).

## Convert a checkpoint

```sh
python3 tools/convert.py RealESRGAN_x4plus.pth RealESRGAN_x4plus.safetensors
python3 tools/convert.py RealESRGAN_x2plus.pth RealESRGAN_x2plus.safetensors
```

The converter needs `torch` and `numpy`; the engine needs neither. It writes the
standard `.safetensors` container, so the result can be inspected with the usual
tooling, and records the architecture constants (`scale`, `num_feat`,
`num_block`, `num_grow_ch`, `in_ch`, `out_ch`) in the header, from which the
engine takes its geometry and validates every tensor shape at load. The model
scale is inferred from `conv_first`'s input channels (3 -> x4, 12 -> x2) unless
`--scale` is given.

## Run

```sh
realesrgan --model RealESRGAN_x4plus.safetensors -i in.png -o out.png
realesrgan --model RealESRGAN_x2plus.safetensors -i in.png -o out.png --device cpu
realesrgan --model RealESRGAN_x4plus.safetensors -i huge.png -o out.png --tile 256 --tile-pad 64

# pipeline use - both streams default to stdin/stdout, and `-` names them too
cat in.png | realesrgan --model RealESRGAN_x4plus.safetensors -q > out.png
convert in.png -resize 200% png:- | realesrgan -m RealESRGAN_x4plus.safetensors | display -
```

| flag | meaning |
| --- | --- |
| `-m, --model` | converted `.safetensors` checkpoint |
| `-i, --input` / `-o, --output` | PNG in / PNG out (8- or 16-bit in, 8-bit RGB out); `-` or omitted means stdin/stdout |
| `--device` | `gpu` or `cpu` (a CPU-only build defaults to `cpu`) |
| `--tile` | process in tiles of this many input pixels; `0` = whole image |
| `--tile-pad` | context added around each tile, in input pixels (default 10) |
| `--outscale` | resize the result to this final scale (separable Lanczos-4) |
| `-q, --quiet` | suppress progress output |

## Tiling

`--tile` exists so an image whose activations do not fit in VRAM can still be
processed. Each tile is cropped from the input with `--tile-pad` pixels of
context on every side, the network runs on that crop, and only the centre is
pasted into the output; at the image border the crop shrinks rather than
extending past the edge.

**Tiling is a memory/speed tradeoff, not an exact mode.** A tile boundary cuts
the network's receptive field, and RRDBNet's is large: 23 RRDB blocks deep with
a global skip, so a few pixels of pad are nowhere near enough context. Measured
on a 256x256 input at `--tile 128`, against the same run untiled:

| pad | max byte diff | pixels off by >1 |
| --- | --- | --- |
| 16 | 12 | 122,483 / 1,048,576 |
| 64 | 1 | 0 |

So if a tiled result has to match an untiled one, the pad must be comparable to
the network's receptive field - use 64 or more. The default of 10 matches the
reference implementation's `tile_pad`.

## Accuracy

Both models are checked against the reference PyTorch implementation, stage by
stage and end to end.

| model and size | result (vs the reference, same input) |
| --- | --- |
| x2plus, 24x24 -> 48x48 | max 1 level; 2 of 6,912 values differ, none by more |
| x4plus, 24x24 -> 96x96 | max 1 level; 5 of 27,648 values differ, none by more |
| x4plus, 256x256 -> 1024x1024 | max 1 level; 3,145,174 of 3,145,728 values identical, none off by more than 1 |

The residual differences are f32 rounding, not an algorithmic gap: the CPU and
GPU implementations use a real fused multiply-add where the reference's scalar
path does not, and the two agree to within a rounding step of each other at every
stage of the graph.
