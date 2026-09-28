# realesrgan

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).

Real-ESRGAN upscaling in one self-contained binary: feed it a small or low-quality
photo and get back the same picture at two or four times the resolution, with
edges and texture reconstructed. No Python, PyTorch, ONNX Runtime, or CUDA
toolkit needed.

```sh
realesrgan -m RealESRGAN_x4plus.safetensors -i photo.png -o photo_4x.png
```

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels (a Winograd 3x3 convolution that does the same work in
  4/9ths of the multiplies). The GPU is used when the CUDA driver can be brought
  up and the CPU path when it cannot, so one binary covers a machine with no
  NVIDIA driver at all; `--device cpu` selects the CPU path explicitly and
  `--gpu` refuses to fall back.
* 1.4 MiB binary, statically linked except `libc` and `libgcc_s`;
  `libcuda.so.1` is `dlopen`ed, so no driver is required on disk.
* Four converted checkpoints ship with the release; the engine takes its scale
  and geometry from the checkpoint header, so `-m` is the whole choice.

Both backends reproduce the upstream PyTorch implementation's output to within
one level of 255 on a handful of values per image, none off by more.

## Download

Prebuilt binary and the converted checkpoints are attached to the
[release](https://github.com/jacobsparts/realesrgan-rs/releases).

| asset | what it is |
|---|---|
| `realesrgan-linux-x86_64` | the engine: x86-64 Linux with glibc >= 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); the GPU path needs a compute capability 6.1+ GPU, and `--device cpu` runs the pure-Rust path anywhere |
| `RealESRGAN_x4plus.safetensors` | x4 upscaling, the usual choice for photographs |
| `RealESRGAN_x2plus.safetensors` | x2 upscaling |
| `RealESRNet_x4plus.safetensors` | x4 without GAN sharpening (smoother, more faithful) |
| `4x_RealisticRescaler_100000_G.safetensors` | community x4 checkpoint trained on JPEG/BC1-degraded textures |

```sh
chmod +x realesrgan-linux-x86_64      # downloads do not carry the executable bit
./realesrgan-linux-x86_64 -m RealESRGAN_x4plus.safetensors -i photo.png -o photo_4x.png
```

The checkpoints are ~64 MiB each, converted from the authors' released `.pth`
files; `python3 tools/convert.py <file>.pth <file>.safetensors` converts any
checkpoint of the same architecture.

## Usage

```sh
realesrgan -m RealESRGAN_x4plus.safetensors -i in.png -o out.png
realesrgan -m RealESRGAN_x2plus.safetensors -i in.png -o out.png --device cpu
realesrgan -m RealESRGAN_x4plus.safetensors -i huge.png -o out.png --tile 256 --tile-pad 64

# pipeline use - both streams default to stdin/stdout, and `-` names them too
convert in.png -resize 200% png:- | realesrgan -m RealESRGAN_x4plus.safetensors | display -
```

| flag | meaning |
|---|---|
| `-m, --model` | converted `.safetensors` checkpoint |
| `-i, --input` / `-o, --output` | PNG in / PNG out (8- or 16-bit in, 8-bit RGB out); `-` or omitted means stdin/stdout |
| `--device` | `gpu` or `cpu` (default: `gpu` when the CUDA driver can be brought up, `cpu` otherwise) |
| `--cpu` / `--gpu` | shorthands for the two, and `--gpu` refuses to fall back |
| `--tile` | process in tiles of this many input pixels; `0` = whole image |
| `--tile-pad` | context added around each tile, in input pixels (default 10) |
| `--outscale` | resize the result to this final scale (separable Lanczos-4) |
| `-q, --quiet` | suppress progress output |

## Tiling

`--tile` exists so an image whose activations do not fit in VRAM can still be
processed: each tile is cropped with `--tile-pad` pixels of context, run, and
only its centre is pasted into the output. Tiling is a memory/speed tradeoff,
not an exact mode - a tile boundary cuts the network's receptive field, which in
RRDBNet is large (23 blocks deep with a global skip), so a pad that is too small
shows as seams. Measured on a 256x256 input at `--tile 128`: pad 16 leaves
values off by up to 12 levels; pad 64 matches the untiled run exactly. Use 64 or
more when a tiled result has to match an untiled one.

## Licence and attribution

The Rust and CUDA code here is MIT licensed (see `LICENSE`). The converted
checkpoints are not covered by it: the three RealESRGAN checkpoints carry the
upstream Real-ESRGAN BSD 3-Clause licence (`MODEL_LICENSE-Real-ESRGAN.txt`), and
the community 4x_RealisticRescaler by Mutin Choler is WTFPL
(`MODEL_LICENSE-RealisticRescaler.txt`). Upstream:
[xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
