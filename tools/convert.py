#!/usr/bin/env python3
"""Convert a Real-ESRGAN .pth checkpoint to a single-file .safetensors blob.

Real-ESRGAN ships `RealESRGAN_x4plus.pth` / `RealESRGAN_x2plus.pth` as
torch pickles holding `{'params_ema': {...}}` (some releases use `'params'`).
The Rust engine reads the RRDBNet weights straight out of a memory-mapped
safetensors file, so it needs neither torch nor pickle parsing at run time.

The converted file is a standard safetensors container:

  u64 header_len | header_len bytes of JSON | tensor payloads

with the usual per-tensor `dtype` / `shape` / `data_offsets` entries. The
header's `__metadata__` records what the engine cannot infer from the tensor
names alone:

  format        "realesrgan-safetensors-v1"
  scale         "4" or "2" (net upsampling factor)
  num_feat      "64"
  num_block     "23"
  num_grow_ch   "32"
  in_ch         "3"
  out_ch        "3"

Usage:

  python3 tools/convert.py RealESRGAN_x4plus.pth RealESRGAN_x4plus.safetensors
  python3 tools/convert.py --scale 2 in.pth out.safetensors

The scale is inferred from `conv_first.weight`'s input channels when not
given (3 -> x4, 12 -> x2), which is the same signal the reference model uses.
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

ALIGN = 8

# Tensor names that make up an RRDBNet, in the exact order the engine walks
# them. The list is generated from the architecture constants so the converter
# and the Rust side agree on a single canonical layout.
def rrdbnet_names(num_block=23):
    names = [("conv_first", ("weight", "bias"))]
    for b in range(num_block):
        for r in range(1, 4):
            for c in range(1, 6):
                names.append((f"body.{b}.rdb{r}.conv{c}", ("weight", "bias")))
    names.append(("conv_body", ("weight", "bias")))
    names.append(("conv_up1", ("weight", "bias")))
    names.append(("conv_up2", ("weight", "bias")))
    names.append(("conv_hr", ("weight", "bias")))
    names.append(("conv_last", ("weight", "bias")))
    return names


def expected_names(num_block=23):
    return [
        f"{base}.{suffix}"
        for base, suffixes in rrdbnet_names(num_block)
        for suffix in suffixes
    ]


def load_checkpoint(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("params_ema", "params"):
            if key in obj:
                return obj[key]
        # Some exports are already a bare state_dict.
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise SystemExit(
        f"{path}: expected a dict with 'params_ema' or 'params', got {type(obj)}"
    )


def infer_scale(state):
    key = "conv_first.weight"
    if key not in state:
        raise SystemExit(f"checkpoint is missing {key}")
    in_ch = int(state[key].shape[1])
    return {3: 4, 12: 2}.get(in_ch)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="input .pth (params_ema / params)")
    ap.add_argument("out", help="output .safetensors")
    ap.add_argument("--scale", type=int, choices=(2, 4), default=None,
                    help="net upsampling factor; inferred from conv_first when omitted")
    args = ap.parse_args()

    state = load_checkpoint(args.checkpoint)
    scale = args.scale if args.scale is not None else infer_scale(state)
    if scale is None:
        raise SystemExit(
            "could not infer scale from conv_first.weight; pass --scale 2 or --scale 4"
        )

    want = expected_names()
    missing = [n for n in want if n not in state]
    extra = [n for n in state if n not in set(want)]
    if missing:
        raise SystemExit(
            f"checkpoint is not an RRDBNet: missing {len(missing)} tensors, "
            f"first few: {missing[:5]}"
        )
    if extra:
        raise SystemExit(f"unexpected extra tensors: {extra[:5]}")

    header = {"__metadata__": {
        "format": "realesrgan-safetensors-v1",
        "scale": str(scale),
        "num_feat": "64",
        "num_block": "23",
        "num_grow_ch": "32",
        "in_ch": "3",
        "out_ch": "3",
        "source": os.path.basename(args.checkpoint),
    }}

    offset = 0
    payloads = []
    order = []
    for name in want:
        t = state[name]
        if not isinstance(t, torch.Tensor):
            raise SystemExit(f"{name} is {type(t)}, not a tensor")
        if t.dtype != torch.float32:
            t = t.to(torch.float32)
        # Contiguous, and in the *reference* layout: the engine stores conv
        # weights as [c_out][c_in][ky][kx] and indexes them the same way the
        # PyTorch module does, so no transposition happens here.
        t = t.detach().contiguous().cpu()
        raw = t.numpy().tobytes()

        pad = (-offset) % ALIGN
        if pad:
            payloads.append(b"\x00" * pad)
            offset += pad

        header[name] = {
            "dtype": "F32",
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        payloads.append(raw)
        offset += len(raw)
        order.append(name)

    header["__metadata__"]["tensor_count"] = str(len(order))

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_json += b" " * ((-(len(header_json) + 8)) % ALIGN)

    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_json)))
        fh.write(header_json)
        for p in payloads:
            fh.write(p)

    mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"wrote {args.out} ({mb:.1f} MiB, {len(order)} tensors, scale x{scale})")


if __name__ == "__main__":
    sys.exit(main())
