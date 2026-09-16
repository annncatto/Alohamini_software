# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect the selected environment and execute one small CUDA operation."""

import argparse


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    import torch

    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("torch.cuda.is_available", torch.cuda.is_available())
    print("torch.cuda.device_count", torch.cuda.device_count())
    if not torch.cuda.is_available():
        print("CUDA is unavailable in this Python environment.")
        return 1
    print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))
    print("torch.cuda.get_arch_list()", torch.cuda.get_arch_list())
    values = torch.ones((32, 32), device="cuda")
    result = values @ values
    torch.cuda.synchronize()
    if not torch.all(result == 32).item():
        raise RuntimeError("CUDA matrix multiplication returned incorrect values")
    print("CUDA tensor operation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
