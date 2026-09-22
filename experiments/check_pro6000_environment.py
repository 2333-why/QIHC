#!/usr/bin/env python3
"""Validate a configurable CUDA runtime, BF16 kernels, and NCCL collectives."""

from __future__ import annotations

import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--min-compute-capability", type=int, default=8)
    parser.add_argument("--min-memory-gib", type=float, default=0.0)
    parser.add_argument("--matrix-size", type=int, default=1024)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    visible = torch.cuda.device_count()
    if visible != args.expected_gpus:
        raise RuntimeError(f"expected {args.expected_gpus} visible GPUs, found {visible}")

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    properties = torch.cuda.get_device_properties(device)
    if properties.major < args.min_compute_capability:
        raise RuntimeError(
            f"GPU {local_rank} is {properties.name} with compute capability "
            f"{properties.major}.{properties.minor}; expected at least "
            f"{args.min_compute_capability}.0"
        )
    memory_gib = properties.total_memory / 2**30
    if memory_gib < args.min_memory_gib:
        raise RuntimeError(
            f"GPU {local_rank} is {properties.name} with {memory_gib:.1f} GiB; "
            f"expected at least {args.min_memory_gib:.1f} GiB"
        )

    if world > 1:
        dist.init_process_group("nccl")

    x = torch.randn(args.matrix_size, args.matrix_size, device=device, dtype=torch.bfloat16)
    checksum = (x @ x).float().mean()
    if world > 1:
        dist.all_reduce(checksum)
        checksum /= world
        dist.barrier()

    print(
        f"rank={rank}/{world} gpu={local_rank} name={properties.name!r} "
        f"cc={properties.major}.{properties.minor} memory_gib={memory_gib:.1f} "
        f"torch={torch.__version__} cuda={torch.version.cuda} bf16_checksum={checksum.item():.6f}",
        flush=True,
    )
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
