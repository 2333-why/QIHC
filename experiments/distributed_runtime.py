"""Shared Gloo initialization, including legacy Windows TCPStore clients."""

from datetime import timedelta
import os
from pathlib import Path
import time


class FileBarrier:
    """Only for independent inference shards; never a replacement for gradient DDP."""

    def __init__(self, rank, world, directory):
        self.rank, self.world, self.phase = rank, world, 0
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def barrier(self):
        marker = self.directory / f"phase{self.phase}-rank{self.rank}"
        marker.touch(exist_ok=False)
        started = time.monotonic()
        while not all((self.directory / f"phase{self.phase}-rank{rank}").exists() for rank in range(self.world)):
            if time.monotonic() - started > 43200:
                raise TimeoutError("An independent inference shard never reached its file barrier")
            time.sleep(0.05)
        self.phase += 1

    def destroy_process_group(self):
        pass


def initialize_gloo(rank: int, world: int):
    import torch.distributed as distributed
    if distributed.is_initialized():
        return distributed
    timeout = timedelta(hours=12)
    if os.name == "nt":
        directory = os.environ.get("QIHC_BARRIER_DIRECTORY")
        if not directory:
            raise RuntimeError("Windows inference shards require a unique QIHC_BARRIER_DIRECTORY; use run_research_suite.py")
        return FileBarrier(rank, world, directory)
    else:
        distributed.init_process_group("gloo", timeout=timeout)
    return distributed
