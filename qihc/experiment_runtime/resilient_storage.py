"""Opt-in atomic writes with bounded retries for transient shared-reader locks.

Windows may reject replacing a heartbeat while another process reads it.
Keep the existing atomic writer/cleanup; never retry indefinitely or mask disk errors.
"""
import time
from . import storage
from .storage import digest, key, load, value_digest, worker_lock  # noqa: F401


def save(path, value):
    for attempt in range(8):
        try:
            return storage.save(path, value)
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.01 * 2**attempt)
