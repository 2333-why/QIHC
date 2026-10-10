#!/usr/bin/env python3
"""Process-local torchrun workaround for Windows wheels without libuv.

Some launcher versions construct TCPStore without consulting USE_LIBUV. Only
this launcher process is patched; no installed package files are modified.
"""

import os


def main():
    if os.name == "nt":
        os.environ["USE_LIBUV"] = "0"
        import torch.distributed as distributed
        from torch.distributed.elastic.rendezvous import c10d_rendezvous_backend
        constructor = distributed.TCPStore

        class LegacyTCPStore(constructor):
            def __init__(self, *args, **kwargs):
                kwargs["use_libuv"] = False
                super().__init__(*args, **kwargs)

        distributed.TCPStore = LegacyTCPStore
        c10d_rendezvous_backend.TCPStore = LegacyTCPStore
    from torch.distributed.run import main as torchrun_main
    torchrun_main()


if __name__ == "__main__":
    main()
