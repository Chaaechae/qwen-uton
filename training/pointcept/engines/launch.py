"""
Launcher

modified from detectron2 (https://github.com/facebookresearch/detectron2)
and from the original Pointcept launch.py — see
third_party/Pointcept/pointcept/engines/launch.py.

This in-repo copy is symlinked into the Pointcept tree by
training/install_into_pointcept.sh. The only functional delta vs upstream
is the distributed-backend resolution:

    backend = os.environ.get("DIST_BACKEND", "NCCL").lower()

Set ``DIST_BACKEND=gloo`` to use the CPU-friendly Gloo backend (necessary
on clusters where NCCL is broken / forbidden / over a fabric Gloo handles
better). Gloo does not support ``ReduceOp.AVG`` — model-side all_reduce
calls must use ``SUM`` + manual division by world_size. The Utonia model
files in this repo (utonia_v1m2/v1m3a/v1m3b) already do that.
"""

import os
import logging
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pointcept.utils import comm

__all__ = ["DEFAULT_TIMEOUT", "launch"]


def _resolve_timeout():
    # Default 60 min like upstream. Gloo clusters with slow data loading
    # sometimes need more; override via DIST_TIMEOUT_MIN (in minutes).
    return timedelta(minutes=int(os.environ.get("DIST_TIMEOUT_MIN", "60")))


DEFAULT_TIMEOUT = _resolve_timeout()


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _resolve_backend():
    backend = os.environ.get("DIST_BACKEND", "NCCL").lower()
    if backend not in ("nccl", "gloo", "mpi"):
        raise ValueError(
            f"Unsupported DIST_BACKEND={backend!r}. Expected one of: nccl, gloo, mpi."
        )
    return backend


def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    cfg=(),
    timeout=DEFAULT_TIMEOUT,
):
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        if dist_url == "auto":
            assert (
                num_machines == 1
            ), "dist_url=auto not supported in multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            logger = logging.getLogger(__name__)
            logger.warning(
                "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
            )
        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                cfg,
                timeout,
            ),
            daemon=False,
        )
    else:
        main_func(*cfg)


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
    cfg,
    timeout=DEFAULT_TIMEOUT,
):
    assert (
        torch.cuda.is_available()
    ), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    backend = _resolve_backend()
    try:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        logger.error("Backend: {}".format(backend))
        raise e

    if global_rank == 0:
        logging.getLogger(__name__).info(
            f"[launch] distributed initialized with backend={backend}, "
            f"world_size={world_size}"
        )

    # Setup the local process group (which contains ranks within the same machine).
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # synchronize is needed here to prevent a possible timeout after calling init_process_group.
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    main_func(*cfg)
