import os
import socket
import torch
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, broadcast

def worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2", ACCELERATE_USE_CPU="true", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    torch.set_num_threads(1)
    Accelerator(cpu=True, kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=30))])
    for kind in ("leaf_view", "owned_leaf", "base_leaf"):
        for variant in ("raw_collective", "plain_copy", "current", "detached_copy", "set_data_copy"):
            values = (torch.arange(12, dtype=torch.float64) + rank * 20)
            if kind == "leaf_view":
                backing = values.reshape(3,4).requires_grad_()
                view = backing.t()
            elif kind == "owned_leaf":
                backing = values.reshape(3,4).clone().requires_grad_()
                view = backing.t()
            else:
                backing = values.requires_grad_()
                view = backing.reshape(3,4).t()
            before = str(view.grad_fn)
            if variant == "raw_collective":
                torch.distributed.broadcast(view, src=0)
            elif variant == "current":
                broadcast(view)
            elif variant == "plain_copy":
                if rank != 0:
                    with torch.no_grad():
                        view.copy_(torch.arange(12,dtype=torch.float64).reshape(3,4).t())
            else:
                temporary = view.detach().contiguous()
                torch.distributed.broadcast(temporary,src=0)
                if rank != 0:
                    if variant == "detached_copy":
                        view.detach().copy_(temporary)
                    else:
                        view.data.copy_(temporary)
            after = str(view.grad_fn)
            view.square().sum().backward()
            print({"rank":rank,"kind":kind,"variant":variant,"backing_leaf":backing.is_leaf,"backing_has_base":backing._base is not None,"grad_is_none":backing.grad is None,"before_grad_fn":before,"after_grad_fn":after,"version":backing._version},flush=True)
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    with socket.socket() as s:
        s.bind(("127.0.0.1",0)); port=s.getsockname()[1]
    torch.multiprocessing.spawn(worker,args=(port,),nprocs=2)
