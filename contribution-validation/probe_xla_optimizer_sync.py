import math
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

from accelerate import Accelerator


class TwoHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(1, 1, bias=False)
        self.second = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(self.first.weight)
        torch.nn.init.ones_(self.second.weight)


def worker(index):
    accelerator = Accelerator()
    print("world", xr.world_size(), "rank", index, "device", accelerator.device, flush=True)
    assert xr.world_size() == 2
    control = torch.tensor([2.0 + index]).to(accelerator.device)
    control = xm.all_reduce("sum", control)
    xm.mark_step()
    print("CONTROL", index, control.cpu().tolist(), flush=True)
    assert control.cpu().item() == 5
    model = TwoHeads().to(accelerator.device)
    one = torch.optim.SGD(model.first.parameters(), lr=0.1)
    two = torch.optim.SGD(model.second.parameters(), lr=0.1)
    model, one, two = accelerator.prepare(model, one, two)
    xm.mark_step()
    x1 = torch.tensor([[2.0 + 2 * index]], device=accelerator.device)
    x2 = torch.tensor([[4.0 + 6 * index]], device=accelerator.device)
    loss = model.first(x1).sum() + model.second(x2).sum()
    accelerator.backward(loss)
    norm = accelerator.clip_grad_norm_(model.parameters(), float("inf"))
    xm.mark_step()
    values = [float(p.grad.cpu().item()) for p in model.parameters()]
    print("OBSERVED", index, values, "norm", norm.cpu().item(), flush=True)
    assert values == [3.0, 7.0], values
    assert abs(norm.cpu().item() - math.sqrt(58)) < 1e-5


if __name__ == "__main__":
    xmp.spawn(worker)
