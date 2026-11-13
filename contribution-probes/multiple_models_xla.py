import copy
import torch
import torch_xla
from accelerate import Accelerator
from accelerate.utils import DistributedType

print("versions", torch.__version__, torch_xla.__version__)
accelerator = Accelerator()
print("accelerator", accelerator.distributed_type, accelerator.device)
assert accelerator.distributed_type == DistributedType.XLA
models = [torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)]
optimizers = [torch.optim.SGD(model.parameters(), lr=0.1) for model in models]
old = [[id(p) for p in model.parameters()] for model in models]
models[0], optimizers[0], models[1], optimizers[1] = accelerator.prepare(
    models[0], optimizers[0], models[1], optimizers[1]
)
print("parameter devices", [[str(p.device) for p in m.parameters()] for m in models])
print("optimizer devices", [[str(p.device) for p in o.param_groups[0]["params"]] for o in optimizers])
print("identity matches", [[p is q for p, q in zip(m.parameters(), o.param_groups[0]["params"])] for m,o in zip(models,optimizers)])
for i, (model, optimizer) in enumerate(zip(models,optimizers)):
    x = torch.tensor([[0.5, 1.0, -0.5], [1.0, -0.5, 0.25]], device=accelerator.device)
    target = torch.tensor([[0.25, -0.75], [1.0, 0.5]], device=accelerator.device)
    before = model.weight.detach().cpu().clone()
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), target)
    accelerator.backward(loss)
    print("loss and grad", i, loss.item(), model.weight.grad.cpu())
    optimizer.step()
    torch_xla.sync()
    after = model.weight.detach().cpu()
    print("updated", i, not torch.equal(before,after))
    assert all(p is q for p,q in zip(model.parameters(), optimizer.param_groups[0]["params"]))
    assert not torch.equal(before, after)
