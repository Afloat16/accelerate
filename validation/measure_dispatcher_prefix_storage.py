import json
import torch
from accelerate import Accelerator
from accelerate.data_loader import DataLoaderDispatcher

Accelerator(cpu=True)
torch.set_num_threads(1)
batch_size, width = 256, 2048

def collate(indices):
    return torch.stack([torch.full((width,), float(index)) for index in indices])

loader = DataLoaderDispatcher(range(batch_size * 3), batch_size=batch_size, collate_fn=collate)
iterator = iter(loader)
first = next(iterator)
del first
second = next(iterator)
del second
# Validation-only observation of the live padding cache's actual tensor storage.
prefix = iterator.gi_frame.f_locals["first_batch"]
result = {
    "torch": torch.__version__,
    "batch_size": batch_size,
    "feature_width": width,
    "element_size": prefix.element_size(),
    "cached_samples": prefix.shape[0],
    "cached_logical_bytes": prefix.numel() * prefix.element_size(),
    "cached_storage_bytes": prefix.untyped_storage().nbytes(),
}
del prefix
list(iterator)
print(json.dumps(result))
