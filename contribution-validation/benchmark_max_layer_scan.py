# Copyright 2026 The HuggingFace Team. All rights reserved.

import gc
import hashlib
import json
import statistics
import sys
import time
import tracemalloc
from functools import partial

import torch
from torch import nn

from accelerate.utils import compute_module_sizes, infer_auto_device_map
from accelerate.utils.modeling import get_max_layer_size


torch.set_num_threads(1)
rows = []
for count in (64, 8192, 65536):
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(2, 2, bias=False, device="meta") for _ in range(count)])
    sizes = compute_module_sizes(model)
    modules = list(model.named_children())
    budget = {"cpu": sizes[""] + 16}
    for operation in ("maximum_layer", "device_map"):
        run = (
            partial(get_max_layer_size, modules, sizes, [])
            if operation == "maximum_layer"
            else partial(infer_auto_device_map, model, max_memory=budget)
        )

        output = run()
        timings = []
        for _ in range(5):
            started = time.perf_counter()
            current = run()
            timings.append(time.perf_counter() - started)
            assert current == output
        gc.collect()
        tracemalloc.start()
        measured = run()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert measured == output
        if operation == "maximum_layer":
            assert output == (16, [f"layers.{index}" for index in range(count)])
        else:
            assert output == {"layers": "cpu"}
        rows.append(
            {
                "layer_count": count,
                "operation": operation,
                "median_seconds": statistics.median(timings),
                "python_peak_bytes": peak,
                "output_sha256": hashlib.sha256(json.dumps(output).encode()).hexdigest(),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    gc.collect()
with open(sys.argv[2], "w") as target:
    json.dump(
        {"source": sys.argv[1], "torch": torch.__version__, "results": rows},
        target,
        indent=2,
    )
