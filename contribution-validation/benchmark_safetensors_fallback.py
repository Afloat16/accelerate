# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import gc
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
import tracemalloc

import torch
from safetensors.torch import save_file

from accelerate.utils.modeling import load_state_dict


torch.set_num_threads(1)
results = []
with tempfile.TemporaryDirectory() as directory, open(os.devnull, "w") as stderr:
    for count in (64, 4096, 32768):
        state = {
            f"{'assigned' if index % 2 == 0 else 'offloaded'}.weight_{index:06d}": torch.tensor(float(index))
            for index in range(count)
        }
        path = os.path.join(directory, "model.safetensors")
        save_file(state, path, metadata={"format": "pt"})
        device_map = {"assigned": "cpu", "offloaded": "disk"}
        with contextlib.redirect_stderr(stderr):
            loaded = load_state_dict(path, device_map)
            assert loaded.keys() == state.keys()
            for name, value in state.items():
                torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)
            digest = hashlib.sha256()
            for name in sorted(loaded):
                digest.update(name.encode())
                digest.update(loaded[name].numpy().tobytes())
            del loaded
            times = []
            for _ in range(3):
                gc.collect()
                start = time.perf_counter()
                loaded = load_state_dict(path, device_map)
                times.append(time.perf_counter() - start)
                del loaded
            gc.collect()
            tracemalloc.start()
            loaded = load_state_dict(path, device_map)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert len(loaded) == count
            del loaded
        results.append(
            {
                "weight_count": count,
                "file_bytes": os.path.getsize(path),
                "seconds": times,
                "median_seconds": statistics.median(times),
                "python_peak_bytes": peak,
                "output_sha256": digest.hexdigest(),
            }
        )
report = {"label": sys.argv[1], "torch": torch.__version__, "results": results}
print(json.dumps(report, indent=2))
with open(sys.argv[2], "w") as output:
    json.dump(report, output, indent=2)
