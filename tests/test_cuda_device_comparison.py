# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Unindexed CUDA device comparisons must honor the active GPU."""

import torch

from accelerate.utils.modeling import check_device_same


def test_unindexed_cuda_refers_to_the_current_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    assert check_device_same(torch.device("cuda"), torch.device("cuda:1"))
    assert check_device_same(torch.device("cuda:1"), torch.device("cuda"))
    assert not check_device_same(torch.device("cuda"), torch.device("cuda:0"))
    assert not check_device_same(torch.device("cuda:0"), torch.device("cuda:1"))


def test_unindexed_cuda_preserves_default_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA must not initialize")),
    )

    assert check_device_same(torch.device("cuda"), torch.device("cuda:0"))
    assert not check_device_same(torch.device("cuda"), torch.device("cuda:1"))
    assert check_device_same(torch.device("cpu"), torch.device("cpu"))
