# Copyright (c) 2026 SandAI. All Rights Reserved.
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

"""A c10d ``Work`` backed by a plain CUDA event.

``wait_tensor(t)`` resolves to whichever ``Work`` was registered for ``t``, so an
op that runs on a stream of its own can publish its completion through the
ordinary functional-collective wait instead of inventing a second synchronization
channel that the Inductor scheduler would not understand.  Used by the
copy-engine weight gather and by the host-to-device weight load.
"""

from __future__ import annotations

import torch
import torch._C._distributed_c10d as _c10d


class EventWork(_c10d.Work):
    """c10d Work whose ``wait()`` is a stream wait on a recorded event."""

    def __init__(self, event: torch.cuda.Event) -> None:
        super().__init__()
        self._event = event

    def wait(self, timeout=None) -> bool:  # noqa: ARG002 - c10d's signature
        torch.cuda.current_stream().wait_event(self._event)
        return True
