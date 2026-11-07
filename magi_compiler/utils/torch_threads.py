# Copyright (c) 2026 SandAI. All Rights Reserved.

from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def intra_op_threads(num_threads: int) -> Iterator[None]:
    """Temporarily change process-wide intra-op threads during serialized model initialization."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)
