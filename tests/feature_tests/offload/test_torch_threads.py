# Copyright (c) 2026 SandAI. All Rights Reserved.

import pytest
import torch

from magi_compiler.utils.torch_threads import intra_op_threads


@pytest.mark.parametrize('should_raise', [False, True])
def test_intra_op_threads_restores_process_setting(should_raise: bool) -> None:
    previous_threads = torch.get_num_threads()
    compare_threads = 1 if previous_threads != 1 else 2
    try:
        with intra_op_threads(compare_threads):
            assert torch.get_num_threads() == compare_threads
            if should_raise:
                raise RuntimeError('comparison failed')
    except RuntimeError as error:
        assert str(error) == 'comparison failed'
    assert torch.get_num_threads() == previous_threads
