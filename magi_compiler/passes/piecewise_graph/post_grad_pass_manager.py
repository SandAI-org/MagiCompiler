# Copyright (c) 2025 SandAI. All Rights Reserved.
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

import functools

from torch import fx as fx
from torch._inductor.custom_graph_pass import CustomGraphPass

from ...config import PassConfig, get_compile_config
from ...utils import magi_logger, set_env_var
from ...utils.envs import MAGI_PATTERN_MATCH_DEBUG
from ..pass_base import InductorPass, get_pass_context, snapshot_original_inductor_configs
from .fix_functionalization import FixFunctionalizationPass
from .post_cleanup import PostCleanupPass


def with_pattern_match_debug(fn):
    """
    Function decorator that turns on inductor pattern match debug
    for the duration of the call.
    Used to avoid logging builtin Inductor pattern matching.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if (debug_val := MAGI_PATTERN_MATCH_DEBUG) is not None:
            # optionally check rank here
            with set_env_var("TORCHINDUCTOR_PATTERN_MATCH_DEBUG", debug_val):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapper


class PostGradPassManager(CustomGraphPass):
    """
    The pass manager for post-grad passes.
    It handles configuration, adding custom passes, and running passes.
    It supports uuid for the Inductor code cache.

    The order of the post-grad post-passes is:
    1. passes (constructor parameter)
    2. default passes (NoopEliminationPass, FusionPass)
    3. config["post_grad_custom_post_pass"] (if it exists)
    4. fix_functionalization
    This way, all passes operate on a functionalized graph.
    """

    def __init__(self):
        self.passes: list[InductorPass] = []

    @with_pattern_match_debug
    def __call__(self, graph: fx.Graph):
        magi_logger.info("Run PostGradPassManager")

        ctx = get_pass_context()
        shape = ctx.runtime_shape
        for pass_ in self.passes:
            applied = pass_(graph)
            if not applied:
                magi_logger.info("Skipping %s with shape %s", pass_, shape)

        # post-cleanup goes before fix_functionalization because it requires a functional graph
        self.post_cleanup(graph)

        # always run fix_functionalization last
        self.fix_functionalization(graph)

    def configure(self, pass_config: PassConfig):
        self.pass_config = pass_config

        if pass_config.enable_conv_channels_last:
            from .conv_channels_last import ConvChannelsLastPass

            self.add(ConvChannelsLastPass())

        if pass_config.enable_nd_tiling_workaround:
            from .nd_tiling_workaround import ND_TilingWorkaroundPass

            self.add(ND_TilingWorkaroundPass())

        if pass_config.enable_mm_epilogue_fusion:
            compile_config = get_compile_config()
            if compile_config.has_cutlass:
                from .fusion.matmul_epilogue_fusion import MatmulEvtEpilogueFusionPass

                self.add(MatmulEvtEpilogueFusionPass())
            else:
                magi_logger.warning(
                    "Skipping matmul epilogue fusion because CUTLASS is unavailable. "
                    "Set MAGI_CUTLASS_ROOT or compile_config.cutlass_root to a valid CUTLASS source tree."
                )

        # needs a functional graph
        self.post_cleanup = PostCleanupPass()
        self.fix_functionalization = FixFunctionalizationPass()

    def add(self, pass_: InductorPass):
        assert isinstance(pass_, InductorPass)
        self.passes.append(pass_)

    def snapshot_original_inductor_configs(self, inductor_compile_config: dict) -> None:
        """Snapshot original values of global Inductor configs potentially mutated by passes."""
        snapshot_original_inductor_configs(self.passes, inductor_compile_config)

    def uuid(self):
        """
        The PostGradPassManager is set as a custom pass in the Inductor and
        affects compilation caching. Its uuid depends on the UUIDs of all
        dependent passes and the pass config. See InductorPass for more info.
        """
        state = {"pass_config": self.pass_config.uuid(), "passes": []}
        for pass_ in self.passes:
            state["passes"].append(pass_.uuid())
        state["passes"].append(self.post_cleanup.uuid())
        state["passes"].append(self.fix_functionalization.uuid())

        return InductorPass.hash_dict(state)
