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

import torch

from ...magi_depyf.timeline import observe_lifecycle
from .remove_item import RemoveItemPass
from .remove_useless_ops import EliminateIdentityViewCastPass
from .replace_sage_atten import ReplaceSageAttentionPass


class FullGraphPassManager:
    """
    A manager to apply various graph passes on the full graph before splitting.
    """

    def __init__(self, pass_config):
        self.pass_config = pass_config
        self.passes = []
        if self.pass_config.enable_sage_attn:
            self.passes.append(ReplaceSageAttentionPass())
        self.passes.append(RemoveItemPass())
        self.passes.append(EliminateIdentityViewCastPass())

    @observe_lifecycle("full_graph_manager")
    def __call__(self, gm: torch.fx.GraphModule):
        for pass_ in self.passes:
            if pass_.is_applicable(gm.graph):
                pass_(gm.graph)
