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

"""Unit tests for the FSDP-overlap weight all-gather bucketing pass
(``magi_compiler.passes.fsdp_overlap.bucket_all_gather`` +
``lower_and_bucket_full_graph``).

Pure-CPU: the bucketing pass operates on ``all_gather_into_tensor`` fx nodes tagged
``magi_fsdp_weight_ag`` (it never runs the ops), so we drive it with SYNTHETIC fx
graphs built with meta-tensor ``example_value``s -- no DTensor / distributed / GPU.

Bucketing is WHOLE-GRAPH: there is no region / subgraph partitioning.  Buckets
break only at program-order dtype changes and the optional byte cap.
"""

from collections import Counter

import pytest
import torch
import torch.fx as fx

from magi_compiler.passes.fsdp_overlap import bucket_weight_all_gather_coalesced, lower_and_bucket_full_graph

_AG = torch.ops._c10d_functional.all_gather_into_tensor.default
_AG_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default


# ---------------------------------------------------------------------------
# synthetic graph builder
# ---------------------------------------------------------------------------
def _build_ag_graph(specs, world=2, group="grp0"):
    """Build an fx graph of independent weight all-gathers.

    ``specs``: list of dicts, each ``{shape, dtype, compute_before?}``.  For each spec
    we emit  weight_shard placeholder -> all_gather_into_tensor (tagged
    magi_fsdp_weight_ag) -> wait_tensor.  A spec with ``compute_before`` inserts an
    opaque compute op (aten.relu here) between gathers, which must NOT break the
    bucket (whole-graph bucketing has no region boundaries).  ALL placeholders are
    declared first (as in a real traced graph) so the hoisted coalesced launch stays
    topologically valid.
    """
    g = fx.Graph()
    locs = []
    for i, s in enumerate(specs):
        loc = g.placeholder(f"w{i}_weight_shard")
        loc.meta["example_value"] = torch.empty(*s["shape"], dtype=s["dtype"], device="meta")
        locs.append(loc)

    outs = []
    for i, (s, loc) in enumerate(zip(specs, locs)):
        if s.get("compute_before"):
            # opaque compute op consuming the previous wait (keeps it in the graph)
            b = g.call_function(torch.ops.aten.relu.default, (outs[-1],)) if outs else None
            if b is not None:
                b.meta["example_value"] = outs[-1].meta["example_value"]
        chunk = s["shape"][0]
        rest = s["shape"][1:]
        gathered = torch.empty(chunk * world, *rest, dtype=s["dtype"], device="meta")
        ag = g.call_function(_AG, (loc, world, group))
        ag.meta["example_value"] = gathered
        ag.meta["magi_fsdp_weight_ag"] = True
        w = g.call_function(_WAIT, (ag,))
        w.meta["example_value"] = gathered
        outs.append(w)
    g.output(tuple(outs))
    gm = fx.GraphModule(torch.nn.Module(), g)
    return gm


def _op_counts(gm) -> Counter:
    return Counter(str(n.target) for n in gm.graph.nodes if n.op == "call_function")


def _n(gm, target) -> int:
    return sum(1 for n in gm.graph.nodes if n.op == "call_function" and n.target is target)


# ---------------------------------------------------------------------------
# bucket_weight_all_gather_coalesced
# ---------------------------------------------------------------------------
def test_coalesce_merges_to_one():
    """N same-(group,dtype) gathers -> ONE coalesced + N getitems + N waits."""
    gm = _build_ag_graph([{"shape": (4, 8), "dtype": torch.bfloat16}] * 3)
    n_buckets = bucket_weight_all_gather_coalesced(gm)

    assert n_buckets == 1
    assert _n(gm, _AG_COALESCED) == 1
    assert _n(gm, _AG) == 0  # all plain gathers replaced
    assert _n(gm, _WAIT) == 3  # one wait per member, preserved
    assert sum(1 for n in gm.graph.nodes if n.op == "call_function" and "getitem" in str(n.target)) == 3
    gm.graph.lint()  # topologically valid


def test_single_gather_not_coalesced():
    """A lone weight gather has nothing to coalesce -> untouched, 0 buckets."""
    gm = _build_ag_graph([{"shape": (4, 8), "dtype": torch.bfloat16}])
    n_buckets = bucket_weight_all_gather_coalesced(gm)
    assert n_buckets == 0
    assert _n(gm, _AG) == 1
    assert _n(gm, _AG_COALESCED) == 0


def test_dtype_change_breaks_bucket():
    """bf16, bf16, fp32, bf16 (program order) -> {bf16,bf16}, {fp32 alone}, {bf16 alone}
    => only the leading same-dtype run of >=2 coalesces -> 1 bucket."""
    gm = _build_ag_graph(
        [
            {"shape": (4, 8), "dtype": torch.bfloat16},
            {"shape": (4, 8), "dtype": torch.bfloat16},
            {"shape": (4, 8), "dtype": torch.float32},
            {"shape": (4, 8), "dtype": torch.bfloat16},
        ]
    )
    n_buckets = bucket_weight_all_gather_coalesced(gm)
    assert n_buckets == 1  # only the {bf16,bf16} run
    assert _n(gm, _AG_COALESCED) == 1
    assert _n(gm, _AG) == 2  # the fp32 and the trailing bf16 stay individual


def test_bucket_size_bytes_caps_run():
    """Same dtype, but bucket_size_bytes forces a split.  4x8 bf16 shard = 64 B each;
    cap at 64 B means each gather is its own bucket -> no coalescing (0 buckets>=2)."""
    gm = _build_ag_graph([{"shape": (4, 8), "dtype": torch.bfloat16}] * 4)
    # 4*8*2 = 64 B local shard each; cap at 64 B => each starts a new bucket -> singletons
    n_buckets = bucket_weight_all_gather_coalesced(gm, bucket_size_bytes=64)
    assert n_buckets == 0
    assert _n(gm, _AG) == 4

    # cap at 128 B => 2 shards per bucket => 2 buckets of 2
    gm2 = _build_ag_graph([{"shape": (4, 8), "dtype": torch.bfloat16}] * 4)
    n2 = bucket_weight_all_gather_coalesced(gm2, bucket_size_bytes=128)
    assert n2 == 2
    assert _n(gm2, _AG_COALESCED) == 2


def test_compute_between_gathers_does_not_break_bucket():
    """Whole-graph bucketing: an interleaved compute op between gathers does NOT
    split them into separate buckets (no region boundaries)."""
    gm = _build_ag_graph(
        [
            {"shape": (4, 8), "dtype": torch.bfloat16},
            {"shape": (4, 8), "dtype": torch.bfloat16},
            {"shape": (4, 8), "dtype": torch.bfloat16, "compute_before": True},
            {"shape": (4, 8), "dtype": torch.bfloat16},
        ]
    )
    n_buckets = bucket_weight_all_gather_coalesced(gm)
    assert n_buckets == 1  # all 4 gathers in ONE bucket despite the relu in between
    assert _n(gm, _AG_COALESCED) == 1
    assert _n(gm, _AG) == 0
    gm.graph.lint()


# ---------------------------------------------------------------------------
# lower_and_bucket_full_graph (entry point)
# ---------------------------------------------------------------------------
def test_lower_and_bucket_mode_none_returns_zero():
    """mode 'none' -> lowering only, no bucketing -> 0 buckets, gathers untouched."""
    gm = _build_ag_graph([{"shape": (4, 8), "dtype": torch.bfloat16}] * 3)
    n = lower_and_bucket_full_graph(gm, "none")
    assert n == 0
    assert _n(gm, _AG) == 3
    assert _n(gm, _AG_COALESCED) == 0


def test_lower_and_bucket_mode_coalesced():
    """mode 'coalesced' -> all same-(group,dtype) gathers in one whole-graph bucket."""
    gm = _build_ag_graph([{"shape": (4, 8), "dtype": torch.bfloat16}] * 3)
    n = lower_and_bucket_full_graph(gm, "coalesced")
    assert n == 1
    assert _n(gm, _AG_COALESCED) == 1


def test_lower_and_bucket_size_cap():
    """bucket_size_bytes flows through the entry point: cap 128 B on 4x64 B shards
    -> 2 buckets of 2."""
    gm = _build_ag_graph([{"shape": (4, 8), "dtype": torch.bfloat16}] * 4)
    n = lower_and_bucket_full_graph(gm, "coalesced", bucket_size_bytes=128)
    assert n == 2
    assert _n(gm, _AG_COALESCED) == 2


def test_lower_and_bucket_unknown_mode_raises():
    gm = _build_ag_graph([{"shape": (4, 8), "dtype": torch.bfloat16}] * 2)
    with pytest.raises(ValueError, match="expected 'none' or 'coalesced'"):
        lower_and_bucket_full_graph(gm, "concat")
