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

import collections
import os
import operator
from typing import Any, Dict

import torch
from torch.fx import GraphModule, Node
from torch.fx.node import map_arg

from magi_compiler.config import CompileConfig
from magi_compiler.offload.profiler import OffloadProfiler
from magi_compiler.offload.scheduler import OffloadRuntimeContext, SchedulerFactory
from magi_compiler.utils.nvtx import add_nvtx_event

from ..magi_depyf.timeline import observe_lifecycle


class OffloadExecutor:
    def __init__(self, graph_module: GraphModule, compile_config: CompileConfig):
        self.graph_module = graph_module
        self.compile_config = compile_config

        self.compute_stream = torch.cuda.current_stream()
        self.h2d_stream = torch.cuda.Stream()

        self.warmup = True
        self.second_call = False
        self._call_idx = 0
        self.buffers: Dict[str, torch.Tensor] = {}
        self.persistent_weights: Dict[str, torch.Tensor] = {}
        self.submod_0_weight_handoff: Dict[Node, torch.Tensor] = {}

        self._analyze_graph()
        self.profiler = OffloadProfiler()

        common_args = {
            "submod_nodes": self.submod_nodes,
            "submod_weights_map": self.submod_weights_map,
            "name_node_map": self.name_node_map,
            "weight_sizes": self.submod_weight_sizes,
        }

        self.scheduler = SchedulerFactory.create(self.compile_config, common_args)
        OffloadExecutor.LAST = self

    def _analyze_graph(self):
        self.submod_nodes = [n for n in self.graph_module.graph.nodes if n.op == "call_module"]

        self.placeholder_nodes = []
        self.arg_index_weight = {}
        self.user_counts = collections.defaultdict(int)
        self.name_node_map = {}

        placeholder_idx = 0
        _src_types = {}
        _weight_count = 0
        _total_ph = 0
        for node in self.graph_module.graph.nodes:
            for input_node in node.all_input_nodes:
                self.user_counts[input_node] += 1

            if node.op == "placeholder":
                is_w = self._is_weight_node(node)
                self.arg_index_weight[placeholder_idx] = is_w
                _total_ph += 1
                if is_w:
                    _weight_count += 1
                ga = node.meta.get("grapharg")
                if ga is not None:
                    st = type(ga.source).__name__
                    _src_types[st] = _src_types.get(st, 0) + 1
                self.placeholder_nodes.append(node)
                self.name_node_map[node.name] = node
                placeholder_idx += 1

        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[offload stats] total_ph={_total_ph} weight={_weight_count} non_weight={_total_ph-_weight_count} source_types={_src_types}", flush=True)

        self.submod_weights_map = {}
        self.submod_weight_sizes = {}

        for node in self.submod_nodes:
            weight_names = []
            size = 0
            for arg in node.args:
                if isinstance(arg, Node) and self._is_weight_node(arg):
                    if arg.name in self.name_node_map:
                        weight_names.append(arg.name)
                        val = arg.meta.get("example_value")
                        if val is not None:
                            size += val.numel() * val.element_size()

            self.submod_weights_map[node.name] = weight_names
            self.submod_weight_sizes[node.name] = size

    def _is_weight_node(self, node: Node) -> bool:
        if node.op != "placeholder":
            return False
        grapharg = node.meta.get("grapharg")
        if grapharg is not None:
            src = str(grapharg.source)
            if "ParamBufferSource" in src:
                return True
            if "LocalSource" in src and "ParamBuffer" not in src:
                return False
        val = node.meta.get("example_value")
        if val is not None and isinstance(val, torch.nn.Parameter):
            return True
        if not hasattr(self, "_dbg_is_w") and int(os.environ.get("RANK", "0")) == 0:
            self._dbg_is_w = True
            print(f"[offload _is_weight] node={node.name} grapharg={'yes' if grapharg else 'no'} val_type={type(val).__name__ if val else 'None'} -> False", flush=True)
        return False

    def _prepare_inputs(self, args) -> Dict[Node, Any]:
        env = {}
        args = list(args)
        submod_0 = self.submod_nodes[0]

        # Debug: count weight vs non-weight and memory
        import os
        if os.environ.get("MAGI_OFFLOAD_DEBUG") == "1" and int(os.environ.get("RANK", "0")) == 0:
            n_w = sum(1 for v in self.arg_index_weight.values() if v)
            n_nw = sum(1 for v in self.arg_index_weight.values() if not v)
            nw_bytes = sum(
                args[i].nbytes for i, v in self.arg_index_weight.items()
                if not v and isinstance(args[i], torch.Tensor)
            )
            w_bytes = sum(
                args[i].nbytes for i, v in self.arg_index_weight.items()
                if v and isinstance(args[i], torch.Tensor)
            )
            print(f"[offload debug] placeholders: {n_w} weight ({w_bytes/1e9:.2f}GiB) + {n_nw} non-weight ({nw_bytes/1e9:.2f}GiB)", flush=True)
            print(f"[offload debug] GPU before: alloc={torch.cuda.memory_allocated()/1e9:.2f}GiB reserved={torch.cuda.memory_reserved()/1e9:.2f}GiB", flush=True)

        for i, node in enumerate(self.placeholder_nodes):
            arg_val = args[i]
            is_weight = self.arg_index_weight[i]

            # case 1: input tensor
            if not is_weight:
                if isinstance(arg_val, torch.Tensor):
                    arg_val = arg_val.to("cuda", non_blocking=False)
                env[node] = arg_val
                continue

            # case 2: kept weight
            if self.scheduler.is_kept(node.name):
                if node.name not in self.persistent_weights:
                    t = arg_val.to("cuda", non_blocking=False) if arg_val.device.type == "cpu" else arg_val
                    self.persistent_weights[node.name] = t
                env[node] = self.persistent_weights[node.name]
                continue

            # case 3: submod 0 weight
            if submod_0 in node.users:
                if self.warmup and arg_val.device.type == "cpu":
                    self.buffers[node.name] = arg_val
                    arg_val = arg_val.to("cuda", non_blocking=False)
                elif not self.warmup:
                    if node in self.submod_0_weight_handoff:
                        arg_val = self.submod_0_weight_handoff[node]
                        del self.submod_0_weight_handoff[node]

            env[node] = arg_val

        if (
            int(os.environ.get("RANK", "0")) == 0
            and self._call_idx == 3
            and not getattr(self, "_checksum_logged", False)
        ):
            self._checksum_logged = True
            n_cmp = n_bad = 0
            max_abs = 0.0
            scalars = []
            samples = []
            for i, node in enumerate(self.placeholder_nodes):
                if i >= len(args):
                    break
                src = args[i]
                if not isinstance(src, torch.Tensor):
                    continue
                if src.numel() == 1 and len(scalars) < 16:
                    scalars.append((node.name, float(src.detach().reshape(()).float().cpu()), str(src.dtype)))
                if not self.arg_index_weight.get(i) or n_cmp >= 16:
                    continue
                dst = env.get(node)
                if not isinstance(dst, torch.Tensor) or src.shape != dst.shape:
                    continue
                s = src.detach()
                d = dst.detach()
                if s.device != d.device:
                    s = s.to(device=d.device, non_blocking=False)
                mx = float((s.float() - d.float()).abs().max().item())
                n_cmp += 1
                if mx > 0:
                    n_bad += 1
                    max_abs = max(max_abs, mx)
                samples.append((node.name, tuple(src.shape), mx, float(s.float().mean())))
            zeros = []
            for i, a in enumerate(args):
                if isinstance(a, torch.Tensor) and a.dim() == 0:
                    zeros.append((i, float(a.detach().float().cpu()), str(a.dtype)))
            print(
                f"[offload checksum] call={self._call_idx} compared={n_cmp} "
                f"nonzero_diff={n_bad} max_abs={max_abs:.6g} "
                f"samples={samples} scalars={scalars}",
                flush=True,
            )
            print(f"[offload scalars0d] n={len(zeros)} {zeros[:32]}", flush=True)
            cpu_bits = []
            for i, node in enumerate(self.placeholder_nodes):
                val = env.get(node)
                if isinstance(val, torch.Tensor) and val.device.type == "cpu":
                    cpu_bits.append(
                        f"{node.name} w={self.arg_index_weight.get(i)} "
                        f"{tuple(val.shape)} {val.dtype} "
                        f"fin={bool(torch.isfinite(val.float()).all()) if val.is_floating_point() else "n/a"} "
                        f"mean={float(val.float().mean()) if val.is_floating_point() else 0:.4g}"
                    )
            print(f"[offload cpu-args] n={len(cpu_bits)} {cpu_bits[:12]}", flush=True)

        n_forced = 0
        for node, val in list(env.items()):
            if isinstance(val, torch.Tensor) and val.device.type == "cpu":
                env[node] = val.to("cuda", non_blocking=False)
                n_forced += 1
        if int(os.environ.get("RANK", "0")) == 0 and n_forced and self._call_idx <= 4:
            print(f"[offload force-cuda] call={self._call_idx} moved={n_forced}", flush=True)

        return env

    def _finalize_warmup(self):
        profile_results = self.profiler.summarize()
        self.scheduler.schedule_kept_weights(profile_results)
        self.warmup = False

    def __call__(self, *args):
        if len(self.submod_nodes) == 0:
            return self.graph_module(*args)

        self._call_idx += 1
        _rank0 = int(os.environ.get("RANK", "0")) == 0
        if _rank0:
            print(
                f"[offload call] i={self._call_idx} warmup={self.warmup} second_call={self.second_call} "
                f"n_submod={len(self.submod_nodes)} n_buf={len(self.buffers)} n_handoff={len(self.submod_0_weight_handoff)}",
                flush=True,
            )
        env = self._prepare_inputs(args)
        if _rank0 and self._call_idx == 3 and not getattr(self, "_ops_dumped", False):
            self._ops_dumped = True
            hits = {}
            try:
                for node in self.submod_nodes:
                    obj = getattr(self.graph_module, node.target)
                    # PiecewiseBackend.graph is GraphModule; GraphModule.graph is fx.Graph
                    gm = obj.graph if hasattr(obj, "graph") else obj
                    fxg = gm.graph if hasattr(gm, "graph") else None
                    if fxg is None:
                        continue
                    for n in fxg.nodes:
                        if n.op not in {"call_function", "call_method"}:
                            continue
                        tgt = str(n.target)
                        tl = tgt.lower()
                        if any(s in tl for s in (
                            "mhc", "gaga4", "sinkhorn", "grouped_linear", "fa_with",
                            "rms", "flash", "sdpa", "attention", "softmax", "sage",
                            "sink", "rotary", "rope", "sigmoid",
                        )):
                            hits[tgt] = hits.get(tgt, 0) + 1
                        if n.op == "call_function" and "aten." in tl:
                            hits[tgt] = hits.get(tgt, 0) + 1
                print(f"[fx-ops] customish={hits}", flush=True)
                for node in self.submod_nodes[:12]:
                    obj = getattr(self.graph_module, node.target)
                    print(
                        f"[fx-type] {node.name} type={type(obj).__name__} "
                        f"mod={type(obj).__module__} graph={hasattr(obj, 'graph')}",
                        flush=True,
                    )
                for want in ("submod_0", "submod_1", "submod_2", "submod_3"):
                    node = next(n for n in self.submod_nodes if n.name == want)
                    obj = getattr(self.graph_module, node.target)
                    # GraphModule.graph is fx.Graph; PiecewiseBackend.graph is GraphModule
                    if type(obj).__name__ == "GraphModule":
                        fxg = obj.graph
                    elif hasattr(obj, "graph") and hasattr(obj.graph, "graph"):
                        fxg = obj.graph.graph
                    else:
                        fxg = None
                    cnt = {}
                    if fxg is None:
                        print(f"[fx-ops-sub] {want} NO_GRAPH type={type(obj).__name__}", flush=True)
                        continue
                    for n in fxg.nodes:
                        if n.op in {"call_function", "call_method", "call_module"}:
                            k = f"{n.op}:{n.target}"
                            cnt[k] = cnt.get(k, 0) + 1
                    print(f"[fx-ops-sub] {want} {cnt}", flush=True)
                for want in ("submod_6", "submod_7", "submod_10"):
                    node = next(n for n in self.submod_nodes if n.name == want)
                    obj = getattr(self.graph_module, node.target)
                    gm = obj.graph if hasattr(obj, "graph") else obj
                    fxg = gm.graph if hasattr(gm, "graph") else None
                    lits = []
                    zeros = []
                    sig = []
                    if fxg is None:
                        continue
                    for n in fxg.nodes:
                        tgt = str(n.target)
                        if "sigmoid" in tgt.lower() or "mul" in tgt.lower():
                            sig.append(f"{n.op}:{tgt} args={n.args[:6]}")
                        def _eat(o):
                            if isinstance(o, (int, float)) and not isinstance(o, bool):
                                lits.append(o)
                            elif isinstance(o, torch.Tensor) and o.dim() == 0:
                                zeros.append(float(o.detach().cpu()))
                        for a in n.args:
                            _eat(a)
                            if isinstance(a, (tuple, list)):
                                for x in a:
                                    _eat(x)
                    scaleish = [x for x in lits if isinstance(x, float) and 1e-4 < abs(x) < 0.05]
                    ones = [x for x in lits if x == 1.0 or x == 1]
                    print(
                        f"[fx-lits] {want} n_lits={len(lits)} scaleish={scaleish[:12]} "
                        f"ones={len(ones)} 0d={zeros[:8]} "
                        f"sample_lits={lits[:20]} sig={sig[:8]}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[fx-ops] dump failed: {exc}", flush=True)
        if _rank0:
            n_cpu = n_gpu = n_other = 0
            for node, val in env.items():
                if isinstance(val, torch.Tensor):
                    if val.device.type == "cpu":
                        n_cpu += 1
                    elif val.is_cuda:
                        n_gpu += 1
                    else:
                        n_other += 1
            print(f"[offload call] i={self._call_idx} env after prepare cpu={n_cpu} gpu={n_gpu} other={n_other}", flush=True)
        current_user_counts = self.user_counts.copy()
        runtime_ctx = OffloadRuntimeContext(
            env=env,
            h2d_stream=self.h2d_stream,
            compute_stream=self.compute_stream,
            buffers=self.buffers,
            submod_0_handoff=self.submod_0_weight_handoff,
            need_profile=self.second_call or self.warmup,
        )
        need_profile = self.second_call

        for node in self.graph_module.graph.nodes:
            if node.op == "placeholder":
                continue

            elif node.op == "call_module":
                self.scheduler.prefetch(node.name, runtime_ctx)
                # lookahead=0 copies *this* submod's weights on h2d_stream and
                # used to return without waiting. Race → 2nd-call NaN.
                self.compute_stream.wait_stream(self.h2d_stream)

                if need_profile:
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
                    self.profiler.start_compute_profile(node.name, self.compute_stream)

                with add_nvtx_event(node.name):
                    with torch.cuda.stream(self.compute_stream):
                        s_args = map_arg(node.args, lambda n: env[n])
                        s_kwargs = map_arg(node.kwargs, lambda n: env[n])
                        _probe = os.environ.get("GAGA4_NAN_PROBE", "").strip() in {"1", "true", "yes"}
                        if _rank0 and _probe and self._call_idx <= 3 and node.name in {"submod_1", "submod_3", "submod_5", "submod_6"}:
                            bits = []
                            for j, a in enumerate(s_args):
                                if isinstance(a, torch.Tensor) and a.is_floating_point():
                                    x = a.detach()
                                    bits.append(
                                        f"i{j} fin={bool(torch.isfinite(x).all())} "
                                        f"max={float(x.float().abs().max()):.4g} "
                                        f"{tuple(x.shape)} {x.device.type}"
                                    )
                                elif isinstance(a, torch.Tensor):
                                    bits.append(f"i{j} {a.dtype} {tuple(a.shape)} {a.device.type}")
                            print(
                                f"[offload in] call={self._call_idx} {node.name} {' '.join(bits[:8])}",
                                flush=True,
                            )
                        if _rank0 and self._call_idx == 3:
                            z = []
                            for j, a in enumerate(s_args):
                                if isinstance(a, torch.Tensor) and a.dim() == 0 and a.is_floating_point():
                                    z.append(f"i{j}={float(a.detach().cpu()):.8g} {a.dtype}")
                            if z and node.name in {"submod_0", "submod_6", "submod_7", "submod_10"}:
                                print(f"[fx-0d] {node.name} n={len(s_args)} 0d={z}", flush=True)
                        env[node] = getattr(self.graph_module, node.target)(*s_args, **s_kwargs)
                        if (
                            self._call_idx == 3
                            and type(getattr(self.graph_module, node.target)).__name__ == "GraphModule"
                            and getattr(self, "_moe_cmp_n", 0) < 2
                        ):
                            gm = getattr(self.graph_module, node.target)
                            targets = [str(getattr(n, "target", "")) for n in gm.graph.nodes]
                            if any("gaga4_mh_moe" in t for t in targets) and s_args:
                                try:
                                    from model.gaga4.modeling import gaga4_mh_moe

                                    desc = []
                                    for i, a in enumerate(s_args):
                                        if isinstance(a, torch.Tensor):
                                            desc.append(f"i{i}:t{tuple(a.shape)}{a.dtype}")
                                        else:
                                            desc.append(f"i{i}:{type(a).__name__}={a}")
                                    if int(__import__("os").environ.get("RANK", "0")) == 0 and getattr(self, "_moe_cmp_n", 0) == 0:
                                        print(f"[moe-args] {node.name} {desc}", flush=True)
                                    tensors = [a for a in s_args if isinstance(a, torch.Tensor)]
                                    baked_key = None
                                    for gn in gm.graph.nodes:
                                        tgt = str(getattr(gn, "target", ""))
                                        if "gaga4_mh_moe" in tgt and gn.op == "call_function":
                                            for a in gn.args:
                                                if isinstance(a, int):
                                                    baked_key = a
                                            if int(__import__("os").environ.get("RANK", "0")) == 0 and getattr(self, "_moe_cmp_n", 0) == 0:
                                                print(
                                                    f"[moe-inner] {node.name} nargs={len(gn.args)} "
                                                    f"ints={[a for a in gn.args if isinstance(a, int)]} "
                                                    f"baked_key={baked_key}",
                                                    flush=True,
                                                )
                                    x = tensors[0]
                                    rest = tensors[1:]
                                    n_none = 23 - len(rest)
                                    if n_none < 0:
                                        n_none = 0
                                    key = 0 if baked_key is None else baked_key
                                    py = gaga4_mh_moe(x, *rest, *([None] * n_none), key)
                                    out = env[node]
                                    ot_ = out[0] if isinstance(out, (tuple, list)) else out
                                    pt = py[0] if isinstance(py, (tuple, list)) else py
                                    self._moe_cmp_n = getattr(self, "_moe_cmp_n", 0) + 1
                                    if int(__import__("os").environ.get("RANK", "0")) == 0:
                                        x0 = s_args[0]
                                        print(
                                            f"[moe-cmp] {node.name} x={tuple(x0.shape)}{x0.dtype} "
                                            f"nargs={len(s_args)} n={self._moe_cmp_n}",
                                            flush=True,
                                        )
                                        if isinstance(ot_, torch.Tensor) and isinstance(pt, torch.Tensor) and ot_.shape == pt.shape:
                                            d = (ot_.detach().float() - pt.detach().float()).abs()
                                            print(
                                                f"[moe-cmp] {node.name} max_abs={float(d.max()):.6g} "
                                                f"mean_abs={float(d.mean()):.6g} same={float(d.max()) < 1e-3} "
                                                f"gm_std={float(ot_.float().std()):.5g} py_std={float(pt.float().std()):.5g}",
                                                flush=True,
                                            )
                                        else:
                                            print(
                                                f"[moe-cmp] shape gm={getattr(ot_, 'shape', type(ot_))} py={getattr(pt, 'shape', type(pt))}",
                                                flush=True,
                                            )
                                except Exception as exc:
                                    if int(__import__("os").environ.get("RANK", "0")) == 0:
                                        print(f"[moe-cmp] failed: {exc}", flush=True)
                        
                        if self._call_idx == 3:
                            out = env[node]
                            cand = out[0] if isinstance(out, (tuple, list)) else out
                            if (
                                isinstance(cand, torch.Tensor)
                                and cand.dim() == 2
                                and cand.shape[-1] == 3072
                            ):
                                layer_i = max(0, getattr(self, "_fa_layer_i", 1) - 1)
                                projs = getattr(self, "_stash_attn_proj", None)
                                if projs is None:
                                    projs = {}
                                    self._stash_attn_proj = projs
                                seqs = getattr(self, "_stash_h3072", None)
                                if seqs is None:
                                    seqs = {}
                                    self._stash_h3072 = seqs
                                seq = seqs.setdefault(layer_i, [])
                                seq.append(cand.detach().clone())
                                all3072 = getattr(self, "_stash_h3072_all", None)
                                if all3072 is None:
                                    all3072 = []
                                    self._stash_h3072_all = all3072
                                if len(all3072) < 24:
                                    all3072.append(cand.detach().clone())
                                if layer_i not in projs:
                                    projs[layer_i] = cand.detach().clone()
                                    pass  # keep collecting (T,3072) until next FA
                        if (
                            self._call_idx == 3
                            and len(s_args) >= 10
                            and type(getattr(self.graph_module, node.target)).__name__ == "GraphModule"
                            and isinstance(s_args[0], torch.Tensor)
                            and s_args[0].dim() == 4
                        ):
                            self._fa_cmp_done = True
                            try:
                                from model.gaga4.modeling import gaga4_fa_with_sink_cp

                                q, k, v = s_args[0], s_args[2], s_args[3]
                                sink, cuq, cuk, cp = s_args[4], s_args[5], s_args[7], s_args[9]
                                blob = (
                                    q.detach().clone(),
                                    k.detach().clone(),
                                    v.detach().clone(),
                                )
                                if node.name == "submod_1":
                                    self._stash_qkv = blob
                                else:
                                    self._stash_qkv_last = blob
                                stashes = getattr(self, "_stash_qkv_layers", None)
                                if stashes is None:
                                    stashes = {}
                                    self._stash_qkv_layers = stashes
                                    self._fa_layer_i = 0
                                stashes[self._fa_layer_i] = blob
                                self._fa_layer_i += 1
                                self._stash_attn_proj_pending = True
                                py = gaga4_fa_with_sink_cp(q, k, v, sink, cuq, cuk, cp, -1.0)
                                out = env[node]
                                ot = out[0] if isinstance(out, (tuple, list)) else out
                                pt = py[0] if isinstance(py, (tuple, list)) else py
                                if _rank0:
                                    print(
                                        f"[fa-cmp] q={tuple(q.shape)}{q.dtype} k={tuple(k.shape)}{k.dtype} "
                                        f"v={tuple(v.shape)}{v.dtype} sink={tuple(sink.shape)} mean={float(sink.float().mean()):.5g} "
                                        f"cuq={cuq.tolist()} cuk={cuk.tolist()} cp={cp.tolist()}",
                                        flush=True,
                                    )
                                    if isinstance(ot, torch.Tensor) and isinstance(pt, torch.Tensor) and ot.shape == pt.shape:
                                        d = (ot.detach().float() - pt.detach().float()).abs()
                                        print(
                                            f"[fa-cmp] {node.name} out={tuple(ot.shape)} "
                                            f"max_abs={float(d.max()):.6g} mean_abs={float(d.mean()):.6g} "
                                            f"gm_std={float(ot.float().std()):.5g} py_std={float(pt.float().std()):.5g} "
                                            f"same={float(d.max()) < 1e-3}",
                                            flush=True,
                                        )
                                    else:
                                        print(
                                            f"[fa-cmp] shape mismatch gm={getattr(ot, 'shape', type(ot))} "
                                            f"py={getattr(pt, 'shape', type(pt))}",
                                            flush=True,
                                        )
                            except Exception as exc:
                                if _rank0:
                                    print(f"[fa-cmp] {node.name} failed: {exc}", flush=True)
                        if _rank0 and self._call_idx == 3 and type(getattr(self.graph_module, node.target)).__name__ == "GraphModule" and node.name in {"submod_1", "submod_3", "submod_5", "submod_7"}:
                            bits = []
                            for j, a in enumerate(s_args):
                                if isinstance(a, torch.Tensor):
                                    bits.append(f"i{j}{tuple(a.shape)}{a.dtype}")
                                else:
                                    bits.append(f"i{j}={type(a).__name__}:{a}")
                            print(f"[gm-args] {node.name} {bits}", flush=True)
                        if _rank0 and self._call_idx == 3:
                            out = env[node]
                            ts = [out] if isinstance(out, torch.Tensor) else (
                                [t for t in out if isinstance(t, torch.Tensor)] if isinstance(out, (tuple, list)) else []
                            )
                            for j, t in enumerate(ts[:3]):
                                if t.is_floating_point() and t.dim() >= 1 and t.numel() > 256:
                                    f = t.detach().float()
                                    print(
                                        f"[fx-sub] {node.name} o{j} {tuple(t.shape)} "
                                        f"mean={float(f.mean()):.5g} std={float(f.std()):.5g} "
                                        f"absmax={float(f.abs().max()):.5g}",
                                        flush=True,
                                    )
                        if _rank0:
                            out = env[node]
                            ts = []
                            if isinstance(out, torch.Tensor):
                                ts = [out]
                            elif isinstance(out, (tuple, list)):
                                ts = [t for t in out if isinstance(t, torch.Tensor)]
                            bits = []
                            bad = False
                            for j, t in enumerate(ts[:3]):
                                if t.is_floating_point():
                                    fin = bool(torch.isfinite(t).all())
                                    bad = bad or (not fin)
                                    bits.append(f"o{j} finite={fin} {tuple(t.shape)} {t.device.type}")
                                else:
                                    bits.append(f"o{j} {t.dtype} {t.device.type}")
                            n_cpu_w = sum(
                                1
                                for a in s_args
                                if isinstance(a, torch.Tensor) and a.device.type == "cpu"
                            )
                            _probe = os.environ.get("GAGA4_NAN_PROBE", "").strip() in {"1", "true", "yes"}
                            if bad or (_probe and self._call_idx <= 3):
                                print(
                                    f"[offload submod] call={self._call_idx} {node.name} cpu_args={n_cpu_w} {' '.join(bits)}",
                                    flush=True,
                                )
                        del s_args, s_kwargs

                if need_profile:
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
                    self.profiler.end_compute_profile(node.name, self.compute_stream)

            elif node.op == "call_function":
                if node.target == operator.getitem:
                    parent_node, idx = node.args
                    env[node] = env[parent_node][idx]
                else:
                    def _ensure_cuda(v):
                        if isinstance(v, torch.Tensor) and not v.is_cuda:
                            return v.to("cuda", non_blocking=True)
                        return v

                    with torch.cuda.stream(self.compute_stream):
                        f_args = map_arg(node.args, lambda n: _ensure_cuda(env[n]))
                        f_kwargs = map_arg(node.kwargs, lambda n: _ensure_cuda(env[n]))
                        env[node] = node.target(*f_args, **f_kwargs)

            elif node.op == "output":
                if self.second_call:
                    self._finalize_warmup()
                    self.second_call = False
                if self.warmup:
                    self.second_call = True
                    self.warmup = False

                return map_arg(node.args[0], lambda n: env[n])

            # Memory Management
            for input_node in node.all_input_nodes:
                current_user_counts[input_node] -= 1
                if current_user_counts[input_node] == 0:
                    if input_node in env:
                        tensor_obj = env[input_node]
                        if isinstance(tensor_obj, torch.Tensor) and tensor_obj.is_cuda:
                            tensor_obj.record_stream(self.compute_stream)
                        del env[input_node]
        return None


class OffloadWrapper:
    @observe_lifecycle("offload_wrap")
    def __init__(self, graph_module: torch.fx.GraphModule, compile_config: CompileConfig):
        self.executor = OffloadExecutor(graph_module, compile_config)

    def __call__(self, *args):
        return self.executor(*args)
