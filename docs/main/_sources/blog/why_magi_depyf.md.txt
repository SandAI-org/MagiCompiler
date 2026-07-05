---
blogpost: true
date: 2026-03-23
author: MagiCompiler Team
category: Design
tags: magi_depyf, observability, debugging
---

# Why magi_depyf

`magi_depyf` is the compilation observability layer for `MagiCompiler`.
It answers three practical questions:

1. **What exactly happened during compilation?**
2. **Did the outcome match expectations?** (for example: graph split shape, cache reuse, retry behavior)
3. **If something failed, where should I look first?**

---

## 1. Positioning: Built-in observability for MagiCompiler

In a real MagiCompiler pipeline, compilation spans multiple stages: Dynamo capture, Magi backend graph transforms/splitting, Inductor codegen, and AOT/JIT reuse.
Plain logs are often not enough to reliably answer:

- Did the failure happen at full-graph level or in a specific subgraph?
- Did a pass change graph behavior unexpectedly?
- Why did this run miss cache?

`magi_depyf` writes these signals as structured events and artifacts on disk, so you can replay, compare, and debug deterministically.

---

## 2. Primary scenario: `magi_compile`

### 2.1 Recommended usage model

For `magi_compile` users, `magi_depyf` is most useful as a built-in observability layer:

- Streams events to `timeline_events/timeline.jsonl` during compilation
- Writes event artifacts under `timeline_events/files/`
- Exports a structured `compiled_functions/` artifact tree

In most cases, you do not need to manually wire custom hooks. Run through the MagiCompiler path and inspect the output directory.

### 2.2 What you get automatically in the Magi compile path

For the `magi_compile` scenario, you do **not** need to call `explain_compilation` manually.
As long as the model runs through MagiCompiler, `magi_depyf` outputs are written automatically.

For example, one real run produced:

- `magi_depyf_torch_compile_debug/magi_depyf/run_20260322_192147/model_1_TwoLayerTransformer_rank_0/timeline_events`

In general, the output pattern is:

- `<cache_root_dir>/magi_depyf/run_*/model_*_rank_*/timeline_events/`

Key artifacts under this directory:

- `timeline.jsonl`: streaming event log written during compilation
- `files/0000_*`, `files/0001_*`, ... event folders with attached artifacts
- `files/*/attributes.json`: structured metadata for each event folder

Related compiled artifact view is also emitted under:

- `<cache_root_dir>/magi_depyf/run_*/model_*_rank_*/compiled_functions/`

See the full runnable example:

- `magi_compiler/magi_depyf/example/magi_compile_transformer_example.py`

Typical questions this path answers quickly:

- Did fullgraph/subgraph events happen in the expected order?
- Which pass changed graph structure?
- Did `RestartAnalysis` happen, and was cache load skipped as expected?
- Which event folder contains the exact graph/code snapshot for debugging?

---

## 3. Secondary scenario: plain `torch.compile` (manual entry)

`magi_depyf` also works with plain `torch.compile`, but this is a secondary workflow and requires manual context wrapping:

```python
import torch

from magi_compiler.magi_depyf.inspect import explain_compilation


@torch.compile
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = -b
    return x * b


with explain_compilation("./magi_depyf_torch_compile_debug"):
    for _ in range(10):
        toy_example(torch.randn(10), torch.randn(10))
```

See the concrete example here:

- `magi_compiler/magi_depyf/example/torch_compile_toy_example.py`

This demo shows the minimal way to use `explain_compilation` in a pure `torch.compile` setup.

---

## 4. Event model (concise)

Each event contains:

- `name`: event name (with `fullgraph_` or `subgraph_N_` prefix)
- `attributes`: structured metadata (for example `lifecycle_name`, `runtime_shape`, `graph_index`, `reason`)
- `files`: attached text artifacts (graph code, inductor output, etc.)

### 4.1 Lifecycle naming pattern

Lifecycle events are normalized into the following pattern:

- `*_before_<lifecycle_name>`
- `*_after_<lifecycle_name>`
- `*_failed_<lifecycle_name>`
- `*_skip_<lifecycle_name>`

Where `*` is `fullgraph` or `subgraph_<N>`.

### 4.2 Representative examples

- `fullgraph_before_graph_split`
- `fullgraph_after_graph_split`
- `fullgraph_before_compiler_manager_compile`
- `fullgraph_after_compiler_manager_load`
- `subgraph_2_before_postcleanuppass`
- `subgraph_2_after_postcleanuppass`

---

## 5. Output layout (current implementation)

```text
debug_output/
  timeline_events/
    timeline.jsonl
    files/
      0000_fullgraph_.../
        attributes.json
        *.py / *.txt
      0001_subgraph_.../
        submod_<N>/
          attributes.json
          *.py / *.txt

  compiled_functions/
    <function_name>/
      overview.md
      decompiled_code.py
      entry_0/
        guards.txt
        compiled_fns/
        piecewise_subgraphs/
```

---

## 6. Recommended triage order

When debugging a compile issue, use this order:

1. `timeline.jsonl`: verify event ordering first
2. `*_failed_*` / cache-related lifecycle events (`compiler_manager_load`, `compiler_manager_cache_store`)
3. Graph files attached to relevant events: verify before/after pass graph state
4. `compiled_functions/*/overview.md`: verify final compiled artifact structure

The key is not just absolute timing. The key is:

- what happened,
- whether it matched expectations,
- and where to drill down when it failed.

---

## 7. Compatibility and entry-point summary

- **Primary entry**: MagiCompiler main path (`magi_compile`)
- **Optional entry**: plain `torch.compile` + manual `explain_compilation`

One-line summary:

`magi_depyf` is first and foremost MagiCompiler's internal observability infrastructure, while still being usable as a manual troubleshooting tool for plain `torch.compile` when needed.
