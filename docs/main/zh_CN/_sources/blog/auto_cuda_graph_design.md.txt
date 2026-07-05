---
blogpost: true
date: 2026-03-23
author: Zhiyao Cen
category: Design
tags: cuda-graph, inference, memory
---

# AutoCudaGraph Design

## Overview
AutoCudaGraph is a CUDA Graph optimization module integrated into the MagiCompiler framework, designed to automate CUDA Graph capture, caching, replay, and tensor memory management for PyTorch-based neural network inference. It targets Transformer architectures with dynamic sequence lengths, optimizing kernel execution by reusing pre-captured computation graphs and static tensor buffers. Core Goals:
* Automate CUDA Graph lifecycle (capture/replay/cache) with minimal code intrusion
* Support dynamic shape adaptation (sequence length expansion)
* Optimize memory efficiency via global memory pool and static tensor reuse
* Ensure consistency between cached graphs and runtime inputs/outputs

## Key Components

### CudaGraphMgr (Core Manager)
Singleton class managing all CUDA Graph operations:
```python
class CudaGraphMgr:
    def __init__(self):
        self.cache: Dict[StaticSignature, StaticTensorEntry] = dict()
        self.graph_mem_pool: Optional[torch.cuda.graph_pool_handle] = None
```

**Core Methods**
| Method | Purpose |
|---------------------------------|----------------------------------------|
| run()                           | Main entry: Replay cached graph or warm up & capture new graph|
| wrapped_graph_capture()         | Capture CUDA Graph with sliced static input/output tensors |
| wrapped_graph_replay()          | Replay cached CUDA Graph with sliced static tensors and output template wrapping
| get_expanded_static_tensors()   | Expand static tensors, reuse buffers if dimensionally compatible|

### Signature System

StaticSignature
```python
@dataclass(unsafe_hash=True)
class StaticSignature(HashableDataclass):
    func_name: str = ""
    tensor_static_infos: Tuple[TensorStaticInfo, ...] = tuple()
```
* Encodes fixed properties of input tensors (dtype, static dimensions)
* Used as primary key for static tensor buffer caching

DynamicSignature
```python
@dataclass(unsafe_hash=True)
class DynamicSignature(HashableDataclass):
    tensor_dynamic_infos: Tuple[TensorDynamicInfo, ...] = tuple()
    literals_info: LiteralsInfo = None
```
* Tracks dynamic dimensions (sequence length) and literal parameters
* Secondary key for graph entry lookup

### Tensor Management
```python
@dataclass
class StaticTensorEntry:
    input_tensors: Optional[List[torch.Tensor]] = None
    output_tensors: Optional[List[torch.Tensor]] = None
    template_entry_dict: Dict[DynamicSignature, OutputTemplateEntry] = None
```
* Memory Reuse: Reuse existing tensor buffers when possible to avoid reallocation
* Dynamic Expansion: Only expand static tensors when new input dimensions exceed current buffer size
* Shape Validation: Ensure static dimensions (non-sequence) match between cached and new tensors

### Graph Management
```python
@dataclass
class GraphEntry:
    graph: Optional[torch.cuda.CUDAGraph] = None
    inconsistent: bool = False
    invalid: bool = False

@dataclass
class OutputTemplateEntry:
    graph_entry_dict: Dict[int, GraphEntry] = None
    output_template: Any = None
```
* Graph State Tracking: GraphEntry tracks CUDA Graph instances and validity states to control replay eligibility.
* Layer-wise Organization: OutputTemplateEntry maps dynamic signatures to per-layer GraphEntry for layer-specific graph reuse.
* Output Consistency: output_template preserves output object structure to ensure consistent result wrapping during replay.

## Execution Flow
### Inline Replay (Fast Path)
* Extract input signatures from runtime arguments
* Look up cached CUDA Graph via StaticSignature + DynamicSignature + layer number
* Validate graph consistency (not inconsistent/invalid)
* Reuse static tensors with dynamic slicing
* Replay graph and return sliced output
### Graph Capture (Slow Path)
Triggered when no valid cached graph exists or tensor expansion is needed:
* Execute function to get output tensors
* Ensure input signatures match post-warmup
* Expand static buffers if new shapes require it
* Capture new CUDA Graph with static tensors
* Store new graph and update tensor entries
* Return warmup execution output as final result
### Sequence Length Handling
* Only last dimension is static for ND tensors (ND > 1)
* All dimension is dynamic for 1D tensors (ND=1)
* Automatic buffer expansion for increasing sequence lengths
* Invalidates old graphs when tensors are expanded

## Examples
```python
import torch
import torch.nn as nn
from magi_compiler.magi_backend.cuda_graph_mgr import cuda_graph_mgr, cuda_graph_enable_if

class SimpleTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.layer_number = 0

    @cuda_graph_enable_if(lambda: torch.cuda.is_available())
    def forward(self, x: torch.Tensor):
        attn_out, _ = self.self_attn(x, x, x)
        out = self.linear(self.layer_norm(x + attn_out))
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformerLayer(hidden_dim=1024, num_heads=8).to(device).eval()
graph_mgr = cuda_graph_mgr()

with torch.no_grad():
    input_1 = torch.randn(2, 512, 1024, device=device)
    output_1 = model(input_1)
    print(f"First run (graph capture): Output shape = {output_1.shape}")
    print(f"Cached graphs count: {graph_mgr.graph_count}")

    input_2 = torch.randn(2, 512, 1024, device=device)
    output_2 = model(input_2)
    print(f"Second run (graph replay): Output shape = {output_2.shape}")
    print(f"Cached graphs count: {graph_mgr.graph_count}")

    input_3 = torch.randn(2, 1024, 1024, device=device)
    output_3 = model(input_3)
    print(f"Third run (tensor expansion): Output shape = {output_3.shape}")
    print(f"Cached graphs count: {graph_mgr.graph_count}")
    print(f"Static tensor memory usage: {graph_mgr.tensor_mem_size:.2f} MB")

    print("\nCUDA Graph Cache Details:")
    print(graph_mgr.formatted_cache_str())
```

## Limitations and Constraints
* No support for data-dependent control flow in captured functions
* Graph capture fails if function contains CPU/GPU synchronization
* Only supports CUDA tensors (CPU tensors trigger fallback)
* Custom input classes must inherit from InplaceSubstituteFakeClass
* Assumes input tensors of captured graphs are not reused externally (risk of cross-scenario static tensor reuse)
* Relies on identical function, input tensors shapes, and constants for valid graph reuse
* No support for multi-stream execution scenarios

## Best Practices
* Dynamic Dimensions: Tensor use sequence length as dimension 0 where possible
* Monitor Memory Usage: Track graph_mem_pool_size and tensor_mem_size to avoid OOM
* Specify Layer IDs: Use layer_number to distinguish graphs across different models/layers
* LRU Cache (Future): Implement cache eviction to limit total graph/tensor count
