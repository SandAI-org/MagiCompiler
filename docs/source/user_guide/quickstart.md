# QuickStart

```{contents}
:local: true
```

## One Decorator to Rule Them All (`@magi_compile`)

Remove scattered `torch.compile` or `torch.compiler.disable` calls. Decorate your
core Transformer block once for automatic full-graph capture and dynamic shape
support (defaulting to dim 0).

```python
import torch
from torch import nn
from magi_compiler import magi_compile

# Decorate your core module once. No more scattered compile tweaks!
@magi_compile
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = Attention(hidden_dim)
        self.mlp = MLP(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(x, mask)
        x = x + self.mlp(x)
        return x

model = TransformerBlock(hidden_dim=1024).cuda()

# Execute normally - whole-graph compilation handles dynamic batches automatically!
out = model(torch.randn(4, 128, 1024, device="cuda"), None)
out = model(torch.randn(8, 128, 1024, device="cuda"), None)
```

## Bridge Custom Kernels (`@magi_register_custom_op`)

Using custom kernels (FlashAttention, MoE routers) that break FX tracing? Don't
disable compilation. Wrap them to teach the compiler how to handle them during
graph partitioning and recomputation.

```python
from magi_compiler import magi_register_custom_op

@magi_register_custom_op(
    name="athena::flash_attn",
    infer_output_meta_fn=["q"],       # Output shape matches parameter 'q'
    is_subgraph_boundary=True,        # Split graph here for subgraph compilation
    is_compute_sensitive=True,        # Retain this output during recomputation
)
def flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    ... # Your custom kernel or C++ extension
```

## Advanced Configurations

Explore `magi_compiler/config.py` for power-user features like custom backend
toggles and fine-grained memory management.

:::{note}
Comprehensive guides for popular training/inference frameworks are coming soon.
:::
