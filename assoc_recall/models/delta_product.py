"""DeltaProduct block for hybrid architectures.

Wraps flash-linear-attention's DeltaProduct as a drop-in replacement
for MambaBlock in the hybrid model. Requires: pip install flash-linear-attention

DeltaProduct uses iterated rank-K DeltaNet updates, significantly more
expressive than rank-1 (Mamba/DeltaNet) at scale.

Reference: Yang et al. (2025) "Gated Delta Networks with Softmax Attention"
"""

import torch
import torch.nn as nn

try:
    # flash-linear-attention renamed the DeltaProduct layer to GatedDeltaProduct
    # (>=0.2). Fall back to the old name for older installs.
    try:
        from fla.layers import GatedDeltaProduct as DeltaProductAttention
    except ImportError:
        from fla.layers import DeltaProductAttention
    HAS_FLA = True
except ImportError:
    HAS_FLA = False


class DeltaProductBlock(nn.Module):
    """Pre-norm DeltaProduct block matching MambaBlock's interface."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        if not HAS_FLA:
            raise ImportError(
                "flash-linear-attention not installed. "
                "Install with: pip install flash-linear-attention"
            )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        num_heads = getattr(config, 'num_attention_heads', 1)
        head_dim = config.hidden_size // num_heads

        self.dp_attn = DeltaProductAttention(
            hidden_size=config.hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        # fla's chunked gated-DeltaProduct kernel only supports bf16 (no fp32).
        # Localize the low precision to this block; keep the residual stream fp32.
        with torch.autocast(device_type=hidden_states.device.type, dtype=torch.bfloat16):
            out = self.dp_attn(hidden_states)
        # fla layers return (output, attn_weights, past_key_values)
        if isinstance(out, tuple):
            out = out[0]
        return out.to(residual.dtype) + residual
