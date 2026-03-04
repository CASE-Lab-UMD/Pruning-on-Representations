from typing import Any, Optional, Tuple


def compute_kv_cache_idx(drop_attn_list, layer_idx: int) -> int:
    """Compute KV-cache layer index after skipping dropped attention layers."""
    if not drop_attn_list or layer_idx <= 0:
        return 0
    return sum(1 for i in range(layer_idx) if not drop_attn_list[i])


def pack_decoder_layer_outputs(
    hidden_states,
    output_attentions: bool,
    use_cache: bool,
    drop_attn: bool,
    self_attn_weights: Optional[Any] = None,
    present_key_value: Optional[Any] = None,
) -> Tuple[Any, ...]:
    """
    Build the decoder layer output tuple while handling dropped-attention branches.
    This avoids unbound local variables when an attention block is removed.
    """
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (None if drop_attn else self_attn_weights,)
    if use_cache and not drop_attn:
        outputs += (present_key_value,)
    return outputs
