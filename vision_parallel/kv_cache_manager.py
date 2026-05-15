from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
import torch
from transformers.cache_utils import DynamicCache

class KVCacheManager(ABC):
    
    @abstractmethod
    def merge_kv_caches(self, kv_caches: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pass
    
    @abstractmethod
    def update_kv_cache(self, existing_cache: Dict[str, torch.Tensor], new_cache: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.mixtral.modeling_mixtral.apply_rotary_pos_emb
def apply_rotary_pos_emb(k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed

def rotate_kv_with_rope(k, rotary_emb, n):
    """
    Apply RoPE rotation to the key in the KV cache so positional encoding stays aligned.
    The value does not need to be rotated.

    Args:
        kv: tuple(key, value), where key and value have shape (batch, num_heads, head_dim, seq_len)
        rotary_emb: Qwen2RotaryEmbedding object.
        n: Length to rotate, i.e. seq_len.

    Returns:
        (rotated_key, rotated_value)
    """
    key = k
    device = key.device
    seq_len = key.shape[2]
    # Generate cos and sin.
    cos, sin = rotary_emb(key, n)
    # Set every cos/sin element to the value at position seq_len.
    target_cos = cos[-1]
    target_sin = sin[-1]
    cos = target_cos.repeat(seq_len, 1)
    sin = target_sin.repeat(seq_len, 1)
    
    # Generate position_ids assuming batch=1.
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    # Call apply_rotary_pos_emb defined in this file.
    rotated_key = apply_rotary_pos_emb(key, cos, sin, position_ids, unsqueeze_dim=1)
    return rotated_key

def merge_kv_caches(rotary_emb, kv_caches: list[DynamicCache], target_device):
    move_kv_cache_to_device(kv_caches[0], target_device)
    merged_kv_cache = kv_caches[0]
    layer_num = len(merged_kv_cache)
    n = merged_kv_cache.get_seq_length()
    for kv_cache in kv_caches[1:]:
        move_kv_cache_to_device(kv_cache, target_device)
        for i in range(layer_num):
            # Apply RoPE rotation to the new key_cache.
            rotated_key = rotate_kv_with_rope(kv_cache.key_cache[i], rotary_emb, n)
            merged_kv_cache.update(rotated_key, kv_cache.value_cache[i], layer_idx=i)
        n += kv_cache.get_seq_length()
    return merged_kv_cache

def merge_kv_caches_simple(kv_caches: list[DynamicCache], target_device):
    move_kv_cache_to_device(kv_caches[0], target_device)
    merged_kv_cache = kv_caches[0]
    layer_num = len(merged_kv_cache)
    for kv_cache in kv_caches[1:]:
        move_kv_cache_to_device(kv_cache, target_device)
        for i in range(layer_num):
            merged_kv_cache.update(kv_cache.key_cache[i], kv_cache.value_cache[i], layer_idx=i)
    return merged_kv_cache

def move_kv_cache_to_device(cache: DynamicCache, device):
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i].to(device)
        cache.value_cache[i] = cache.value_cache[i].to(device)

def from_batch_splits_with_padding(splits: List["DynamicCache"]) -> "DynamicCache":
    """
    Merge multiple DynamicCache objects into one along the batch dimension and pad along the sequence dimension.
    Note that this assumes the original DynamicCache objects all have batch size 1.
    """
    cache = DynamicCache()
    for idx in range(len(splits[0])):
        key_cache = [current.key_cache[idx][0] for current in splits if current.key_cache[idx].numel()]
        value_cache = [current.value_cache[idx][0] for current in splits if current.value_cache[idx].numel()]
        if key_cache != []:
            layer_keys = torch.nn.utils.rnn.pad_sequence(key_cache, batch_first=True, padding_value=0.0)
            layer_values = torch.nn.utils.rnn.pad_sequence(value_cache, batch_first=True, padding_value=0.0)
            cache.update(layer_keys, layer_values, idx)
    return cache

def batch_kv_caches(kv_caches):
    """
    Concatenate multiple kv_cache objects along the batch dimension into one kv_cache.

    Args:
        kv_caches: List of kv_cache objects, where each kv_cache is a list of tuple(key, value).

    Returns:
        batched_kv_cache: A kv_cache whose keys and values are concatenated along the batch dimension.
    """
    batched_kv_cache = []
    num_layers = None
    for kv_cache in kv_caches:
        if kv_cache is not None:
            num_layers = len(kv_cache)
            break
    if num_layers is None:
        return None
    for layer_idx in range(num_layers):
        keys = [
            kv_cache[layer_idx][0].squeeze(0) if kv_cache is not None else torch.empty(0)
            for kv_cache in kv_caches
        ]
        values = [
            kv_cache[layer_idx][1].squeeze(0) if kv_cache is not None else torch.empty(0)
            for kv_cache in kv_caches
        ]
        batched_key = torch.nn.utils.rnn.pad_sequence(
            [key for key in keys], batch_first=True, padding_value=0.0
        )
        batched_value = torch.nn.utils.rnn.pad_sequence(
            [value for value in values], batch_first=True, padding_value=0.0
        )
        batched_kv_cache.append((batched_key, batched_value))
    return tuple(batched_kv_cache)
