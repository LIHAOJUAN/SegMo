import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, BatchFeature
from PIL import Image
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from transformers.cache_utils import DynamicCache
import types
import logging

from vision_parallel.config import Config
from vision_parallel.request import InputData
from vision_parallel.kv_cache_manager import from_batch_splits_with_padding
from util.performance_monitor import get_monitor

processor = None
tokenizer = None
sys_msg = None

logger = logging.getLogger("vision-parallel")

def initialize(model_path, sys_message):
    global processor, tokenizer, sys_msg
    sys_msg = sys_message
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
def get_embedding(model, inputs: list[InputData]):
    prepare_input_start_time = time.perf_counter()
    data = {
        "tgt_sizes": [],
        "pixel_values": [],
        "image_bound": [],
        "input_ids": None,
    }

    keys = ["tgt_sizes", "image_bound"]
    empty_value = {
        "tgt_sizes": torch.zeros((0, 2), dtype=torch.long, device=model.device),
        "image_bound": torch.zeros((0, 2), dtype=torch.long, device=model.device),
    }
    for input_data in inputs:
        if all(key in input_data.data for key in keys):
            for key in keys:
                data[key].extend(input_data.data[key])
        else:
            for key in keys:
                data[key].append(empty_value[key])
    
    for input_data in inputs:
        if "pixel_values" in input_data.data and len(input_data.data["pixel_values"]) > 0:
            for imgs in input_data.data["pixel_values"]:
                data["pixel_values"].append([img.to(model.device) for img in imgs])
        else:
            data["pixel_values"].append([])

    # Collect all input_ids
    input_ids_list = [input_data.all_input_ids[:, input_data.computed_tokens:] for input_data in inputs]
    # Pad input_ids and remove the second dimension if present
    data["input_ids"] = torch.nn.utils.rnn.pad_sequence(
        [ids.squeeze(0) for ids in input_ids_list], batch_first=True, padding_value=0
    ).to(model.device)

    attention_mask = (data["input_ids"] != 0).to(model.device)
    position_ids_list = [input_data.data["position_ids"] for input_data in inputs]
    position_ids = torch.nn.utils.rnn.pad_sequence(
        [ids.squeeze(0) for ids in position_ids_list], batch_first=True, padding_value=0
    ).to(model.device)

    prepare_input_end_time = time.perf_counter()
    get_monitor().record("prepare_input_latency", prepare_input_end_time-prepare_input_start_time)

    vllm_embedding, _ = model.get_vllm_embedding(data)

    if model.config.init_audio:
        vllm_embedding = model.get_omni_embedding(
            data, input_embeddings=vllm_embedding, chunk_length=model.config.audio_chunk_length
        )
    
    return vllm_embedding, attention_mask, position_ids

def forward(model, inputs: list[InputData], output_attention=False):
    with torch.no_grad():
        vllm_embedding, attention_mask, position_ids = get_embedding(model, inputs)
        past_kv_cache = from_batch_splits_with_padding([input_data.past_key_values for input_data in inputs])
        start_time = time.perf_counter()
        metric_name = "decode_latency" if vllm_embedding.shape[1] == 1 else "prefill_latency"
        output = model.llm(
            inputs_embeds=vllm_embedding,
            # attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_kv_cache,
            use_cache=True,
            output_attentions=output_attention,
        )
        end_time = time.perf_counter()
        get_monitor().record(metric_name, end_time-start_time)
    return output

def vpm_embedding_forward(
    self,
    pixel_values: torch.FloatTensor,
    patch_attention_mask: torch.BoolTensor,
    tgt_sizes: Optional[torch.IntTensor] = None,
) -> torch.Tensor:
    batch_size = pixel_values.size(0)

    patch_embeds = self.patch_embedding(pixel_values)
    embeddings = patch_embeds.flatten(2).transpose(1, 2)

    max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
    max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
    boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
    position_ids = torch.full(
        size=(
            batch_size,
            max_nb_patches_h * max_nb_patches_w,
        ),
        fill_value=0,
    )

    for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
        if tgt_sizes is not None:
            nb_patches_h = tgt_sizes[batch_idx][0]
            nb_patches_w = tgt_sizes[batch_idx][1]
        else:
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

        fractional_coords_h = torch.linspace(0, 1 - 1/nb_patches_h, nb_patches_h)
        fractional_coords_w = torch.linspace(0, 1 - 1/nb_patches_w, nb_patches_w)

        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

        pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
        position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

    position_ids = position_ids.to(self.position_embedding.weight.device)

    embeddings = embeddings + self.position_embedding(position_ids)
    return embeddings

def prepare_model(model):
    logger.debug(f"正在为 MiniCPM 模型应用 embedding forward patch。")
    model.vpm.embeddings.forward = types.MethodType(vpm_embedding_forward, model.vpm.embeddings)
