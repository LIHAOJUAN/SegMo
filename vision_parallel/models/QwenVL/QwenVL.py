import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, BatchFeature
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from typing import Tuple
import types
import logging

from vision_parallel.config import Config
from vision_parallel.request import InputData
from vision_parallel.kv_cache_manager import batch_kv_caches
import vision_parallel.video_util as video_util
from util.performance_monitor import get_monitor
from util.analyse_attention import calculate_frame_attention_scores, draw_frame_attention_heatmap

default_tts_chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>' }}{% endif %}"

processor = None
tokenizer = None

spatial_merge_size = None
image_token_id = None
video_token_id = None
vision_start_token_id = None

logger = logging.getLogger("vision-parallel")

def initialize(model_path):
    global processor, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # processor = AutoProcessor.from_pretrained(model_path, max_pixels=448*448)
    processor = AutoProcessor.from_pretrained(model_path)


def generate(model, inputs: list[InputData]):
    global tokenizer
    gen_kwargs = {
        "max_new_tokens": 128,
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
    }
    input = inputs[0].data
    del input["mrope_position_deltas"]
    del input["position_ids"]
    input.to(model.device)
    ret = model.generate(
        **input,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True if gen_kwargs["temperature"] > 0 else False,
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        num_beams=gen_kwargs["num_beams"],
        max_new_tokens=gen_kwargs["max_new_tokens"],
    )
    return ret

def forward(model, inputs: list[InputData], output_attention=False):
    with torch.no_grad():
        input = inputs[0]
        input_ids = input.all_input_ids[:, input.computed_tokens:]
        input_ids = input_ids.to(model.device)
        pixel_values=input.data["pixel_values"].to(model.device) if "pixel_values" in input.data else None
        image_grid_thw=input.data["image_grid_thw"].to(model.device) if "image_grid_thw" in input.data else None
        pixel_values_videos=input.data["pixel_values_videos"].to(model.device) if "pixel_values_videos" in input.data else None
        video_grid_thw=input.data["video_grid_thw"].to(model.device) if "video_grid_thw" in input.data else None
        position_ids = input.data['position_ids'].to(model.device)
        
        vision_encoder_start_time = time.perf_counter()
        inputs_embeds = get_inputs_embeds(model, input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw)
        vision_encoder_end_time = time.perf_counter()
        get_monitor().record("vision_encoder_latency", vision_encoder_end_time-vision_encoder_start_time)
        
        forward_start_time = time.perf_counter()
        metric_name = "decode_latency" if input_ids.shape[1] == 1 else "prefill_latency"
        output = model(
            inputs_embeds = inputs_embeds,
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            past_key_values=input.past_key_values,
            use_cache=True,
            output_attentions=output_attention,
        )
        forward_end_time = time.perf_counter()
        get_monitor().record(metric_name, forward_end_time-forward_start_time)
    return output

def get_inputs_embeds(model, input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw):
    inputs_embeds = model.model.embed_tokens(input_ids)
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.get_dtype())
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_mask = (
            (input_ids == model.config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.get_dtype())
        video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )
        video_mask = (
            (input_ids == model.config.video_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    
    return inputs_embeds


def get_rope_index(
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    start_position: Optional[int] = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    global spatial_merge_size, image_token_id, video_token_id, vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
        )
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i].to(input_ids.device) == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        position_ids = position_ids + start_position
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas

def prepare_model(model):
    logger.debug(f"正在初始化 get_rope_index 方法 ")
    global spatial_merge_size, image_token_id, video_token_id, vision_start_token_id
    spatial_merge_size = model.config.vision_config.spatial_merge_size
    image_token_id = model.config.image_token_id
    video_token_id = model.config.video_token_id
    vision_start_token_id = model.config.vision_start_token_id
