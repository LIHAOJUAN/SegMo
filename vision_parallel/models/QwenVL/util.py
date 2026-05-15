import math, os, torch
import numpy as np
from PIL import Image
from moviepy import VideoFileClip
from copy import deepcopy
from decord import VideoReader, cpu 
from qwen_vl_utils import process_vision_info

from vision_parallel.models.QwenVL import QwenVL
from vision_parallel.request import InputData, ModelInput, Request
from vision_parallel.kv_cache_manager import merge_kv_caches_simple

def encode_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    duration = len(vr) / fps
    times = list(np.arange(0, duration, 1.0))
    frame_idx = [min(int(round(t * fps)), len(vr) - 1) for t in times]
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames]
    return frames, 1

def construct_frames_from_video(video_path, flatten=True, save_frames=False, save_path=None):
    video = VideoFileClip(video_path)
    num_units = math.ceil(video.duration)
    
    contents= []
    for i in range(0, num_units, 1):
        frame = video.get_frame(i)
        image = Image.fromarray((frame).astype(np.uint8))
        contents.append(image)
        if save_frames:
            video_dir = f"{save_path}/video"
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            image.save(f"{video_dir}/frame_{i}.png")
    
    return contents, 1

def get_target_input(question, video_content, zero_frames, index, video_slice_num):
    target_content = []
    # target_content.append({
    #     "type": "text",
    #     "text": f"There are {video_slice_num} video segments in total. This is the content of the {index}th video segment:"
    # })
    # for image in video_content:
    #     target_content.append({
    #         "type": "image",
    #         "image": image,
    #     })
    target_content.append({
        "type": "video",
        "video": video_content,
        # "total_pixels": 32768*28*28*0.9/video_slice_num
        "total_pixels": 21000*28*28/video_slice_num,
        "min_pixels": 16*28*28
    })
    target_content.append({
        "type": "text",
        "text": question
    })
    target_msg = {"role":"user", "content": target_content}
    global_content = []
    global_content.append({
        "type": "text",
        "text": "This is a global information summary of the video:"
    })
    # max_pixels = 32768*28*28*0.05/video_slice_num/len(zero_frames)
    max_pixels = 128*28*28
    for zero_frame in zero_frames:
        global_content.append({
            "type": "image",
            "image": zero_frame,
            "max_pixels": max_pixels
        })
    global_msg = {"role":"user", "content": global_content}
    final_msg = [global_msg, target_msg]
    # final_msg = [target_msg]
    text = QwenVL.processor.apply_chat_template(
        final_msg, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(final_msg)
    target_input = QwenVL.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

    return target_input

def get_target_inputs(prompt_header, video_contents, zero_frames, 
                      image_slice_num, calculate_attention = False):
    target_inputs = []

    start_position = 0
    for index, video_content in enumerate(video_contents):
        target_input = get_target_input(prompt_header, video_content, zero_frames, index+1, len(video_contents))
        target_input['position_ids'], mrope_position_deltas = QwenVL.get_rope_index(
            input_ids=target_input['input_ids'],
            image_grid_thw=target_input['image_grid_thw'] if 'image_grid_thw' in target_input else None,
            video_grid_thw=target_input['video_grid_thw'] if 'video_grid_thw' in target_input else None,
            start_position=start_position
        )
        target_input['mrope_position_deltas'] = mrope_position_deltas[0]
        start_position += (target_input['input_ids'].shape[1] + target_input['mrope_position_deltas'])
        target_inputs.append(target_input)
    return target_inputs

def get_decode_position_ids(position):
    position_ids = torch.tensor([position]).view(1, -1).expand(3, 1, -1)
    return position_ids

def deal_prefill_output(output, decode_input: ModelInput, request: Request):
    decode_input.input_data.data['mrope_position_deltas'] = sum([model_input.input_data.data['mrope_position_deltas'] for model_input in request.inputs])
    decode_input.input_data.past_key_values = merge_kv_caches_simple(output["kv_cache"], decode_input.target_device)
    decode_input.input_data.data['position_ids'] = get_decode_position_ids(
        decode_input.input_data.computed_tokens + decode_input.input_data.data['mrope_position_deltas']
        ).to(decode_input.target_device)

def deal_decode_output(output, request: Request):
    request.inputs[0].input_data.past_key_values = output["kv_cache"][0]
    request.inputs[0].input_data.data['position_ids'] = get_decode_position_ids(
        request.inputs[0].input_data.computed_tokens + request.inputs[0].input_data.data['mrope_position_deltas']
        ).to(request.decode_device)
