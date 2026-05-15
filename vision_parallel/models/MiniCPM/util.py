import math, os, torch
import numpy as np
from PIL import Image
from moviepy import VideoFileClip
from copy import deepcopy

from vision_parallel.models.MiniCPM import MiniCPMO
from vision_parallel.request import InputData, ModelInput, Request
from vision_parallel.kv_cache_manager import merge_kv_caches_simple

default_tts_chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>' }}{% endif %}"

def construct_frames_from_video(video_path, flatten=True, save_frames=False, save_path=None):
    video = VideoFileClip(video_path)
    num_units = math.ceil(video.duration)
    
    contents= []
    for i in range(0, num_units, 1):
        frame = video.get_frame(i)
        image = Image.fromarray((frame).astype(np.uint8))
        if flatten:
            contents.extend(["<unit>", image])
        else:
            contents.append(["<unit>", image])
        if save_frames:
            video_dir = f"{save_path}/"
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            image.save(f"{video_dir}/frame_{i}.png")
    
    return contents, 2

def deal_msg(msgs_list, omni_input=False, use_tts_template=False):
    if isinstance(msgs_list[0], list):
        batched = True
    else:
        batched = False
    if batched is False:
        msgs_list = [msgs_list]

    prompts_lists = []
    input_images_list = []
    input_audios_list = []
    audio_parts_list = []
    for msgs in msgs_list:
        copy_msgs = deepcopy(msgs)

        images = []
        audios = []
        audio_parts = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["system", "user", "assistant"]
            if i == 0:
                assert role in ["user", "system"], "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, np.ndarray):
                    audios.append(c)
                    audio_parts.append(i)
                    cur_msgs.append("(<audio>./</audio>)")
                    use_tts_template = True
                elif isinstance(c, str):
                    cur_msgs.append(c)
            if omni_input:
                msg["content"] = "".join(cur_msgs)
            else:
                msg["content"] = "\n".join(cur_msgs)

        prompts_lists.append(
            MiniCPMO.processor.tokenizer.apply_chat_template(
                copy_msgs,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=default_tts_chat_template if use_tts_template else None,
            )
        )
        input_images_list.append(images)
        input_audios_list.append(audios)
        audio_parts_list.append(audio_parts)
    return prompts_lists, input_images_list, input_audios_list, audio_parts_list

def get_target_input(question, video_content, zero_frames, index, 
                     video_slice_num, image_slice_num, calculate_attention):
    target_content = []
    target_content.append(f"There are {video_slice_num} video segments in total. This is the content of the {index}th video segment:")
    target_content.extend(video_content)
    target_content.append(question)
    target_msg = {"role":"user", "content": target_content}
    global_content = []
    global_content.append("This is a global information summary of the video:")
    for zero_frame in zero_frames:
        global_content.extend(["<unit>", zero_frame])
    global_msg = {"role":"user", "content": global_content}
    final_msg = [MiniCPMO.sys_msg, global_msg, target_msg]
    # final_msg = [MiniCPMO.sys_msg, target_msg]
    prompts_lists, input_images_list, input_audios_list, audio_parts_list = deal_msg(
        final_msg, omni_input=True, use_tts_template=True
    )
    target_input = MiniCPMO.processor(
            prompts_lists,
            input_images_list,
            input_audios_list,
            audio_parts_list,
            return_tensors="pt",
            max_slice_nums = image_slice_num,
            use_image_id=False,
        )

    return target_input

def get_target_inputs(prompt_header, video_contents, zero_frames, 
                      image_slice_num, calculate_attention = False):
    target_inputs = []
    
    start_position = 0
    for index, video_content in enumerate(video_contents):
        target_input = get_target_input(prompt_header, video_content, zero_frames, index+1, len(video_contents), image_slice_num,
                                        calculate_attention)
        target_input["position_ids"] = torch.arange(
            start_position, start_position+target_input["input_ids"].shape[1], dtype=torch.long).unsqueeze(0)
        target_inputs.append(target_input)
        start_position += target_input["input_ids"].shape[1]
    return target_inputs

def deal_prefill_output(output, decode_input: ModelInput, request: Request):
    decode_input.input_data.past_key_values = merge_kv_caches_simple(output["kv_cache"], decode_input.target_device)
    decode_input.input_data.data['position_ids'] = torch.tensor([decode_input.input_data.computed_tokens]).unsqueeze(0).to(decode_input.target_device)

def deal_decode_output(output, request: Request):
    request.inputs[0].input_data.past_key_values = output["kv_cache"][0]
    request.inputs[0].input_data.data['position_ids'] = torch.tensor([request.inputs[0].input_data.computed_tokens]).unsqueeze(0).to(request.decode_device)