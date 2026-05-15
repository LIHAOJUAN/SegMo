from PIL import Image
import numpy as np
import torch

import vision_parallel.models.MiniCPM.MiniCPMO as MiniCPMO

def get_image_bound(input_ids):
    start_cond = (input_ids == MiniCPMO.tokenizer.im_start_id) | (input_ids == MiniCPMO.tokenizer.slice_start_id)
    end_cond = (input_ids == MiniCPMO.tokenizer.im_end_id) | (input_ids == MiniCPMO.tokenizer.slice_end_id)

    image_start_idx = torch.where(start_cond)[0]
    image_start_idx += 1
    image_end_idx = torch.where(end_cond)[0]

    valid_image_nums = max(len(image_start_idx), len(image_end_idx))

    image_bounds = [
        (int(image_start_idx[i]), int(image_end_idx[i]))
        for i in range(valid_image_nums)
    ]

    return image_bounds

def analyze_video_frame_tokens(target_content, sys_msg, tokenizer, processor, image_slice_num):
    """
    Analyze the token positions corresponding to each frame in video content.
    
    Args:
        target_content: List containing video content ["<unit>", image, audio, ...]
        sys_msg: System message.
        tokenizer: Tokenizer.
        processor: Processor.
    
    Returns:
        dict: Dictionary containing token position information for each frame.
    """
    # Build the messages.
    target_msg = {"role":"user", "content": target_content}
    full_msg = [sys_msg, target_msg]
    
    # Process the messages and get detailed information.
    prompts_lists, input_images_list, input_audios_list, audio_parts_list = MiniCPMO.deal_msg(
        full_msg, omni_input=True, use_tts_template=True
    )
    
    # Process the inputs.
    processed_input = processor(
        prompts_lists,
        input_images_list,
        input_audios_list,
        audio_parts_list,
        return_tensors="pt",
        max_slice_nums = image_slice_num,
    )
    
    input_ids = processed_input["input_ids"][0]  # Remove the batch dimension.
    
    # Analyze the token sequence.
    frame_token_map = {}
    current_pos = 0
    
    # First, determine how many tokens the system message occupies.
    sys_prompts, _, _, _ = MiniCPMO.deal_msg([sys_msg], omni_input=True, use_tts_template=True)
    sys_processed = processor(sys_prompts, None, None, None, return_tensors="pt")
    sys_token_count = sys_processed["input_ids"].shape[1]
    
    current_pos = sys_token_count
    
    # Compute the token count for an empty message structure for later calculations.
    empty_msg = {"role":"user", "content": []}
    empty_full_msg = [{"role": "system", "content": ""}, empty_msg]
    empty_prompts, _, _, _ = MiniCPMO.deal_msg(empty_full_msg, omni_input=True, use_tts_template=True)
    empty_processed = processor(empty_prompts, None, None, None, return_tensors="pt")
    empty_msg_token_count = empty_processed["input_ids"].shape[1]
    
    # Analyze the tokens corresponding to each content element.
    frame_count = 0
    i = 0
    
    for j in range(min(20, len(input_ids))):
        token_text = tokenizer.decode([input_ids[j]])
    
    while i < len(target_content):
        if target_content[i] == "<unit>":
            # This marks the start of a new frame.
            frame_start_pos = current_pos
            
            # Analyze the tokens for the <unit> marker.
            unit_token_msg = {"role":"user", "content": ["<unit>"]}
            unit_full_msg = [{"role": "system", "content": ""}, unit_token_msg]  # Empty system message.
            unit_prompts, _, _, _ = MiniCPMO.deal_msg(unit_full_msg, omni_input=True, use_tts_template=True)
            unit_processed = processor(unit_prompts, None, None, None, return_tensors="pt")
            
            unit_token_count = unit_processed["input_ids"].shape[1] - empty_msg_token_count
            
            current_pos += unit_token_count
            
            # Analyze the image part.
            if i + 1 < len(target_content):
                image = target_content[i + 1]
                if isinstance(image, Image.Image):
                    # Analyze image tokens.
                    image_msg = {"role":"user", "content": [image]}
                    image_full_msg = [{"role": "system", "content": ""}, image_msg]
                    image_prompts, image_list, _, _ = MiniCPMO.deal_msg(image_full_msg, omni_input=True, use_tts_template=True)
                    image_processed = processor(
                        image_prompts, image_list, None, None, return_tensors="pt", max_slice_nums = image_slice_num
                    )
                    image_token_count = image_processed["input_ids"].shape[1] - empty_msg_token_count  # Subtract empty-structure tokens.
                    
                    image_start = current_pos
                    current_pos += image_token_count
            
            # Analyze the audio part.
            if i + 2 < len(target_content):
                audio = target_content[i + 2]
                if isinstance(audio, np.ndarray):
                    # Analyze audio tokens.
                    audio_msg = {"role":"user", "content": [audio]}
                    audio_full_msg = [{"role": "system", "content": ""}, audio_msg]
                    audio_prompts, _, audio_list, audio_parts = MiniCPMO.deal_msg(audio_full_msg, omni_input=True, use_tts_template=True)
                    audio_processed = processor(audio_prompts, None, audio_list, audio_parts, return_tensors="pt")
                    audio_token_count = audio_processed["input_ids"].shape[1] - empty_msg_token_count  # Subtract empty-structure tokens.
                    
                    audio_start = current_pos
                    current_pos += audio_token_count
            
            # Record the complete information for this frame.
            frame_end_pos = current_pos - 1
            frame_token_map[frame_count] = {
                'start': frame_start_pos,
                'end': frame_end_pos,
                'total_tokens': frame_end_pos - frame_start_pos + 1,
                'unit_tokens': unit_token_count if 'unit_token_count' in locals() else 0,
                'image_tokens': image_token_count if 'image_token_count' in locals() else 0,
                'audio_tokens': audio_token_count if 'audio_token_count' in locals() else 0,
            }
            
            frame_count += 1
            i += 3  # Skip <unit>, image, audio.
        else:
            i += 1
    
    # Analyze the question part.
    if len(target_content) > 0 and isinstance(target_content[-1], str) and target_content[-1] != "<unit>":
        question = target_content[-1]
        question_msg = {"role":"user", "content": [question]}
        question_full_msg = [{"role": "system", "content": ""}, question_msg]
        question_prompts, _, _, _ = MiniCPMO.deal_msg(question_full_msg, omni_input=True, use_tts_template=True)
        question_processed = processor(question_prompts, None, None, None, return_tensors="pt")
        
        # Reuse the previously computed token count for the empty message structure.
        question_token_count = question_processed["input_ids"].shape[1] - empty_msg_token_count
        
        question_start = current_pos
        question_end = current_pos + question_token_count - 1
        
        frame_token_map['question'] = {
            'start': question_start,
            'end': question_end,
            'total_tokens': question_token_count
        }
    
    return frame_token_map, input_ids
