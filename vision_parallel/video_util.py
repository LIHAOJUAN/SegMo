import torch
import vision_parallel.models.MiniCPM.MiniCPMO as MiniCPMO
import os, math, time
from PIL import Image
from moviepy import VideoFileClip
import logging
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict
from scenedetect import detect, AdaptiveDetector

logger = logging.getLogger("vision-parallel")

def get_next_token(logits):
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    return next_token

def get_duration_from_cv2(filename):
    cap = cv2.VideoCapture(filename)
    if cap.isOpened():
        rate = cap.get(cv2.CAP_PROP_FPS)
        frame_num =cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_num/rate
        cap.release() # Releasing resources promptly is a good habit.
        return duration
    return -1

def get_scene_list(video_path: str, scene_detection_parallel_slice: int, adaptive_threshold: float) -> List[Tuple[int, int]]:
    """
    Use multithreading to accelerate scene detection for long videos.
    Apply overlapping splits and result merging to avoid missing boundary scenes.
    """
    start_time = time.perf_counter()
    
    # scene_list = detect(video_path, AdaptiveDetector(adaptive_threshold=adaptive_threshold))
    # end_time = time.perf_counter()
    
    # return scene_list, end_time-start_time
    
    duration = get_duration_from_cv2(video_path)
    if duration <= 0:
        logger.warning(f"无法获取视频时长: {video_path}。退回至单线程检测。")
        return detect(video_path, AdaptiveDetector())

    # --- Key change 1: define the overlap duration. ---
    # A 2-second overlap is usually enough to capture boundary scenes.
    overlap_sec = 2.0
    num_slice = scene_detection_parallel_slice
    slice_ranges = []
    slice_length = duration / num_slice

    for i in range(num_slice):
        start = i * slice_length
        # For all but the first segment, shift the start time backward by the overlap duration.
        if i > 0:
            start = max(0, start - overlap_sec)
        
        end = (i + 1) * slice_length if i < num_slice - 1 else duration
        slice_ranges.append((start, end))
    
    logger.debug(f"已将视频 '{video_path}' 切分为以下时间段 (含重叠): {slice_ranges}")

    futures = []
    with ThreadPoolExecutor(max_workers=num_slice) as executor:
        for start, end in slice_ranges:
            future = executor.submit(
                detect,
                video_path=video_path,
                detector=AdaptiveDetector(adaptive_threshold=adaptive_threshold),
                start_time=start,
                end_time=end
            )
            futures.append(future)
        
        all_scenes = []
        for i, future in enumerate(futures):
            try:
                scene_list = future.result()
                logger.debug(f"Future {i} (时间段 {slice_ranges[i]}) 已完成, 发现 {len(scene_list)} 个场景： {scene_list}")
                all_scenes.extend(scene_list)
            except Exception as e:
                logger.error(f"视频 '{video_path}' 的一个场景检测线程失败: {e}", exc_info=True)

    # --- Key change 2: merge and deduplicate detected scenes. ---
    # First, sort by start time.
    all_scenes.sort()

    if not all_scenes:
        merged_scenes = []
    else:
        merged_scenes = [all_scenes[0]]
        for next_scene in all_scenes[1:]:
            last_scene_end_time = merged_scenes[-1][1].get_seconds()
            next_scene_start_time = next_scene[0].get_seconds()

            # If the next scene starts before the end of the last merged scene, it came from overlap detection.
            if next_scene_start_time < last_scene_end_time:
                # Merge the two scenes and keep the later end time.
                merged_scenes[-1] = (merged_scenes[-1][0], max(merged_scenes[-1][1], next_scene[1]))
            elif next_scene_start_time == last_scene_end_time:
                merged_scenes.append(next_scene)
            else:
                # This means one thread detected no scene boundary, so the adjacent ranges should be merged.
                merged_scenes[-1] = (merged_scenes[-1][0], next_scene[1])
    # ---------------------------------------------
    end_time = time.perf_counter()
    return merged_scenes, end_time-start_time
