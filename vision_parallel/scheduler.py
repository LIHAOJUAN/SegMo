from abc import ABC
from typing import List, Tuple, Dict
from collections import deque
import math
import numpy as np
import threading
import torch
import time
import logging
import clip

from vision_parallel.config import Config
from vision_parallel.request import Request
from vision_parallel.request import ModelInput, InputData
from vision_parallel.parallel_process import SimplePipeWorker
from util.performance_monitor import get_monitor
from vision_parallel.video_util import get_duration_from_cv2

import vision_parallel.models.MiniCPM.util as MiniCPM_util
import vision_parallel.models.QwenVL.util as QwenVL_util

# Get the logger instance; it will automatically use the configuration from the main script.
logger = logging.getLogger("vision-parallel")

class WorkerStatus:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.GPU_memory_budget = 0  # The original available memory of this GPU.
        self.current_memory_usage = 0  # Currently used memory.
        self.available_memory = 0  # Currently available memory (GPU_memory_budget - current_memory_usage).

class Scheduler(ABC):
    def __init__(self, config: Config = None, workers: List = None):
        self.config = config
        self.workers = workers
        self.static_sample_threshold = config.test.static_sample_threshold
        if config:
            self.worker_num = config.test.worker_num
            self.video_slice_num = config.test.video_slice_num
            self.max_frames_limit = config.test.max_frames_per_video
            self.scene_detection_parallel_slice = config.test.scene_detection_parallel_slice
        else:
            self.worker_num = 1
            self.video_slice_num = 1
            self.max_frames_limit = 64
            self.scene_detection_parallel_slice = 4 
            
        self.waiting_lock = threading.Lock()
        self.waiting: deque[Request] = deque() # Accessed by the run thread and one or more generate threads.
        self.running: list[Request] = [] # Accessed only by the run thread.
        self.log_stats = False
        self.worker_status = [WorkerStatus(i) for i in range(self.worker_num)]
        self.model_config_cache: dict = None
        self.update_worker_status()
        self.all_gpu_memory_budget = sum([worker.GPU_memory_budget for worker in self.worker_status])
        
        self.pipe_worker = SimplePipeWorker(config)
        
        self.monitor = get_monitor()

        self.clip_model, self.clip_preprocess = clip.load(config.model.CLIP_model_path, device="cuda")

        self.scene_difference_weight = config.test.scene_difference_weight
        self.scene_relevance_weight = config.test.scene_relevance_weight

        if config.model.model_name == "MiniCPM":
            self.construct_frames_from_video = MiniCPM_util.construct_frames_from_video
            self.get_target_inputs = MiniCPM_util.get_target_inputs
        elif config.model.model_name == "QwenVL":
            self.construct_frames_from_video = QwenVL_util.encode_video
            self.get_target_inputs = QwenVL_util.get_target_inputs

    def add_requests(self, requests: list[Request]) -> None:
        if not requests:
            return
        with self.waiting_lock:
            self.waiting.extend(requests)

    def schedule(self, prev_requests: list[Request]) -> list[Request]:
        logger.debug(f"Schedule 开始: {len(self.waiting)} 等待中, {len(self.running)} 运行中。")
        # Return a list of requests to process based on the waiting and running queues.
        # The waiting queue uses popleft().

        # 1. Predict each worker's future available memory while prev_requests are running.
        predicted_available_memory = {}

        # 1.1 Calculate prev_requests' memory usage on each worker.
        prev_requests_usage = {i: 0 for i in range(self.worker_num)}
        if prev_requests:
            for req in prev_requests:
                estimated_usage = self._estimate_memory_usage(req)
                for worker_id, usage in estimated_usage.items():
                    prev_requests_usage[worker_id] += usage

        # 1.2 Calculate each worker's predicted available memory.
        for worker in self.worker_status:
            worker_id = worker.worker_id
            baseline_memory_used = worker.current_memory_usage
            newly_used_memory = prev_requests_usage[worker_id]
            predicted_available_memory[worker_id] = worker.GPU_memory_budget - (baseline_memory_used + newly_used_memory)

        scheduled_requests = []
        can_schedule = True
        # 2. Schedule decode requests.
        idx = 0
        while can_schedule and idx < len(self.running):
            request_to_check = self.running[idx]
            
            # Estimate its memory usage.
            estimated_usage = self._estimate_memory_usage(request_to_check)

            # Check whether this request can fit.
            can_fit = True
            # for worker_id, usage in estimated_usage.items():
            #     if predicted_available_memory.get(worker_id, 0) < usage:
            #         can_fit = False
            #         break
            
            if can_fit:
                scheduled_requests.append(request_to_check)
                for worker_id, usage in estimated_usage.items():
                    predicted_available_memory[worker_id] -= usage      
            else:
                # If the head request does not fit, later requests will not fit either, so stop scheduling.
                can_schedule = False
            idx += 1

        # 3. Schedule new prefill requests.
        while can_schedule and self.waiting:
            request_to_check = self.waiting[0]

            # Prepare inputs for this request and estimate its memory usage.
            try:
                pre_process_start_time = time.perf_counter()
                self._pre_inference_processing(request_to_check, predicted_available_memory)
                pre_process_end_time = time.perf_counter()
                self.monitor.record("pre_process_latency", pre_process_end_time-pre_process_start_time)
                
            except Exception as e:
                self.waiting.popleft()
                request_to_check.answer = str(e)
                self.finish_request(request_to_check)
                continue

            estimated_usage = self._estimate_memory_usage(request_to_check)

            # Check whether this request can fit.
            can_fit = True
            # for worker_id, usage in estimated_usage.items():
            #     if predicted_available_memory.get(worker_id, 0) < usage:
            #         can_fit = False
            #         break
            
            if can_fit:
                with self.waiting_lock:
                    self.waiting.popleft()
                    self.running.append(request_to_check)
                scheduled_requests.append(request_to_check)

                for worker_id, usage in estimated_usage.items():
                    predicted_available_memory[worker_id] -= usage      
            else:
                # If the head request does not fit, later requests will not fit either, so stop scheduling.
                request_to_check.inputs = []
                request_to_check.status = 'pending'
                can_schedule = False

        logger.debug(f"Schedule 结束: 本轮调度了 {len(scheduled_requests)} 个请求。")
        return scheduled_requests
    
    def _pre_inference_processing(self, request: Request, available_memory_map: Dict[int,int]) -> None:
        logger.debug(f"_pre_inference_processing预处理开始。")
        # 1. Call Scheduler.construct_frames_from_video or Scheduler.construct_frames_from_images.
        #    Read the video data and obtain video_contents and scene_list.
        # 2. Call slice_frames to split the video data and obtain slice_video_contents and zero_frames.
        # 3. Call get_target_inputs to obtain model_inputs, then build ModelInput objects from the slices and put them into Request.inputs.
        multi_modal_data = request.data["multi_modal_data"]
        data_type = multi_modal_data["type"]
        data_path = multi_modal_data["path"]
        prompt_header = request.data.get("prompt_header", "")
        video_sclice_num = self.video_slice_num
        image_slice_num = self.config.test.image_slice_num

        try:
            slice_video_contents, zero_frames = self.slice_frames(data_type, data_path,available_memory_map, prompt_header)
        except Exception as e:
            raise e
        
        logger.debug(f"slice_video_contents, zero_frames获取完成。")
        logger.debug(f"开始get_target_inputs。")
        get_target_inputs_start_time = time.perf_counter()
        video_inputs = self.get_target_inputs(prompt_header, slice_video_contents, zero_frames, image_slice_num)
        get_target_inputs_end_time = time.perf_counter()
        self.monitor.record("get_target_inputs_latency", get_target_inputs_end_time-get_target_inputs_start_time)
        request.inputs = []
        for idx, video_input in enumerate(video_inputs):
            model_input = ModelInput()
            model_input.slice_index = idx
            model_input.target_device = idx % video_sclice_num
            input_data = InputData(video_input)
            input_data.all_input_ids = video_input['input_ids']
            input_data.all_position_ids = video_input['position_ids']
            input_data.computed_tokens = 0
            model_input.input_data = input_data
            self.monitor.record("seq_len", video_input['input_ids'].shape[1])
            request.inputs.append(model_input)
        logger.debug(f"get_target_inputs完成。")
        logger.debug(f"_pre_inference_processing预处理完成。")

    def _estimate_memory_usage(self, request: Request) -> Dict[int, int]:
        # Estimate request memory usage on each worker.
        memory_map = {i: 0 for i in range(self.worker_num)}
        if not request.inputs:
            return memory_map
        
        model_config = self._get_model_config()
        
        for model_input in request.inputs:
            if model_input.input_data is None or model_input.input_data.all_input_ids is None:
                continue

            if request.status == 'pending':
                new_seq_len = model_input.input_data.all_input_ids.shape[1]
            else: # 'prefill' or 'decode' (meaning the next round will run decode)
                new_seq_len = 1

            kv_cache_size = (2 * model_config['num_layers'] * new_seq_len * model_config['hidden_size'] * model_config['bytes_per_element'])
            activation_size = (new_seq_len * model_config['hidden_size'] * model_config['num_layers'] *
                               model_config['activation_coefficient'] * model_config['bytes_per_element'])

            if request.status == 'pending':
                total_estimated_size = kv_cache_size + activation_size
            else: # 'prefill' or 'decode'
                total_estimated_size = kv_cache_size

            memory_map[model_input.target_device] += int(total_estimated_size)
        return memory_map
    
    def _get_model_config(self)-> Dict:
        if self.model_config_cache is None:
            if not self.workers:
                raise ValueError("No workers available to get model config.")
            worker_model = self.workers[0].model
            self.model_config_cache = {
                "num_layers": worker_model.config.num_hidden_layers,
                "hidden_size": worker_model.config.hidden_size,
                "bytes_per_element": 2, # for bfloat16
                "activation_coefficient": 16 # Tunable parameter; adjust based on experiments.
            }
            
        return self.model_config_cache
    
    def update_status(self, requests: list[Request]) -> None:
        # Update request state and worker state.
        self.update_requests(requests)
        self.update_worker_status()

    def update_requests(self, requests: list[Request]) -> None:
        # Update request state:
        # 1. prefill requests: move them into the running queue.
        # 2. decode requests: remove them from the running queue once completed.

        # Collection of completed requests.
        complete_requests = [req for req in requests if req.status == 'completed']
        
        for req in complete_requests:
            if req in self.running:
                self.running.remove(req)
            self.finish_request(req)

    def update_worker_status(self) -> None:    
        # Update worker state through CUDA APIs.
        for worker in self.worker_status:
            device_id = worker.worker_id
            
            if worker.GPU_memory_budget == 0:
                # Get total memory for the current device.
                worker.GPU_memory_budget = torch.cuda.get_device_properties(device_id).total_memory

            # Get current memory usage for the device.
            current_memory_usage = torch.cuda.memory_allocated(device_id)
            
            worker.current_memory_usage = current_memory_usage
            worker.available_memory = worker.GPU_memory_budget - current_memory_usage
            
            # Synchronize state via the CUDA API.
            torch.cuda.synchronize(device_id)

    def finish_request(self, request: Request) -> None:
        # The request is complete and its result has been returned, so remove it from requests.
        # if request.request_id in self.requests:
        request.future.set_result(request.answer)

    def slice_frames(self, data_type, data_path, available_memory_map: Dict[int, int], prompt_header):
        if data_type == "video":
            try:
                slice_video_contents, zero_frames = self.get_sliced_frames_and_zero_frames_from_video(data_path, available_memory_map, prompt_header)
            except Exception as e:
                raise e
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        return slice_video_contents, zero_frames
    
    def get_sliced_frames_and_zero_frames_from_video(self, video_path, available_memory_map: Dict[int, int], prompt_header):
        logger.debug(f"开始get_sliced_frames_and_zero_frames_from_video，为视频 '{video_path}' 提取帧...")
        video_duration = get_duration_from_cv2(video_path)
        if video_duration >= self.static_sample_threshold:
            get_scene_list_future = self.pipe_worker.submit_task(
                video_path, 
                self.scene_detection_parallel_slice
            )
        construct_frames_from_video_start_time = time.perf_counter()
        video_contents, frame_unit_size = self.construct_frames_from_video(video_path)
        construct_frames_from_video_end_time = time.perf_counter()
        self.monitor.record("construct_frames_from_video_latency", construct_frames_from_video_end_time-construct_frames_from_video_start_time)
        slice_frames_by_scene_with_GPU_memory_start_time = time.perf_counter()
        if video_duration >= self.static_sample_threshold:
            try:
                scene_list, detect_scene_latency = get_scene_list_future.result()
                if not scene_list:
                    logger.warning("no scene detected, rollback to uniform sampling")
                    slice_video_contents, zero_frames = self.select_frames_equally(video_contents, frame_unit_size)
                else:
                    self.monitor.record("detect_scene_latency", detect_scene_latency)
                    wait_scene_list_end_time = time.perf_counter()
                    self.monitor.record("wait_scene_list_latency", wait_scene_list_end_time-slice_frames_by_scene_with_GPU_memory_start_time)
                    slice_video_contents, zero_frames = self.slice_frames_by_scene_with_GPU_memory(video_contents, scene_list, video_path, available_memory_map, frame_unit_size, prompt_header)
            except Exception as e:
                raise e
        else:
            slice_video_contents, zero_frames = self.select_frames_equally(video_contents, frame_unit_size)

        slice_frames_by_scene_with_GPU_memory_end_time = time.perf_counter()
        self.monitor.record("slice_frames_by_scene_with_GPU_memory_latency", slice_frames_by_scene_with_GPU_memory_end_time-slice_frames_by_scene_with_GPU_memory_start_time)
        
        # record frame num
        frame_num = 0
        for slice_video_content in slice_video_contents:
            frame_num += (len(slice_video_content)/frame_unit_size)
        self.monitor.record("frames_num", frame_num)
        
        return slice_video_contents, zero_frames
    
    def select_frames_equally(self, frames, frame_unit_size=2):
        
        def get_seq_frames(total_num_frames, desired_num_frames):
            """
            Calculate the indices of frames to extract from a video.

            Parameters:
            total_num_frames (int): Total number of frames in the video.
            desired_num_frames (int): Desired number of frames to extract.

            Returns:
            list: List of indices of frames to extract.
            """
            # Calculate the size of each segment from which a frame will be extracted
            seg_size = float(total_num_frames - 1) / desired_num_frames
            seq = []
            for i in range(desired_num_frames):
                # Calculate the start and end indices of each segment
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                # Append the middle index of the segment to the list
                seq.append((start + end) // 2)
            return seq
        
        # Calculate the total number of units.
        total_units = len(frames) // frame_unit_size
        selected_frames = []
        
        if total_units <= self.max_frames_limit:
            # If the number of units is below self.max_frames_limit, return all units.
            selected_frames = frames
        else:
            seqs = get_seq_frames(total_units, self.max_frames_limit)
            selected_frames = []
            for frame_idx in seqs:
                start_idx = frame_idx * frame_unit_size
                end_idx = start_idx + frame_unit_size
                frame_data = frames[start_idx:end_idx]
                selected_frames.extend(frame_data)
        
        total_units = len(selected_frames) // frame_unit_size
        
        units_per_slice = total_units // self.video_slice_num
        sliced_frames = []
        
        for i in range(self.video_slice_num):
            start_unit = i * units_per_slice
            end_unit = min(start_unit + units_per_slice, total_units)
            
            # Convert units into frame indices.
            start_frame_idx = start_unit * frame_unit_size
            end_frame_idx = end_unit * frame_unit_size
            
            slice_frames = selected_frames[start_frame_idx:end_frame_idx]
            sliced_frames.append(slice_frames)
        
        zero_frames = self.get_zero_frame(sliced_frames, frame_unit_size)
        
        return sliced_frames, zero_frames
    
    def compare_two_images(self, img1, img2):
        # Return the difference score between two frames.
        """
        Calculate the difference between two image frames.
        Return a float between 0 and 1, where larger values indicate greater difference.
        """
        # Convert to grayscale to simplify the calculation.
        img1_gray = img1.convert('L')
        img2_gray = img2.convert('L')

        # Convert to NumPy arrays.
        np_img1 = np.array(img1_gray, dtype=float)
        np_img2 = np.array(img2_gray, dtype=float)

        # Ensure the image sizes match.
        if np_img1.shape != np_img2.shape:
            # If sizes differ, resize them or directly return the maximum difference value.
            return 1.0

        # Compute mean absolute difference and normalize it to the [0, 1] range.
        diff = np.mean(np.abs(np_img1 - np_img2))
        return diff / 255.0

    def select_frame_from_scene(self, scene_frames: list, frame_unit_size=3):
        # Choose a representative sampling stride from multiple frames in a scene.
        if not scene_frames or len(scene_frames) < frame_unit_size * 2:
            return 1
        
        # Extract the first and last frame.
        first_frame = scene_frames[1]
        last_frame = scene_frames[len(scene_frames)-(frame_unit_size-1)]
        difference = self.compare_two_images(first_frame, last_frame)

        # Decide the sampling stride based on the magnitude of the difference.
        if difference < 0.05:   
            return 4  # Small intra-scene variation, so use a larger stride.
        elif difference < 0.2:
            return 2  # Moderate intra-scene variation, so use a medium stride.
        else:
            return 1  # Large intra-scene variation, so use a smaller stride.

    def slice_frames_by_scene_with_GPU_memory(self, frames, scene_list, video_path, available_memory_map: Dict[int, int], frame_unit_size=2, prompt_header=""):
        logger.debug(f"开始slice_frames_by_scene_with_GPU_memory,为视频 '{video_path}' 切分帧...")

        # Remove scenes whose start and end fall within the same second.
        filtered_scenes = []
        for scene in scene_list:
            start_s = scene[0].get_seconds()
            end_s = scene[1].get_seconds()
            # If start and end fall within the same second (same ceil value), treat it as invalid and skip it.
            if math.ceil(start_s) == math.ceil(end_s):
                continue
            filtered_scenes.append(scene)
        scene_list = filtered_scenes

        # Compute a relevance score between each scene and the question.
        representative_frames = [frames[math.ceil(scene[0].get_seconds()) * frame_unit_size + (frame_unit_size-1)] for scene in scene_list]
        start_idx = prompt_header.find("Question")
        # end_idx = prompt_header.find("Options")
        if start_idx != -1:
            question = prompt_header[start_idx: ]
        else:
            question = prompt_header
        scene_scores = self.get_frame_score(representative_frames, question)
        # all_frame_scores = self.get_frame_score(frames, question)

        scene_num = len(scene_list)
        self.monitor.record("scene_num", scene_num)
        # Precompute the maximum valid frame index for boundary checks. ---
        max_frame_index = (len(frames) // frame_unit_size) - 1

        # ==================== Entry point for the new decision logic ====================
        if scene_num > self.max_frames_limit:
            # Strategy 1: when the number of scenes exceeds the frame limit, select N scenes and give each one frame.
            logger.warning(f"Warning: Scene count ({scene_num}) exceeds frame limit ({self.max_frames_limit}). Switching to scene selection strategy.")
            # Select the top max_frames_limit scenes by scene_scores while preserving temporal order.
            top_k = min(self.max_frames_limit, len(scene_scores))
            top_indices = np.argsort(-np.array(scene_scores))[:top_k]
            top_indices_sorted = sorted(top_indices.tolist())
            scene_list = [scene_list[i] for i in top_indices_sorted]
            allocated_frames = [1] * len(scene_list)
        elif max_frame_index < self.max_frames_limit:
            # Strategy 2: when the video frame count is below the frame limit, select all frames.
            # Allocate self.max_frames_limit to every scene so each scene can keep as many frames as possible.
            allocated_frames = [self.max_frames_limit] * len(scene_list)
        else:
            # Strategy 3: when the number of scenes is within the frame limit, use the original dynamic allocation and scaling logic.
            frames_budget = max(3 * scene_num, self.max_frames_limit)
            scene_differences = []
            for i in range(scene_num):
                start_frame_idx = math.ceil(scene_list[i][0].get_seconds())
                end_frame_idx = math.floor(scene_list[i][1].get_seconds())
            
                # Critical fix 2: restore and strengthen boundary checks here as well.
                if start_frame_idx > max_frame_index or start_frame_idx >= end_frame_idx:
                    scene_differences.append(0) # Add an invalid difference value to keep the list length consistent.
                    continue

                safe_end_idx = min(end_frame_idx, max_frame_index)
                try:
                    # Compare the first and last frame of each video segment and record the difference in scene_differences.
                    first_frame = frames[start_frame_idx * frame_unit_size + (frame_unit_size-1)]
                    last_frame = frames[safe_end_idx * frame_unit_size + (frame_unit_size-1)]
                    diff = self.compare_two_images(first_frame, last_frame)
                    scene_differences.append(diff)
                except IndexError:
                    logger.warning(f"Scene {i} in video {video_path} caused 'list index out of range'. Skipping.")
                    continue

            # Scene difference weight.
            all_differences_sum = sum(scene_differences) + 1e-5 # Avoid division-by-zero errors.
            diff_weights = [diff / all_differences_sum for diff in scene_differences]

            # Scene-question relevance weight.
            scores = np.array(scene_scores, dtype=float)
            shifted = scores - scores.min()
            score_weights = shifted / shifted.sum() if shifted.sum() > 0 else np.full_like(shifted, 1.0/shifted.size, dtype=float)

            # Combined weight.
            combined_weights = [
                dw*self.scene_difference_weight + sw*self.scene_relevance_weight
                for dw, sw in zip(diff_weights, score_weights)]

            allocated_frames = [max(int(round(w * frames_budget)), 1) for w in combined_weights]
            total_allocated_frames = sum(allocated_frames)
            if total_allocated_frames > self.max_frames_limit:
                logger.warning(f"Video {video_path} initial frame count ({total_allocated_frames}) exceeds limit ({self.max_frames_limit}). Scaling down.")
                scale_factor = self.max_frames_limit / total_allocated_frames
                allocated_frames = [max(1, int(round(count * scale_factor))) for count in allocated_frames]

        # =================================================================

        # 1. Compute each scene's "cost" (the final frame count after sparse sampling), but do not sample yet.
        scene_info = [] # (original frames, step size, cost)
        total_cost = 0  # Total frame count of the video.

        for i, allocated_frame in enumerate(allocated_frames):
            # Here we assume frames are sampled at 1 fps, so math.ceil(scene_tuple[1].get_seconds()) can be used directly.
            # Map scene timestamps to frame indices.
            
            # Critical fix: clamp the scene end time within the video's maximum frame index.
            scene_start_index = math.ceil(scene_list[i][0].get_seconds())
            scene_end_index = math.ceil(scene_list[i][1].get_seconds())
            end_frame = min(scene_end_index, max_frame_index)
            
            length = end_frame - scene_start_index
            allocated_frame = min(allocated_frame, max(1, length)) # cost cannot exceed the original frame count.
            if allocated_frame == 0:
                continue
            scene_info.append({
                'start': scene_start_index,
                'end':  end_frame,
                'step': max(1, (float)((length+1) / allocated_frame)),
                'cost': allocated_frame,  # cost cannot exceed the original frame count.
            })
            total_cost += allocated_frame

        # 2. Find the ideal cost cut points for each worker.
        total_memory = sum(available_memory_map.values())
        if total_memory == 0:
            return self.select_frames_equally(frames, frame_unit_size)
        
        ideal_cost_cutoffs = []
        cumulative_cost = 0
        for i in range(self.worker_num):
            worker_id = self.worker_status[i].worker_id
            proportion = available_memory_map.get(worker_id, 0) / total_memory
            cumulative_cost += proportion * total_cost
            ideal_cost_cutoffs.append(cumulative_cost)

        # 3. Find the scene boundaries closest to the ideal cut points.
        worker_scene_assignments = [[] for _ in range(self.worker_num)]
        current_cost = 0
        worker_idx = 0
        for i, info in enumerate(scene_info):
            # If assigning the next scene would cross the current worker's cut point,
            # decide whether it is better to assign it to the current worker or the next worker.
            if worker_idx < len(ideal_cost_cutoffs) - 1:
                cost_if_assign_current = abs(current_cost + info['cost'] - ideal_cost_cutoffs[worker_idx])
                cost_if_assign_next = abs(current_cost - ideal_cost_cutoffs[worker_idx])
                if cost_if_assign_next < cost_if_assign_current:
                    worker_idx += 1

            worker_scene_assignments[worker_idx].append(info)
            current_cost += info['cost']
        
        # 4. Actually start slicing and sparse sampling.
        logger.debug(f"开始真正切片,为视频 '{video_path}' 切分帧和稀疏采样...")
        slice_video_contents = []
        
        # for debug
        assigned_frame_idx_lists = []
        
        for assignment in worker_scene_assignments:
            worker_frames = []
            assigned_frame_idx_list = []
            for info in assignment:
                step = info['step']
                start = info['start']
                end = info['end']
                cur: float = start
                while(cur < end):
                    frame_idx = math.floor(cur) * frame_unit_size
                    if frame_idx + frame_unit_size <= len(frames):
                        worker_frames.extend(frames[frame_idx: frame_idx + frame_unit_size])
                    assigned_frame_idx_list.append(frame_idx/frame_unit_size)
                    cur += step
            slice_video_contents.append(worker_frames)
            assigned_frame_idx_lists.append(assigned_frame_idx_list)

        # zero_frames = self.get_topk_zero_frame(representative_frames, scene_scores, max(math.log2(self.max_frames_limit), 1))
        zero_frames = self.get_topk_zero_frame(representative_frames, scene_scores, 2)
        # zero_frames = self.get_zero_frame(slice_video_contents, frame_unit_size)

        return slice_video_contents, zero_frames

    def get_frame_score(self, images, question):
        start_time = time.perf_counter()
        question = question[:77]
        text = clip.tokenize([question]).to("cuda")
        images = [self.clip_preprocess(image) for image in images]
        image_batch = torch.stack(images).to("cuda")  # [batch, 3, H, W]
        with torch.no_grad():
            logits_per_image, _ = self.clip_model(image_batch, text)  # logits_per_image: [batch, 1]
            scores = logits_per_image.squeeze(1).cpu().numpy()  # [batch]
        end_time = time.perf_counter()
        self.monitor.record("CLIP_latency", end_time-start_time)
        return scores

    @staticmethod
    def get_topk_zero_frame(representative_frames, scene_scores, topk):
        if len(representative_frames) != len(scene_scores):
            raise ValueError("帧数组与得分数组长度必须一致")

        topk = min(topk, len(representative_frames))
        if topk <= 0:
            return []
        
        combined = list(zip(scene_scores, representative_frames))
        combined.sort(reverse=True, key=lambda x: x[0])  # Sort by score in descending order.
        
        # Extract the top-k frames.
        topk_frames = [frame for (score, frame) in combined[:int(topk)]]
        
        return topk_frames

    @staticmethod
    def get_zero_frame(slice_video_contents, frame_unit_size=2):
        if not slice_video_contents:
            return None
        zero_frames = []
        for content in slice_video_contents:
            zero_frame = content[frame_unit_size-1]
            zero_frames.append(zero_frame)
        return zero_frames
    
    def shutdown(self):
        if getattr(self, "pipe_worker", None) is not None:
            self.pipe_worker.shutdown()
