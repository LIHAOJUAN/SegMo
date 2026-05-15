from abc import ABC
from typing import Dict, List, Tuple, Union
import torch
from concurrent.futures import Future
import threading
import logging
import time
from transformers.feature_extraction_utils import BatchFeature
from transformers.cache_utils import DynamicCache
from concurrent.futures import ThreadPoolExecutor, as_completed

from vision_parallel.config import Config
from vision_parallel.scheduler import Scheduler
from vision_parallel.worker import Worker
from vision_parallel.request import Request, ModelInput, InputData
from vision_parallel.kv_cache_manager import merge_kv_caches, merge_kv_caches_simple
from vision_parallel.video_util import get_next_token
import vision_parallel.models.MiniCPM.MiniCPMO as MiniCPMO
import vision_parallel.models.MiniCPM.util as MiniCPMO_util
import vision_parallel.models.QwenVL.QwenVL as QwenVL
import vision_parallel.models.QwenVL.util as QwenVL_util
from vision_parallel.util import AtomicInteger
from util.performance_monitor import get_monitor

logger = logging.getLogger("vision-parallel")

class ParallelInferenceEngine(ABC):
    def __init__(self, config: Config):
        """
        Initialize the ParallelInferenceEngine with the given configuration.
        Args:
            config (Config): Configuration object containing model and inference settings.
        """
        logger.info("正在初始化 ParallelInferenceEngine...")
        self.config = config
        self.use_generate = config.test.use_generate
        self.device_count = config.test.video_slice_num
        
        if self.device_count > 1 and self.use_generate:
            raise ValueError("use_generate=True is not supported when video_slice_num > 1.")

        self.workers: list[Worker] = []
        for i in range(self.device_count):
            self.workers.append(Worker(config, i))

        if config.model.model_name == "MiniCPM":
            sys_message = self.workers[0].model.get_sys_prompt(mode='omni', language='en')
            MiniCPMO.initialize(config.model.model_path, sys_message)
            self.tokenizer = MiniCPMO.tokenizer
        if config.model.model_name == "QwenVL":
            QwenVL.initialize(config.model.model_path)
            self.tokenizer = QwenVL.tokenizer

        self.request_id: AtomicInteger = AtomicInteger(0)

        self.scheduler = Scheduler(config=config, workers=self.workers)
        logger.info(f"All Workers init successfully.")
        
        self.scheduler.update_worker_status()
        
        self.is_finish = False
        
        self.monitor = get_monitor()
        
        self.prev_schedule_start_time = None

        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        logger.info("Engine is running.")

    def finish(self):
        if getattr(self, "_closed", False):
            return

        self.is_finish = True

        if self.thread.is_alive():
            self.thread.join()

        if getattr(self, "scheduler", None) is not None:
            self.scheduler.shutdown()

        self._closed = True

    def run(self):
        """
        Continuously call Scheduler.schedule to get pending requests and assign them to idle workers.
        """
        futures: dict[int, Future] = {}
        outputs: dict[int, any] = {}
        requests: list[Request] = []
        prev_requests: list[Request] = []
        scheduler_start_time = time.perf_counter()
        while True:
            prev_requests = requests
            logger.debug(f"调用 scheduler.schedule。当前队列状态: {len(self.scheduler.waiting)} 等待中, {len(self.scheduler.running)} 运行中。")
            self.prev_schedule_start_time = scheduler_start_time
            scheduler_start_time = time.perf_counter()
            requests = self.scheduler.schedule(prev_requests)
            scheduler_end_time = time.perf_counter()
            if len(requests) > 0:
                self.monitor.record("schedule_latency", scheduler_end_time-scheduler_start_time)
            
            logger.debug(f"调度器返回了 {len(requests)} 个请求进行处理。")
            
            # Assign requests to workers based on Request.inputs.
            # The mapping between inputs and workers is stored in a dict.
            # The key is worker_id and the value is a list of (request_id, slice_id).
            # After a worker finishes, use the KV cache manager to merge results and update the Request object.

            # Wait for the previous round of worker.run computations to finish.
            outputs: dict[int, any] = {}
            for worker_id, future in list(futures.items()):
                logger.debug(f"正在等待 Worker {worker_id} 的 future.result()...")
                # Block until each future completes.
                outputs[worker_id] = future.result()
                logger.debug(f"Worker {worker_id} 的 future 已完成。")

            if self.is_finish:
                break
            
            futures = {}

            # Call scheduler.update_request.
            logger.debug("正在调用 deal_outputs...")
            if self.use_generate:
                self.deal_generate_outputs(outputs, prev_requests)
            else:
                self.deal_outputs(outputs, prev_requests)
                
            logger.debug("正在调用 scheduler.update_status...")
            self.scheduler.update_status(prev_requests)

            # Call worker.process in a non-blocking way.
            worker_inputs = {i: [] for i in range(self.device_count)}
            requests = [req for req in requests if req.status != 'completed']
            for request in requests:
                # Assume request.inputs is a list and each element is assigned to one worker.
                for i, input in enumerate(request.inputs):
                    worker_id = input.target_device
                    worker_inputs[worker_id].append(input.input_data)
                    input.batch_index = len(worker_inputs[worker_id]) - 1  # Set the index within the batch.
                if request.status == 'pending':
                    request.status = 'prefill'
                elif request.status == 'prefill':
                    request.status = 'decode'

            for worker_id, inputs in worker_inputs.items():
                if inputs:
                    logger.debug(f"正在提交 {len(inputs)} 个输入数据到 Worker {worker_id} 进行异步处理...")
                    future = self.workers[worker_id].process_async(inputs)
                    futures[worker_id] = future

    def deal_generate_outputs(self, outputs: dict[int, any], requests: list[Request]):
        for request in requests:
            input_data = request.inputs[0].input_data
            worker_id = request.inputs[0].target_device
            output = outputs[worker_id]
            generated_ids_trimmed = output[0][input_data.data["input_ids"].shape[1]:]
            if self.config.model.model_name == 'QwenVL':
                answers = QwenVL.processor.batch_decode([generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                request.answer = answers[0]
            request.status = "completed"

    def deal_outputs(self, outputs: dict[int, any], requests: list[Request]):
        """
        Process worker outputs and update the corresponding Request objects.

        Args:
            results (dict[int, any]): Mapping of worker_id to their output results.
            requests (list[Request]): List of Request objects that were processed.
        """
        for output in outputs.values():
            past_key_values :DynamicCache = output["past_key_values"]
            batch_size = past_key_values.key_cache[0].shape[0]
            output["past_key_values"] = past_key_values.batch_split(batch_size, 1)
        for request in requests:
            # Get the corresponding worker output using batch_index in request.inputs.
            req_id = request.request_id
            logger.debug(f"Request {req_id}: 正在处理 '{request.status}' 阶段的输出...")
            output = {}
            output["kv_cache"] = []
            for input_data in request.inputs:
                worker_id = input_data.target_device
                batch_index = input_data.batch_index
                past_key_values = outputs[worker_id]["past_key_values"]
                # Keep the batch dimension with length 1.
                # Slice key and value by batch_index separately while preserving the tuple structure.
                seq_len = input_data.input_data.all_input_ids.shape[1]
                output["kv_cache"].append(past_key_values[batch_index])
                seq_len = 1 if request.status == 'decode' else input_data.input_data.all_input_ids.shape[1]
                output["logits"] = outputs[worker_id]["logits"][batch_index:batch_index+1, seq_len-1:seq_len, ...]
            if request.status == 'prefill':
                decode_input = ModelInput()
                decode_input.target_device = request.decode_device
                decode_input_data = InputData(data=BatchFeature({}))
                decode_input.input_data = decode_input_data
                decode_input_data.all_input_ids = torch.cat([model_input.input_data.all_input_ids.to(request.decode_device) for model_input in request.inputs], dim=-1)
                decode_input_data.computed_tokens = decode_input_data.all_input_ids.shape[1]
                merge_kv_cache_start_time = time.perf_counter()
                if self.config.model.model_name == "MiniCPM":
                    MiniCPMO_util.deal_prefill_output(output, decode_input, request)
                elif self.config.model.model_name == "QwenVL":
                    QwenVL_util.deal_prefill_output(output, decode_input, request)
                merge_kv_cache_end_time = time.perf_counter()
                self.monitor.record("merge_kv_cache_latency", merge_kv_cache_end_time-merge_kv_cache_start_time)
                request.inputs = [decode_input]
                token_id = self.tokenizer.encode("Answer", add_special_tokens=False)[0]
                next_token = torch.tensor([token_id]).to(request.decode_device)
                prefill_end_time = time.perf_counter()
                self.monitor.record("TTFT_latency", prefill_end_time-self.prev_schedule_start_time)
            elif request.status == 'decode':
                if len(output["kv_cache"]) > 1:
                    logger.warning(f"Request {request.request_id} in decode status has multiple kv_caches")
                request.inputs[0].input_data.computed_tokens += 1
                if self.config.model.model_name == "MiniCPM":
                    MiniCPMO_util.deal_decode_output(output, request)
                elif self.config.model.model_name == "QwenVL":
                    QwenVL_util.deal_decode_output(output, request)
                next_token = get_next_token(output["logits"]).to(request.decode_device)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                request.status = 'completed'
            else:
                decoded_token = self.tokenizer.decode(next_token.item())
                input_data = request.inputs[0].input_data
                input_data.all_input_ids = torch.cat([input_data.all_input_ids, next_token.unsqueeze(1)], dim=-1)
                request.answer += decoded_token
                request.generate_len += 1
                if request.max_generate_len is not None and request.generate_len>=request.max_generate_len:
                    request.status = 'completed'
                logger.debug(f"输出token：{decoded_token}")

    def generate(self, inputs, params) -> list[str]:
        """
        Perform parallel inference on the given inputs with specified parameters.
        Args:
            inputs: Input data for inference.
            params: Parameters for generation.
        Returns:
            str: Generated output from the model.
        """

        """
        Structure of each element in inputs:
        {
            "prompt": str,
            "multi_modal_data": {
                "type": "video" or "image",
                "path": str, # Path to the video file or image folder.
            }
        }
        """

        requests: list[Request] = []
        for input in inputs:
            cur_request_id = self.request_id.increment()
            request = Request(data=input, params=params, request_id=cur_request_id)
            requests.append(request)

        self.scheduler.add_requests(requests)

        # Block until all requests are completed.
        answers = []
        for request in requests:
            result = request.future.result()
            answers.append(result)
        return answers
