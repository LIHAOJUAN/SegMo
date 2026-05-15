from abc import ABC
import torch
import concurrent.futures
import types
import logging
from transformers import AutoModel, Qwen2VLForConditionalGeneration

from vision_parallel.config import Config
from vision_parallel.request import Request, ModelInput, InputData
import vision_parallel.models.MiniCPM.MiniCPMO as MiniCPMO
import vision_parallel.models.QwenVL.QwenVL as QwenVL

logger = logging.getLogger("vision-parallel")

class Worker(ABC):
    def __init__(self, config: Config, target_device: int):
        """
        Initialize the Worker instance.
        Args:
            config (Config): Configuration object for the worker.
            target_device (int): The target device ID for the worker.
        """
        self.config = config
        self.model_name = config.model.model_name
        self.use_generate = config.test.use_generate
        if self.model_name == 'MiniCPM':
            self.model = AutoModel.from_pretrained(
                config.model.model_path, 
                trust_remote_code=True,
                attn_implementation='sdpa',
                torch_dtype=torch.bfloat16
            ).eval().cuda(target_device)
            self.model.init_tts()
        elif self.model_name == 'QwenVL':
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                config.model.model_path, 
                device_map={ "": f"cuda:{target_device}" },
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            ).eval()
        logger.info(f"Worker (device:{target_device}): model is load to cuda:{target_device}。")

        if self.model_name == 'MiniCPM':
            MiniCPMO.prepare_model(self.model)
        elif self.model_name == 'QwenVL':
            QwenVL.prepare_model(self.model)

        self.rotary_emb = None
        if self.model_name == 'MiniCPM':
            self.rotary_emb = self.model.llm.model.rotary_emb

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def process_async(self, batch_inputs):
        """
        Asynchronously process batch_inputs and return a Future object.
        """
        if self.use_generate:
            return self.executor.submit(self.generate, batch_inputs)
        else:
            return self.executor.submit(self.forward, batch_inputs)

    def forward(self, batch_inputs: list[InputData]):
        """
        Process the given batch inputs using the worker's model.
        Args:
            batch_inputs: Inputs to be processed by the model.
        Returns:
            Model outputs after processing the inputs.
        """
        logger.debug(f"Worker (device:{self.model.device}): 开始处理一个批次，包含 {len(batch_inputs)} 个输入。")
        if len(batch_inputs) == 0:
            return None
        
        if self.model_name == 'MiniCPM':
            outputs = MiniCPMO.forward(self.model, batch_inputs)
        elif self.model_name == 'QwenVL':
            outputs = QwenVL.forward(self.model, batch_inputs)

        logger.debug(f"Worker (device:{self.model.device}): 批次处理完成。")
        return outputs

    def generate(self, batch_inputs: list[InputData]):
        if len(batch_inputs) == 0:
            return None

        if self.model_name == 'MiniCPM':
            outputs = MiniCPMO.generate(self.model, batch_inputs)
        elif self.model_name == 'QwenVL':
            outputs = QwenVL.generate(self.model, batch_inputs)

        return outputs
