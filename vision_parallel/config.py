import yaml
import os
from PIL import Image

DEFAULT_MODEL_NAME = "llama"
DEFAULT_MODEL_PATH = "~/models/llama"
DEFAULT_CLIP_MODEL_PATH = "ViT-B/32"

class Config:
    def __init__(self, model_config_path="config.yaml", test_config_path="test.yaml"):
        self.model_config_path = model_config_path
        self.test_config_path = test_config_path
        self.model = ModelConfig()
        self.test = TestConfig()
        self.load_config()
    
    def load_config(self):
        if not os.path.exists(self.model_config_path):
            print(f"{self.model_config_path} does not exist. Setting default configuration.")
            self.model.set_defaults()
            return
        with open(self.model_config_path, 'r', encoding='utf-8') as f:
            model_config_data = yaml.safe_load(f)
        self.model.set(model_config_data)
        
        if not os.path.exists(self.test_config_path):
            print(f"{self.test_config_path} does not exist. Setting default test configuration.")
            self.test.set_defaults()
            return
        with open(self.test_config_path, 'r', encoding='utf-8') as f:
            test_config_data = yaml.safe_load(f)
        self.test.set(test_config_data)

class ModelConfig:
    def set_defaults(self):
        self.model_name = DEFAULT_MODEL_NAME
        self.model_path = DEFAULT_MODEL_PATH
        self.CLIP_model_path = DEFAULT_CLIP_MODEL_PATH
    def set(self, model_config_data):
        if 'model' in model_config_data:
            model_config = model_config_data['model']
            self.model_name = model_config.get('name', DEFAULT_MODEL_NAME)
            self.model_path = model_config.get('path', DEFAULT_MODEL_PATH)
        self.CLIP_model_path = model_config_data.get("CLIP_model_path", DEFAULT_CLIP_MODEL_PATH)

class TestConfig:
    def set_defaults(self):
        self.parallel_num = 1
        self.text = "give a brief introduction"
        self.type = "text"
        self.image_list = []
        self.audio_list = []
        self.image_slice_num = 1
        self.data_path = ""
        self.video_slice_num = 1
        self.worker_num = 1
        self.max_frames_per_video = 64
        self.scene_detection_parallel_slice = 4
        self.log_path = "inference.log"
        self.output_path = None
        self.monitor_path = None
        self.scene_difference_weight = 0.5
        self.scene_relevance_weight = 0.5
        self.adaptive_threshold = 3.0
        self.use_generate = False
        self.static_sample_threshold = 0
        
    def set(self, test_config_data):
        self.parallel_num = test_config_data.get('parallel_num', 1)
        self.text = test_config_data.get('text', "give a brief introduction")
        self.type = test_config_data.get('type', "text")

        if 'image' not in test_config_data:
            self.image_list = []
        else:
            if isinstance(test_config_data['image'], str):
                self.image_list = [Image.open(test_config_data['image']).convert("RGB")]
            elif isinstance(test_config_data['image'], list):
                self.image_list = [Image.open(img).convert("RGB") for img in test_config_data['image']]
            else:
                raise ValueError("Invalid image format in test configuration. Expected str or list of str.")

        if 'audio' not in test_config_data:
            self.audio_list = []
        else:
            if isinstance(test_config_data['audio'], str):
                self.audio_list = [test_config_data['audio']]
            elif isinstance(test_config_data['audio'], list):
                self.audio_list = test_config_data['audio']
            else:
                raise ValueError("Invalid audio format in test configuration. Expected str or list of str.")

        if 'video' not in test_config_data:
            self.video = None
        else:
            if isinstance(test_config_data['video'], str):
                self.video = test_config_data['video']
            else:
                raise ValueError("Invalid video format in test configuration. Expected str.")
        
        if 'slice' not in test_config_data:
            self.slice = [(0, 9)]
        else:
            self.slice = test_config_data['slice']

        if 'image_slice_num' not in test_config_data:
            self.image_slice_num = 1
        else:
            self.image_slice_num = test_config_data['image_slice_num']
        
        if 'data_path' not in test_config_data:
            self.data_path = ""
        else:
            self.data_path = test_config_data['data_path']

        if 'video_slice_num' not in test_config_data:
            self.video_slice_num = 1
        else:
            self.video_slice_num = test_config_data['video_slice_num']

        if 'output_path' not in test_config_data:
            self.output_path = None
        else:
            self.output_path = test_config_data['output_path']
        
        if 'worker_num' not in test_config_data:
            self.worker_num = 1
        else:
            self.worker_num = test_config_data['worker_num']

        if 'max_frames_per_video' not in test_config_data:
            self.max_frames_per_video = 64
        else:
            self.max_frames_per_video = test_config_data['max_frames_per_video']
            
        if 'scene_detection_parallel_slice' not in test_config_data:
            self.scene_detection_parallel_slice = 4 
        else:
            self.scene_detection_parallel_slice = test_config_data['scene_detection_parallel_slice']
        
        if 'log_path' not in test_config_data:
            self.log_path = "inference.log"
        else:
            self.log_path = test_config_data['log_path']
        
        if 'monitor_path' not in test_config_data:
            self.monitor_path = None
        else:
            self.monitor_path = test_config_data['monitor_path']
        
        if 'scene_difference_weight' not in test_config_data:
            self.scene_difference_weight = 0.5
        else:
            self.scene_difference_weight = test_config_data['scene_difference_weight']

        if 'scene_relevance_weight' not in test_config_data:
            self.scene_relevance_weight = 0.5
        else:
            self.scene_relevance_weight = test_config_data['scene_relevance_weight']
        
        if 'adaptive_threshold' not in test_config_data:
            self.adaptive_threshold = 3.0
        else:
            self.adaptive_threshold = test_config_data['adaptive_threshold']
        
        if 'use_generate' not in test_config_data:
            self.use_generate = False
        else:
            self.use_generate = test_config_data['use_generate']
        
        if 'static_sample_threshold' not in test_config_data:
            self.static_sample_threshold = 0
        else:
            self.static_sample_threshold = test_config_data['static_sample_threshold']