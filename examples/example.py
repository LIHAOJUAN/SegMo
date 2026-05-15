import os
import logging

from vision_parallel.engine import ParallelInferenceEngine
from vision_parallel.config import Config
from vision_parallel.util import init_log

if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    model_config_path = os.path.join(current_dir, "../model_config/example.yaml")
    test_config_path = os.path.join(current_dir, "../test_config/example.yaml")

    config = Config(model_config_path, test_config_path)
    try:
        init_log(config.test.log_path)
    except Exception as e:
        print(str(e))
        exit(0)
    logger = logging.getLogger("vision-parallel")
    
    engine: ParallelInferenceEngine = ParallelInferenceEngine(config=config)

    video_path = config.test.data_path
    question = "What happens in the video?"
    
    mm_data = {"type": "video", "path": video_path}
    inputs = [{
        "prompt_header": question,
        "multi_modal_data": mm_data
    }]

    params = {
        "max_generate_len": 1000
    }
    
    response = engine.generate(inputs, params)
    
    print(response)            
    
    engine.finish()