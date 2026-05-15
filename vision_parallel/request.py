import concurrent.futures
from transformers.cache_utils import DynamicCache

class ModelInput:
    def __init__(self):
        self.slice_index = 1 # Indicates which video slice this is.
        self.target_device = 0
        self.input_data: InputData = None
        self.batch_index = 0  # Indicates this slice's position in the target worker's batch.

class InputData:
    def __init__(self, data):
        self.data = data
        self.computed_tokens = 0
        self.all_input_ids = []
        self.all_position_ids = []
        self.past_key_values = DynamicCache()

class Request:
    def __init__(self, data, params, request_id):
        self.data = data
        self.params = params
        self.request_id = request_id
        self.answer = ""
        self.status = 'pending'  # could be 'pending', 'prefill', 'decode' ,'completed'
        self.inputs: list[ModelInput] = []  # List of ModelInput objects
        self.decode_device = 0
        self.future = concurrent.futures.Future()
        self.generate_len = 0
        self.max_generate_len = None
        
        if params != None and "max_generate_len" in params:
            self.max_generate_len = params["max_generate_len"]

"""
Data structure:
{
    "prompt": str,
    "multi_modal_data": {
        "type": "video" or "image",
        "path": str, # Path to the video file or image folder.
    }
}
"""
