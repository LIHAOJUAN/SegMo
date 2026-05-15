import json

def get_target_video_index(duration_file_path, min_duration, max_duration):
    with open(duration_file_path, 'r', encoding='utf-8') as f:
        duration_map = json.load(f)
        target_key = f"{min_duration}_{max_duration}"
        if target_key not in duration_map:
            return []
        target_index_list = [item[0] for item in duration_map[target_key]]
        return target_index_list