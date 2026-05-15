import matplotlib.pyplot as plt
import numpy as np
import torch
import threading
import os
import srt
import logging
import sys

generation_config = {
    "top_p": 0.8,
    "top_k": 100,
    "temperature": 0.7,
    "do_sample": True,
    "repetition_penalty": 1.05,
}

def init_log(log_path: str):
    if log_path.find("try") == -1 and os.path.exists(log_path):
        raise FileExistsError(f"日志文件已存在: {log_path}")
    class MyFilter(logging.Filter):
        def filter(self, record):
            return record.name.startswith("vision-parallel")
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.addFilter(MyFilter())
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.addFilter(MyFilter())
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            file_handler,
            stream_handler
        ]
    )

class AtomicInteger:
    def __init__(self, value=0):
        self._value = value
        self._lock = threading.Lock()
    
    def get(self):
        with self._lock:
            return self._value
    
    def set(self, value):
        with self._lock:
            self._value = value
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    def compare_and_swap(self, expected, new_value):
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False

import re

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Answer",
        "answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""
    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        return ""
    return matches[0]

def get_subtitle(subtitle_path):
    subtitles_text = ""
    if not os.path.exists(subtitle_path):
        return ""
    else:
        with open(subtitle_path, "r", encoding="utf-8") as f:
            srt_content = f.read()
        subtitles = list(srt.parse(srt_content))
        if len(subtitles) == 0:
            return ""
        subtitles_text = "This video's subtitles are listed below:\n"
        for sub in subtitles:
            content = re.sub(r"<.*?>", "", sub.content)
            subtitles_text += content
            subtitles_text += "\n"

    return subtitles_text

def get_prompt_header(question, options=None, subtitle_path = None):
    subtitles_text = ""
    if subtitle_path is not None:
        subtitles_text = get_subtitle(subtitle_path)

    instruction = f"""Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.
"""

    if options is not None:
        prompt_header = f"""{subtitles_text}\n
        {instruction}\n
        {question}\n
        {options}\n
        Answer with the option's letter from the given choices directly.\n
        """
    else:
        prompt_header = f"""{subtitles_text}\n
        {instruction}\n
        {question}\n
        Answer with the option's letter from the given choices directly.\n
        """
    
    return prompt_header
