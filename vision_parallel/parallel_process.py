import multiprocessing as mp
import threading
import time
import os, math
from PIL import Image
from moviepy import VideoFileClip
import logging
import numpy as np

from vision_parallel.video_util import get_scene_list
from vision_parallel.config import Config

class SimplePipeWorker:
    def __init__(self, config: Config):
        # Create the pipe.
        self.parent_conn, self.child_conn = mp.Pipe()
        
        # Start the worker process.
        ctx = mp.get_context('spawn')
        
        self.adaptive_threshold = config.test.adaptive_threshold

        self.worker = ctx.Process(
            target=self._worker_process,
            args=(self.child_conn, self.adaptive_threshold),
            daemon=True
        )
        
        self.worker.start()
        
        # Task ID counter.
        self.task_id = 0
        self.lock = threading.Lock()
    
    @staticmethod
    def _worker_process(conn, adaptive_threshold):
        """Main loop for the worker process."""
        while True:
            try:
                # Receive a task.
                task = conn.recv()
                if task is None:  # Stop signal.
                    break
                
                task_id, video_path, scene_detection_parallel_slice = task
                
                try:
                    # Execute the task.
                    result = get_scene_list(video_path, scene_detection_parallel_slice, adaptive_threshold)
                    # Send the successful result.
                    conn.send((task_id, 'success', result))
                except Exception as e:
                    # Send the error result.
                    conn.send((task_id, 'error', e))
                    
            except EOFError:
                break
            except Exception as e:
                # Send an unknown error.
                conn.send((task_id if 'task_id' in locals() else -1, 'error', e))
    
    def submit_task(self, video_path, scene_detection_parallel_slice):
        """Submit a task."""
        with self.lock:
            self.task_id += 1
            current_task_id = self.task_id
        
        # Send the task to the worker process.
        self.parent_conn.send((current_task_id, video_path, scene_detection_parallel_slice))
        
        return SimpleFuture(current_task_id, self.parent_conn)
    
    def shutdown(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True

        try:
            self.parent_conn.send(None)
        except (BrokenPipeError, EOFError, OSError):
            pass

        if self.worker.is_alive():
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                self.worker.terminate()
                self.worker.join(timeout=1)

        try:
            self.parent_conn.close()
        except OSError:
            pass

        try:
            self.child_conn.close()
        except OSError:
            pass

class SimpleFuture:
    """A simple Future implementation."""
    def __init__(self, task_id, conn):
        self.task_id = task_id
        self.conn = conn
    
    def result(self, timeout=3000):
        """Get the result."""
        if self.conn.poll(timeout):  # Wait for the result.
            result_task_id, status, result = self.conn.recv()
            
            # Check that the task ID matches.
            if result_task_id != self.task_id:
                raise RuntimeError(f"Task ID mismatch: expected {self.task_id}, got {result_task_id}")
            
            # Process the result.
            if status == 'error':
                raise result
            else:
                return result
        else:
            raise TimeoutError(f"Task {self.task_id} timeout after {timeout}s")
