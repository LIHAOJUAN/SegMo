import numpy as np
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional

metrics_name = [
    "video_duration",
    "schedule_latency",
    "pre_process_latency",
    "construct_frames_from_video_latency",
    "detect_scene_latency",
    "wait_scene_list_latency",
    "slice_frames_by_scene_with_GPU_memory_latency",
    "slice_frame_latency",
    "CLIP_latency",
    "frames_num",
    "scene_num",
    "prepare_input_latency",
    "seq_len",
    "vision_encoder_latency",
    "prefill_latency",
    "merge_kv_cache_latency",
    "decode_latency",
    "get_target_inputs_latency",
    "TTFT_latency",
]

class Monitor:
    def __init__(self):
        self.records = {}
        self.locks = {}
        for metric_name in metrics_name:
            self.records[metric_name] = []
            self.locks[metric_name] = threading.Lock()
    
    def record(self, metric_name, data):
        if metric_name not in self.records:
            return
        with self.locks[metric_name]:
            self.records[metric_name].append(data)
    
    def calculate_statistics(self, metric_name: str) -> Optional[Dict]:
        """Calculate statistics for a single metric."""
        if metric_name not in self.records or not self.records[metric_name]:
            return None
    
        data = np.array(self.records[metric_name])
        
        if len(data) == 0:
            return None
        
        data_min = np.min(data)
        data_max = np.max(data)
        data_sum = np.sum(data)
        
        if len(data) > 2:
            data_mean = (data_sum-data_max-data_min)/(len(data)-2)
        else:
            data_mean = data_sum/len(data)
        
        return {
            "count": len(data),
            "mean": float(data_mean),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "min": float(data_min),
            "max": float(data_max),
            "p50": float(np.percentile(data, 50)),
            "p90": float(np.percentile(data, 90)),
            "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99)),
            "sum": float(data_sum)
        }
    
    def get_all_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all metrics."""
        statistics = {}
        for metric_name in self.records:
            stats = self.calculate_statistics(metric_name)
            if stats is not None:
                statistics[metric_name] = stats
        
        return statistics
    
    def print_statistics(self, metric_names: Optional[List[str]] = None):
        """Print statistics."""
        if metric_names is None:
            metric_names = list(self.records.keys())
        
        print(f"\n{'='*80}")
        print(f"Performance Statistics Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        for metric_name in metric_names:
            stats = self.calculate_statistics(metric_name)
            if stats is None:
                print(f"\n{metric_name}: No data available")
                continue
            
            print(f"\n{metric_name}:")
            print(f"  Count:  {stats['count']:>10}")
            print(f"  Mean:   {stats['mean']:>10.3f}")
            print(f"  Median: {stats['median']:>10.3f}")
            print(f"  Std:    {stats['std']:>10.3f}")
            print(f"  Min:    {stats['min']:>10.3f}")
            print(f"  Max:    {stats['max']:>10.3f}")
            print(f"  P50:    {stats['p50']:>10.3f}")
            print(f"  P90:    {stats['p90']:>10.3f}")
            print(f"  P95:    {stats['p95']:>10.3f}")
            print(f"  P99:    {stats['p99']:>10.3f}")
            print(f"  Sum:    {stats['sum']:>10.3f}")
    
    def save_statistics_to_json(self, filename: str = 'performance_statistics.json'):
        """Save statistics to a JSON file."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'statistics': self.get_all_statistics(),
            'raw_data': self.records
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Statistics saved to {filename}")
    
    def get_summary_table(self) -> str:
        """Generate a summary table string."""
        stats = self.get_all_statistics()
        if not stats:
            return "No data available for statistics"
        
        # Header row.
        header = f"{'Metric':<25} {'Count':<8} {'Mean':<10} {'P50':<10} {'P90':<10} {'P99':<10}"
        separator = "-" * len(header)
        
        lines = [header, separator]
        
        # Data rows.
        for metric_name, metric_stats in stats.items():
            line = (f"{metric_name:<25} "
                   f"{metric_stats['count']:<8} "
                   f"{metric_stats['mean']:<10.3f} "
                   f"{metric_stats['p50']:<10.3f} "
                   f"{metric_stats['p90']:<10.3f} "
                   f"{metric_stats['p99']:<10.3f}")
            lines.append(line)
        
        return "\n".join(lines)
    
    def reset_metrics(self, metric_names: Optional[List[str]] = None):
        """Reset data for the specified metrics."""
        if metric_names is None:
            metric_names = list(self.records.keys())
        
        for metric_name in metric_names:
            if metric_name in self.records:
                self.records[metric_name] = []
    
    def get_metric_data(self, metric_name: str) -> List:
        """Get raw data for the specified metric."""
        return self.records.get(metric_name, [])

monitor = Monitor()

def get_monitor():
    global monitor
    return monitor
