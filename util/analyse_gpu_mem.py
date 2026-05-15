"""
GPU memory analysis utility module

Provides detailed GPU memory usage analysis features, including:
- GPU memory usage monitoring
- Tensor footprint analysis
- Tracking memory changes between iterations
- Memory footprint analysis for specific variables
"""

import torch
import gc
import inspect


def print_gpu_memory_usage(step=""):
    """
    Print current GPU memory usage.
    
    Args:
        step (str): Description of the current step.
        
    Returns:
        float: Currently allocated GPU memory in GB.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[{step}] GPU 显存 - 已分配: {allocated:.3f} GB, 缓存: {reserved:.3f} GB, 峰值: {max_allocated:.3f} GB")
        return allocated
    else:
        print("CUDA 不可用")
        return 0


def get_all_gpu_tensors(include_var_names=True):
    """
    Get information about all GPU tensors.
    
    Args:
        include_var_names (bool): Whether to include variable names.
    
    Returns:
        tuple: (tensors_info, total_memory)
            - tensors_info (dict): Tensor information dictionary.
            - total_memory (int): Total GPU memory usage in bytes.
    """
    tensors_info = {}
    total_memory = 0
    
    # Get the variable-name mapping.
    var_names = {}
    if include_var_names:
        var_names = _get_tensor_variable_names()
    
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            size_bytes = obj.numel() * obj.element_size()
            total_memory += size_bytes
            
            # Get the variable name.
            var_name = var_names.get(id(obj), "unknown")
            
            # Group by tensor shape and size.
            key = f"shape_{tuple(obj.shape)}_dtype_{obj.dtype}_device_{obj.device}"
            if key in tensors_info:
                tensors_info[key]['count'] += 1
                tensors_info[key]['total_size'] += size_bytes
                tensors_info[key]['var_names'].append(var_name)
            else:
                tensors_info[key] = {
                    'shape': obj.shape,
                    'dtype': obj.dtype,
                    'device': obj.device,
                    'count': 1,
                    'total_size': size_bytes,
                    'var_names': [var_name]
                }
    
    return tensors_info, total_memory


def compare_tensors(tensors_before, tensors_after, step=""):
    """
    Compare the difference between two tensor snapshots.
    
    Args:
        tensors_before (dict): Tensor information before.
        tensors_after (dict): Tensor information after.
        step (str): Description of the current step.
    """
    print(f"\n=== [{step}] 显存变化分析 ===")
    
    # Find newly added tensors.
    new_tensors = {}
    for key, info in tensors_after.items():
        if key not in tensors_before:
            new_tensors[key] = info
        elif info['count'] > tensors_before[key]['count']:
            # Tensor count increased for the same type.
            diff_count = info['count'] - tensors_before[key]['count']
            diff_size = info['total_size'] - tensors_before[key]['total_size']
            
            # Find newly added variable names.
            old_var_names = set(tensors_before[key].get('var_names', []))
            new_var_names = set(info.get('var_names', []))
            added_var_names = list(new_var_names - old_var_names)
            
            new_tensors[f"{key}_增加"] = {
                'shape': info['shape'],
                'dtype': info['dtype'],
                'count': diff_count,
                'total_size': diff_size,
                'var_names': added_var_names
            }
    
    if new_tensors:
        print("新增/增加的张量:")
        for key, info in sorted(new_tensors.items(), key=lambda x: x[1]['total_size'], reverse=True):
            size_mb = info['total_size'] / 1024**2
            if size_mb > 0.1:  # Only show tensors larger than 0.1 MB.
                var_names_str = ", ".join(info.get('var_names', ['unknown'])[:5])  # Show up to 5 variable names.
                if len(info.get('var_names', [])) > 5:
                    var_names_str += "..."
                print(f"  {info['shape']} ({info['dtype']}) x{info['count']}: {size_mb:.2f} MB")
                print(f"    变量名: {var_names_str}")
    else:
        print("没有发现新的显存占用")


def analyze_specific_variables(**variables):
    """
    Analyze GPU memory usage of specific variables.
    
    Args:
        **variables: Variables to analyze, passed in as keyword arguments.
    """
    print("\n--- 关键变量显存占用 ---")
    for name, var in variables.items():
        if torch.is_tensor(var):
            if var.is_cuda:
                size_mb = var.numel() * var.element_size() / 1024**2
                print(f"{name}: {var.shape} ({var.dtype}) = {size_mb:.2f} MB")
        elif isinstance(var, (list, tuple)) and len(var) > 0:
            if torch.is_tensor(var[0]) and hasattr(var[0], 'is_cuda'):
                # Handle nested structures such as KV cache.
                total_size = 0
                for i, item in enumerate(var):
                    if isinstance(item, (list, tuple)):
                        for j, subitem in enumerate(item):
                            if torch.is_tensor(subitem) and subitem.is_cuda:
                                total_size += subitem.numel() * subitem.element_size()
                    elif torch.is_tensor(item) and item.is_cuda:
                        total_size += item.numel() * item.element_size()
                size_mb = total_size / 1024**2
                print(f"{name}: {len(var)} layers, 总计 {size_mb:.2f} MB")


def get_tensor_memory_usage():
    """
    Get memory usage statistics for all GPU tensors.
    
    Returns:
        tuple: (total_memory, tensor_count)
    """
    total_memory = 0
    tensor_count = 0
    
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            size_bytes = obj.numel() * obj.element_size()
            total_memory += size_bytes
            tensor_count += 1
    
    print(f"总共 {tensor_count} 个 GPU 张量，占用显存: {total_memory / 1024**3:.2f} GB")
    return total_memory, tensor_count


def get_variable_memory(var_name, var):
    """
    Get the GPU memory usage of a single variable.
    
    Args:
        var_name (str): Variable name.
        var: Variable object.
        
    Returns:
        float: GPU memory usage in GB.
    """
    if torch.is_tensor(var) and var.is_cuda:
        size_gb = var.numel() * var.element_size() / 1024**3
        print(f"{var_name}: {var.shape}, {size_gb:.4f} GB")
        return size_gb
    return 0


def monitor_memory_in_loop(func, *args, step_name="", threshold_mb=1.0, **kwargs):
    """
    Decorator function for monitoring memory changes inside a loop.
    
    Args:
        func: Function to monitor.
        *args: Function arguments.
        step_name (str): Step name.
        threshold_mb (float): Memory increase threshold in MB; detailed analysis runs only above this value.
        **kwargs: Function keyword arguments.
        
    Returns:
        Function execution result.
    """
    # State before execution.
    tensors_before, memory_before = get_all_gpu_tensors()
    allocated_before = torch.cuda.memory_allocated() / 1024**3
    
    # Execute the function.
    result = func(*args, **kwargs)
    
    # State after execution.
    tensors_after, memory_after = get_all_gpu_tensors()
    allocated_after = torch.cuda.memory_allocated() / 1024**3
    
    # Analyze the changes.
    memory_increase = allocated_after - allocated_before
    if memory_increase > threshold_mb / 1024:  # Convert to GB.
        print(f"\n[{step_name}] 显存增加: {memory_increase:.3f} GB")
        compare_tensors(tensors_before, tensors_after, step_name)
    
    return result


def clear_gpu_cache():
    """Clear the GPU cache."""
    torch.cuda.empty_cache()
    print("GPU 缓存已清理")


def detailed_kv_cache_analysis(kv_cache, cache_name="KV Cache"):
    """
    Perform a detailed analysis of KV Cache GPU memory usage.
    
    Args:
        kv_cache: KV Cache object.
        cache_name (str): Cache name.
    """
    print(f"\n--- {cache_name} 详细分析 ---")
    if kv_cache is None:
        print("KV Cache 为空")
        return
    
    total_size = 0
    for i, layer_cache in enumerate(kv_cache):
        if isinstance(layer_cache, (list, tuple)) and len(layer_cache) >= 2:
            key_tensor, value_tensor = layer_cache[0], layer_cache[1]
            if torch.is_tensor(key_tensor) and torch.is_tensor(value_tensor):
                key_size = key_tensor.numel() * key_tensor.element_size()
                value_size = value_tensor.numel() * value_tensor.element_size()
                layer_size = key_size + value_size
                total_size += layer_size
                
                key_size_mb = key_size / 1024**2
                value_size_mb = value_size / 1024**2
                layer_size_mb = layer_size / 1024**2
                
                print(f"Layer {i}: Key {key_tensor.shape} ({key_size_mb:.2f} MB), "
                      f"Value {value_tensor.shape} ({value_size_mb:.2f} MB), "
                      f"Total: {layer_size_mb:.2f} MB")
    
    total_size_gb = total_size / 1024**3
    print(f"{cache_name} 总计: {total_size_gb:.3f} GB")
    return total_size


class MemoryMonitor:
    """Memory monitor for continuously tracking GPU memory usage during code execution."""
    
    def __init__(self, name="MemoryMonitor"):
        self.name = name
        self.history = []
        self.baseline_tensors = None
        self.baseline_memory = 0
    
    def set_baseline(self):
        """Set the baseline memory state."""
        self.baseline_tensors, self.baseline_memory = get_all_gpu_tensors()
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"[{self.name}] 基线显存设置: {allocated:.3f} GB")
    
    def checkpoint(self, step_name=""):
        """Record a checkpoint."""
        allocated = torch.cuda.memory_allocated() / 1024**3
        tensors_info, total_memory = get_all_gpu_tensors()
        
        checkpoint_data = {
            'step': step_name,
            'allocated': allocated,
            'tensors_info': tensors_info,
            'total_memory': total_memory
        }
        self.history.append(checkpoint_data)
        
        if len(self.history) > 1:
            prev_allocated = self.history[-2]['allocated']
            increase = allocated - prev_allocated
            if increase > 0.001:  # 1MB
                print(f"[{self.name}] {step_name}: +{increase:.3f} GB (当前: {allocated:.3f} GB)")
    
    def compare_with_baseline(self):
        """Compare against the baseline."""
        if self.baseline_tensors is None:
            print("未设置基线")
            return
        
        current_tensors, current_memory = get_all_gpu_tensors()
        current_allocated = torch.cuda.memory_allocated() / 1024**3
        baseline_allocated = self.baseline_memory / 1024**3
        
        increase = current_allocated - baseline_allocated
        print(f"[{self.name}] 与基线相比增加: {increase:.3f} GB")
        
        if increase > 0.001:
            compare_tensors(self.baseline_tensors, current_tensors, "与基线比较")
    
    def summary(self):
        """Print the monitoring summary."""
        if not self.history:
            print("没有监控数据")
            return
        
        print(f"\n=== {self.name} 监控摘要 ===")
        for i, checkpoint in enumerate(self.history):
            if i == 0:
                print(f"{checkpoint['step']}: {checkpoint['allocated']:.3f} GB (起始)")
            else:
                prev = self.history[i-1]['allocated']
                diff = checkpoint['allocated'] - prev
                print(f"{checkpoint['step']}: {checkpoint['allocated']:.3f} GB (+{diff:.3f} GB)")


def _get_tensor_variable_names():
    """
    Get variable names corresponding to tensors.
    
    Returns:
        dict: Mapping from tensor ID to variable name.
    """
    var_names = {}
    
    # Get variables from the current frame and parent frames.
    frame = inspect.currentframe()
    try:
        # Traverse the call stack.
        for i in range(10):  # Inspect up to 10 stack frames.
            if frame is None:
                break
            
            # Check local variables.
            if frame.f_locals:
                for name, obj in frame.f_locals.items():
                    if torch.is_tensor(obj) and obj.is_cuda:
                        var_names[id(obj)] = f"{name}(local)"
                    elif isinstance(obj, (list, tuple)):
                        # Handle nested structures such as KV cache.
                        _extract_nested_tensor_names(obj, name, var_names)
            
            # Check global variables.
            if frame.f_globals:
                for name, obj in frame.f_globals.items():
                    if torch.is_tensor(obj) and obj.is_cuda:
                        var_names[id(obj)] = f"{name}(global)"
                    elif isinstance(obj, (list, tuple)):
                        _extract_nested_tensor_names(obj, name, var_names)
            
            frame = frame.f_back
    finally:
        del frame
    
    return var_names


def _extract_nested_tensor_names(obj, base_name, var_names, max_depth=3, current_depth=0):
    """
    Extract tensor names from nested structures.
    
    Args:
        obj: Object to inspect.
        base_name (str): Base variable name.
        var_names (dict): Variable-name mapping dictionary.
        max_depth (int): Maximum recursion depth.
        current_depth (int): Current recursion depth.
    """
    if current_depth >= max_depth:
        return
    
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            if torch.is_tensor(item) and item.is_cuda:
                var_names[id(item)] = f"{base_name}[{i}]"
            elif isinstance(item, (list, tuple)):
                _extract_nested_tensor_names(item, f"{base_name}[{i}]", var_names, max_depth, current_depth + 1)
    elif hasattr(obj, '__dict__'):
        # Handle object attributes.
        for attr_name, attr_value in obj.__dict__.items():
            if torch.is_tensor(attr_value) and attr_value.is_cuda:
                var_names[id(attr_value)] = f"{base_name}.{attr_name}"
            elif isinstance(attr_value, (list, tuple)):
                _extract_nested_tensor_names(attr_value, f"{base_name}.{attr_name}", var_names, max_depth, current_depth + 1)


def print_all_gpu_tensors_with_names():
    """
    Print all GPU tensors and their variable names.
    """
    print("\n=== 所有GPU张量及变量名 ===")
    tensors_info, total_memory = get_all_gpu_tensors(include_var_names=True)
    
    if not tensors_info:
        print("没有找到GPU张量")
        return
    
    # Sort by memory usage.
    sorted_tensors = sorted(tensors_info.items(), key=lambda x: x[1]['total_size'], reverse=True)
    
    total_size_gb = 0
    for key, info in sorted_tensors:
        size_mb = info['total_size'] / 1024**2
        total_size_gb += info['total_size'] / 1024**3
        
        if size_mb > 0.1:  # Only show tensors larger than 0.1 MB.
            var_names_str = ", ".join(info['var_names'][:3])  # Show up to 3 variable names.
            if len(info['var_names']) > 3:
                var_names_str += f"... (+{len(info['var_names'])-3} more)"
            
            print(f"  {info['shape']} ({info['dtype']}) x{info['count']}: {size_mb:.2f} MB")
            print(f"    变量: {var_names_str}")
    
    print(f"\n总计GPU张量显存: {total_size_gb:.3f} GB")


# Usage example.
if __name__ == "__main__":
    print("GPU 显存分析工具")
    print_gpu_memory_usage("测试")
    
    # Create some test tensors.
    test_tensor1 = torch.randn(1000, 1000, device='cuda')
    test_tensor2 = torch.randn(500, 500, device='cuda')
    kv_cache = [(torch.randn(1, 8, 100, 64, device='cuda'), torch.randn(1, 8, 100, 64, device='cuda'))]
    
    print_gpu_memory_usage("创建测试张量后")
    
    # Show all tensors and their variable names.
    print_all_gpu_tensors_with_names()
    
    analyze_specific_variables(test_tensor1=test_tensor1, test_tensor2=test_tensor2, kv_cache=kv_cache)
    
    # Clean up.
    del test_tensor1, test_tensor2, kv_cache
    clear_gpu_cache()
    print_gpu_memory_usage("清理后")
