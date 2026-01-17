"""
System utilities for huaju4k video enhancement.

This module provides system information gathering, hardware detection,
and dependency checking utilities.
"""

import os
import platform
import subprocess
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available - system monitoring will be limited")


def get_system_info() -> Dict[str, any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dictionary containing system information
    """
    try:
        info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
        }
        
        if HAS_PSUTIL:
            info.update({
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'total_memory_mb': int(psutil.virtual_memory().total / (1024 * 1024)),
                'available_memory_mb': int(psutil.virtual_memory().available / (1024 * 1024)),
            })
        else:
            # Fallback values when psutil is not available
            info.update({
                'cpu_cores': os.cpu_count() or 1,
                'cpu_threads': os.cpu_count() or 1,
                'total_memory_mb': 0,  # Unknown
                'available_memory_mb': 0,  # Unknown
            })
        
        # Add GPU information if available
        gpu_info = check_gpu_availability()
        info.update(gpu_info)
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {'error': str(e)}


def check_gpu_availability() -> Dict[str, any]:
    """
    Check GPU availability and capabilities.
    
    Returns:
        Dictionary containing GPU information
    """
    gpu_info = {
        'gpu_available': False,
        'gpu_count': 0,
        'gpu_memory_mb': 0,
        'gpu_name': '',
        'cuda_available': False,
        'opencl_available': False
    }
    
    try:
        # Check NVIDIA GPU with nvidia-ml-py
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            
            if gpu_count > 0:
                gpu_info['gpu_available'] = True
                gpu_info['gpu_count'] = gpu_count
                
                # Get info for first GPU
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_info['gpu_name'] = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_info['gpu_memory_mb'] = int(mem_info.total / (1024 * 1024))
                
        except ImportError:
            logger.debug("pynvml not available, trying alternative GPU detection")
        except Exception as e:
            logger.debug(f"NVIDIA GPU detection failed: {e}")
        
        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                if not gpu_info['gpu_available']:  # Fallback GPU info from PyTorch
                    gpu_info['gpu_available'] = True
                    gpu_info['gpu_count'] = torch.cuda.device_count()
                    gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
                    gpu_info['gpu_memory_mb'] = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        except ImportError:
            logger.debug("PyTorch not available for CUDA detection")
        except Exception as e:
            logger.debug(f"CUDA detection failed: {e}")
        
        # Check OpenCL availability
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            if platforms:
                gpu_info['opencl_available'] = True
        except ImportError:
            logger.debug("PyOpenCL not available")
        except Exception as e:
            logger.debug(f"OpenCL detection failed: {e}")
        
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")
    
    return gpu_info


def get_memory_info() -> Dict[str, int]:
    """
    Get detailed memory information.
    
    Returns:
        Dictionary containing memory information in MB
    """
    if not HAS_PSUTIL:
        logger.warning("psutil not available - returning empty memory info")
        return {}
    
    try:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total_mb': int(memory.total / (1024 * 1024)),
            'available_mb': int(memory.available / (1024 * 1024)),
            'used_mb': int(memory.used / (1024 * 1024)),
            'free_mb': int(memory.free / (1024 * 1024)),
            'cached_mb': int(getattr(memory, 'cached', 0) / (1024 * 1024)),
            'swap_total_mb': int(swap.total / (1024 * 1024)),
            'swap_used_mb': int(swap.used / (1024 * 1024)),
            'swap_free_mb': int(swap.free / (1024 * 1024)),
            'memory_percent': memory.percent
        }
        
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
        return {}


def check_dependencies() -> Dict[str, bool]:
    """
    Check availability of required dependencies.
    
    Returns:
        Dictionary mapping dependency names to availability status
    """
    dependencies = {
        'ffmpeg': False,
        'opencv': False,
        'numpy': False,
        'pillow': False,
        'torch': False,
        'librosa': False,
        'yaml': False
    }
    
    # Check FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        dependencies['ffmpeg'] = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # Check Python packages
    python_packages = {
        'opencv': ['cv2'],
        'numpy': ['numpy'],
        'pillow': ['PIL'],
        'torch': ['torch'],
        'librosa': ['librosa'],
        'yaml': ['yaml']
    }
    
    for dep_name, module_names in python_packages.items():
        try:
            for module_name in module_names:
                __import__(module_name)
            dependencies[dep_name] = True
        except ImportError:
            pass
    
    return dependencies


def get_cpu_info() -> Dict[str, any]:
    """
    Get detailed CPU information.
    
    Returns:
        Dictionary containing CPU information
    """
    if not HAS_PSUTIL:
        return {
            'physical_cores': os.cpu_count() or 1,
            'logical_cores': os.cpu_count() or 1,
            'architecture': platform.machine(),
            'processor_name': platform.processor()
        }
    
    try:
        cpu_freq = psutil.cpu_freq()
        
        info = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'max_frequency_mhz': cpu_freq.max if cpu_freq else None,
            'current_frequency_mhz': cpu_freq.current if cpu_freq else None,
            'cpu_usage_percent': psutil.cpu_percent(interval=1),
            'architecture': platform.machine(),
            'processor_name': platform.processor()
        }
        
        # Get per-core usage
        per_core_usage = psutil.cpu_percent(percpu=True, interval=1)
        info['per_core_usage'] = per_core_usage
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting CPU info: {e}")
        return {}


def check_disk_space(path: str) -> Dict[str, int]:
    """
    Check disk space for given path.
    
    Args:
        path: Path to check disk space for
        
    Returns:
        Dictionary containing disk space information in MB
    """
    if not HAS_PSUTIL:
        logger.warning("psutil not available - cannot check disk space")
        return {}
    
    try:
        disk_usage = psutil.disk_usage(path)
        
        return {
            'total_mb': int(disk_usage.total / (1024 * 1024)),
            'used_mb': int(disk_usage.used / (1024 * 1024)),
            'free_mb': int(disk_usage.free / (1024 * 1024)),
            'usage_percent': (disk_usage.used / disk_usage.total) * 100
        }
        
    except Exception as e:
        logger.error(f"Error checking disk space for {path}: {e}")
        return {}


def is_admin() -> bool:
    """
    Check if current process has administrator privileges.
    
    Returns:
        True if running with admin privileges
    """
    try:
        if os.name == 'nt':  # Windows
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        else:  # Unix-like systems
            return os.geteuid() == 0
    except Exception:
        return False


def get_process_info() -> Dict[str, any]:
    """
    Get information about current process.
    
    Returns:
        Dictionary containing process information
    """
    if not HAS_PSUTIL:
        return {
            'pid': os.getpid(),
            'is_admin': is_admin()
        }
    
    try:
        process = psutil.Process()
        
        return {
            'pid': process.pid,
            'memory_mb': int(process.memory_info().rss / (1024 * 1024)),
            'cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads(),
            'create_time': process.create_time(),
            'status': process.status(),
            'is_admin': is_admin()
        }
        
    except Exception as e:
        logger.error(f"Error getting process info: {e}")
        return {}


def estimate_processing_capability() -> Dict[str, any]:
    """
    Estimate system's video processing capability.
    
    Returns:
        Dictionary containing capability estimates
    """
    try:
        system_info = get_system_info()
        memory_info = get_memory_info()
        cpu_info = get_cpu_info()
        
        # Estimate based on system specs
        capability = {
            'max_video_resolution': '1080p',  # Conservative default
            'recommended_tile_size': 512,
            'max_concurrent_tiles': 1,
            'estimated_speed_multiplier': 1.0
        }
        
        # Adjust based on memory
        available_memory = memory_info.get('available_mb', 0)
        if available_memory >= 16000:  # 16GB+
            capability['max_video_resolution'] = '4K'
            capability['recommended_tile_size'] = 1024
            capability['max_concurrent_tiles'] = 4
        elif available_memory >= 8000:   # 8GB+
            capability['max_video_resolution'] = '1440p'
            capability['recommended_tile_size'] = 768
            capability['max_concurrent_tiles'] = 2
        
        # Adjust based on CPU
        cpu_cores = cpu_info.get('physical_cores', 1)
        if cpu_cores >= 8:
            capability['estimated_speed_multiplier'] *= 2.0
        elif cpu_cores >= 4:
            capability['estimated_speed_multiplier'] *= 1.5
        
        # Adjust based on GPU
        if system_info.get('gpu_available', False):
            gpu_memory = system_info.get('gpu_memory_mb', 0)
            if gpu_memory >= 8000:  # 8GB+ GPU
                capability['estimated_speed_multiplier'] *= 3.0
                capability['max_concurrent_tiles'] *= 2
            elif gpu_memory >= 4000:  # 4GB+ GPU
                capability['estimated_speed_multiplier'] *= 2.0
        
        return capability
        
    except Exception as e:
        logger.error(f"Error estimating processing capability: {e}")
        return {}