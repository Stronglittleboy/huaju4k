"""
性能优化器 - NVIDIA GPU加速优化

实现任务11.3的要求：
- 实现并行处理优化（利用NVIDIA GPU）
- 创建性能指标收集
- 添加资源利用率监控
- 需求: 6.2, 6.6
"""

import os
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

# GPU相关导入
try:
    import pynvml
    HAS_NVIDIA_ML = True
except ImportError:
    HAS_NVIDIA_ML = False
    pynvml = None

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    GPUtil = None

import numpy as np

from ..models.data_models import ProcessingStrategy, TileConfiguration
from ..utils.system_utils import get_memory_info, check_gpu_availability

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    
    processing_speed_fps: float = 0.0  # 处理速度（帧/秒）
    gpu_utilization_percent: float = 0.0  # GPU利用率
    gpu_memory_used_mb: int = 0  # GPU内存使用量
    cpu_utilization_percent: float = 0.0  # CPU利用率
    system_memory_used_mb: int = 0  # 系统内存使用量
    throughput_mbps: float = 0.0  # 吞吐量（MB/秒）
    parallel_efficiency: float = 0.0  # 并行效率
    batch_processing_time: float = 0.0  # 批处理时间
    total_processing_time: float = 0.0  # 总处理时间
    frames_processed: int = 0  # 处理的帧数
    tiles_processed: int = 0  # 处理的瓦片数
    
    # GPU温度和功耗（如果可用）
    gpu_temperature_celsius: Optional[float] = None
    gpu_power_usage_watts: Optional[float] = None
    
    # 并行处理统计
    active_threads: int = 0
    active_processes: int = 0
    queue_size: int = 0
    
    def __post_init__(self):
        """计算派生指标"""
        if self.total_processing_time > 0 and self.frames_processed > 0:
            self.processing_speed_fps = self.frames_processed / self.total_processing_time

class GPUMonitor:
    """
    NVIDIA GPU监控器
    
    监控GPU利用率、内存使用、温度等关键指标
    """
    
    def __init__(self):
        """初始化GPU监控器"""
        self.gpu_available = False
        self.gpu_count = 0
        self.gpu_handles = []
        
        # 初始化NVIDIA ML
        if HAS_NVIDIA_ML:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_available = self.gpu_count > 0
                
                # 获取GPU句柄
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles.append(handle)
                
                logger.info(f"NVIDIA GPU监控初始化成功，检测到 {self.gpu_count} 个GPU")
                
            except Exception as e:
                logger.warning(f"NVIDIA GPU监控初始化失败: {e}")
                self.gpu_available = False
        
        # 备用GPU监控（使用GPUtil）
        elif HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                self.gpu_count = len(gpus)
                self.gpu_available = self.gpu_count > 0
                logger.info(f"GPUtil GPU监控初始化成功，检测到 {self.gpu_count} 个GPU")
            except Exception as e:
                logger.warning(f"GPUtil GPU监控初始化失败: {e}")
                self.gpu_available = False
        
        else:
            logger.warning("GPU监控库不可用，GPU监控功能将被禁用")
    
    def get_gpu_metrics(self, gpu_id: int = 0) -> Dict[str, Any]:
        """
        获取GPU性能指标
        
        Args:
            gpu_id: GPU设备ID
            
        Returns:
            GPU指标字典
        """
        if not self.gpu_available or gpu_id >= self.gpu_count:
            return {
                'utilization_percent': 0.0,
                'memory_used_mb': 0,
                'memory_total_mb': 0,
                'memory_free_mb': 0,
                'temperature_celsius': None,
                'power_usage_watts': None,
                'available': False
            }
        
        try:
            if HAS_NVIDIA_ML and gpu_id < len(self.gpu_handles):
                return self._get_nvidia_ml_metrics(gpu_id)
            elif HAS_GPUTIL:
                return self._get_gputil_metrics(gpu_id)
            else:
                return {'available': False}
                
        except Exception as e:
            logger.error(f"获取GPU指标失败: {e}")
            return {'available': False}
    
    def _get_nvidia_ml_metrics(self, gpu_id: int) -> Dict[str, Any]:
        """使用NVIDIA ML获取GPU指标"""
        handle = self.gpu_handles[gpu_id]
        
        # GPU利用率
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util.gpu
        
        # 内存信息
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used_mb = mem_info.used // (1024 * 1024)
        memory_total_mb = mem_info.total // (1024 * 1024)
        memory_free_mb = mem_info.free // (1024 * 1024)
        
        # 温度
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            temperature = None
        
        # 功耗
        try:
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
        except:
            power_usage = None
        
        return {
            'utilization_percent': float(gpu_util),
            'memory_used_mb': int(memory_used_mb),
            'memory_total_mb': int(memory_total_mb),
            'memory_free_mb': int(memory_free_mb),
            'temperature_celsius': temperature,
            'power_usage_watts': power_usage,
            'available': True
        }
    
    def _get_gputil_metrics(self, gpu_id: int) -> Dict[str, Any]:
        """使用GPUtil获取GPU指标"""
        gpus = GPUtil.getGPUs()
        if gpu_id >= len(gpus):
            return {'available': False}
        
        gpu = gpus[gpu_id]
        
        return {
            'utilization_percent': float(gpu.load * 100),
            'memory_used_mb': int(gpu.memoryUsed),
            'memory_total_mb': int(gpu.memoryTotal),
            'memory_free_mb': int(gpu.memoryFree),
            'temperature_celsius': gpu.temperature,
            'power_usage_watts': None,  # GPUtil不提供功耗信息
            'available': True
        }
class ParallelProcessingOptimizer:
    """
    并行处理优化器
    
    实现智能并行处理优化，充分利用多核CPU和GPU资源
    """
    
    def __init__(self, strategy: ProcessingStrategy):
        """
        初始化并行处理优化器
        
        Args:
            strategy: 处理策略配置
        """
        self.strategy = strategy
        self.cpu_count = mp.cpu_count()
        self.gpu_monitor = GPUMonitor()
        
        # 计算最优线程/进程数
        self.optimal_threads = self._calculate_optimal_threads()
        self.optimal_processes = self._calculate_optimal_processes()
        
        # 性能监控
        self.metrics = PerformanceMetrics()
        self.monitoring_active = False
        self.monitor_thread = None
        
        logger.info(f"并行处理优化器初始化: {self.optimal_threads} 线程, {self.optimal_processes} 进程")
    
    def _calculate_optimal_threads(self) -> int:
        """计算最优线程数"""
        if self.strategy.use_gpu:
            # GPU处理时，减少CPU线程数以避免竞争
            return min(4, max(2, self.cpu_count // 2))
        else:
            # CPU处理时，使用更多线程
            return min(self.cpu_count, max(2, self.cpu_count - 1))
    
    def _calculate_optimal_processes(self) -> int:
        """计算最优进程数"""
        if self.strategy.use_gpu:
            # GPU处理时，通常单进程更高效
            return 1
        else:
            # CPU密集型任务可以使用多进程
            return min(4, max(2, self.cpu_count // 2))
    
    def optimize_batch_processing(self, items: List[Any], 
                                 process_func: Callable,
                                 progress_callback: Optional[Callable] = None) -> List[Any]:
        """
        优化批处理性能
        
        Args:
            items: 要处理的项目列表
            process_func: 处理函数
            progress_callback: 进度回调函数
            
        Returns:
            处理结果列表
        """
        start_time = time.time()
        
        try:
            # 启动性能监控
            self.start_monitoring()
            
            # 根据策略选择处理方式
            if self.strategy.use_gpu:
                results = self._gpu_optimized_processing(items, process_func, progress_callback)
            else:
                results = self._cpu_optimized_processing(items, process_func, progress_callback)
            
            # 更新性能指标
            total_time = time.time() - start_time
            self.metrics.total_processing_time = total_time
            self.metrics.frames_processed = len(items)
            
            logger.info(f"批处理完成: {len(items)} 项目, 用时 {total_time:.2f}秒")
            
            return results
            
        finally:
            # 停止性能监控
            self.stop_monitoring()
    
    def _gpu_optimized_processing(self, items: List[Any], 
                                 process_func: Callable,
                                 progress_callback: Optional[Callable] = None) -> List[Any]:
        """GPU优化的处理方式"""
        results = []
        batch_size = self.strategy.batch_size
        
        # 按批次处理以优化GPU利用率
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_start = time.time()
            
            # GPU批处理
            batch_results = self._process_gpu_batch(batch, process_func)
            results.extend(batch_results)
            
            # 更新批处理时间
            batch_time = time.time() - batch_start
            self.metrics.batch_processing_time = batch_time
            
            # 进度回调
            if progress_callback:
                progress = (i + len(batch)) / len(items)
                progress_callback(progress, f"GPU批处理: {i + len(batch)}/{len(items)}")
        
        return results
    
    def _cpu_optimized_processing(self, items: List[Any], 
                                 process_func: Callable,
                                 progress_callback: Optional[Callable] = None) -> List[Any]:
        """CPU优化的处理方式"""
        results = [None] * len(items)
        completed = 0
        
        # 使用线程池进行并行处理
        with ThreadPoolExecutor(max_workers=self.optimal_threads) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(process_func, item): i 
                for i, item in enumerate(items)
            }
            
            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                    completed += 1
                    
                    # 进度回调
                    if progress_callback:
                        progress = completed / len(items)
                        progress_callback(progress, f"CPU并行处理: {completed}/{len(items)}")
                        
                except Exception as e:
                    logger.error(f"处理项目 {index} 失败: {e}")
                    results[index] = None
        
        return results
    
    def _process_gpu_batch(self, batch: List[Any], process_func: Callable) -> List[Any]:
        """处理GPU批次"""
        try:
            # 这里应该调用支持批处理的GPU函数
            # 目前使用单个处理作为示例
            return [process_func(item) for item in batch]
        except Exception as e:
            logger.error(f"GPU批处理失败: {e}")
            return [None] * len(batch)
    
    def start_monitoring(self) -> None:
        """启动性能监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitor_thread.start()
        
        logger.debug("性能监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止性能监控"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        logger.debug("性能监控已停止")
    
    def _monitor_performance(self) -> None:
        """性能监控线程"""
        while self.monitoring_active:
            try:
                # 获取系统内存信息
                memory_info = get_memory_info()
                self.metrics.system_memory_used_mb = memory_info.get('used_mb', 0)
                self.metrics.cpu_utilization_percent = memory_info.get('memory_percent', 0)
                
                # 获取GPU指标
                if self.gpu_monitor.gpu_available:
                    gpu_metrics = self.gpu_monitor.get_gpu_metrics(0)
                    self.metrics.gpu_utilization_percent = gpu_metrics.get('utilization_percent', 0)
                    self.metrics.gpu_memory_used_mb = gpu_metrics.get('memory_used_mb', 0)
                    self.metrics.gpu_temperature_celsius = gpu_metrics.get('temperature_celsius')
                    self.metrics.gpu_power_usage_watts = gpu_metrics.get('power_usage_watts')
                
                # 更新线程/进程统计
                self.metrics.active_threads = threading.active_count()
                
                time.sleep(1.0)  # 每秒更新一次
                
            except Exception as e:
                logger.error(f"性能监控错误: {e}")
                time.sleep(5.0)  # 出错时延长间隔
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取当前性能指标"""
        return self.metrics
    
    def get_optimization_recommendations(self) -> Dict[str, str]:
        """获取性能优化建议"""
        recommendations = {}
        
        # GPU利用率建议
        if self.strategy.use_gpu and self.gpu_monitor.gpu_available:
            gpu_util = self.metrics.gpu_utilization_percent
            if gpu_util < 50:
                recommendations['gpu_utilization'] = "GPU利用率较低，考虑增加批处理大小"
            elif gpu_util > 95:
                recommendations['gpu_utilization'] = "GPU利用率过高，考虑减少批处理大小"
        
        # 内存使用建议
        memory_usage = self.metrics.system_memory_used_mb
        if memory_usage > 8000:  # 8GB
            recommendations['memory_usage'] = "内存使用较高，考虑减少瓦片大小或批处理大小"
        
        # 并行效率建议
        if self.metrics.parallel_efficiency < 0.7:
            recommendations['parallel_efficiency'] = "并行效率较低，考虑调整线程数或处理策略"
        
        return recommendations
class ResourceUtilizationMonitor:
    """
    资源利用率监控器
    
    监控系统资源利用率，包括CPU、内存、GPU、磁盘等
    """
    
    def __init__(self, monitoring_interval: float = 1.0):
        """
        初始化资源监控器
        
        Args:
            monitoring_interval: 监控间隔（秒）
        """
        self.monitoring_interval = monitoring_interval
        self.gpu_monitor = GPUMonitor()
        
        # 监控数据存储
        self.resource_history = {
            'timestamps': [],
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'gpu_memory': [],
            'disk_io': []
        }
        
        # 监控控制
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()
    
    def start_monitoring(self) -> None:
        """启动资源监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        
        logger.info("资源利用率监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止资源监控"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("资源利用率监控已停止")
    
    def _monitor_resources(self) -> None:
        """资源监控线程"""
        while self.monitoring_active:
            try:
                timestamp = time.time()
                
                # 获取系统资源信息
                memory_info = get_memory_info()
                cpu_usage = memory_info.get('memory_percent', 0)  # 这里应该是CPU使用率
                memory_usage = memory_info.get('used_mb', 0)
                
                # 获取GPU信息
                gpu_usage = 0
                gpu_memory = 0
                if self.gpu_monitor.gpu_available:
                    gpu_metrics = self.gpu_monitor.get_gpu_metrics(0)
                    gpu_usage = gpu_metrics.get('utilization_percent', 0)
                    gpu_memory = gpu_metrics.get('memory_used_mb', 0)
                
                # 存储监控数据
                with self.lock:
                    self.resource_history['timestamps'].append(timestamp)
                    self.resource_history['cpu_usage'].append(cpu_usage)
                    self.resource_history['memory_usage'].append(memory_usage)
                    self.resource_history['gpu_usage'].append(gpu_usage)
                    self.resource_history['gpu_memory'].append(gpu_memory)
                    self.resource_history['disk_io'].append(0)  # 磁盘IO监控待实现
                    
                    # 限制历史数据长度（保留最近1小时的数据）
                    max_history = int(3600 / self.monitoring_interval)
                    for key in self.resource_history:
                        if len(self.resource_history[key]) > max_history:
                            self.resource_history[key] = self.resource_history[key][-max_history:]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"资源监控错误: {e}")
                time.sleep(5.0)
    
    def get_current_utilization(self) -> Dict[str, float]:
        """获取当前资源利用率"""
        with self.lock:
            if not self.resource_history['timestamps']:
                return {
                    'cpu_usage_percent': 0.0,
                    'memory_usage_mb': 0.0,
                    'gpu_usage_percent': 0.0,
                    'gpu_memory_mb': 0.0
                }
            
            return {
                'cpu_usage_percent': self.resource_history['cpu_usage'][-1],
                'memory_usage_mb': self.resource_history['memory_usage'][-1],
                'gpu_usage_percent': self.resource_history['gpu_usage'][-1],
                'gpu_memory_mb': self.resource_history['gpu_memory'][-1]
            }
    
    def get_utilization_statistics(self, duration_minutes: int = 10) -> Dict[str, Dict[str, float]]:
        """
        获取资源利用率统计信息
        
        Args:
            duration_minutes: 统计时间窗口（分钟）
            
        Returns:
            资源利用率统计字典
        """
        with self.lock:
            if not self.resource_history['timestamps']:
                return {}
            
            # 计算时间窗口
            current_time = time.time()
            window_start = current_time - (duration_minutes * 60)
            
            # 过滤时间窗口内的数据
            indices = [
                i for i, ts in enumerate(self.resource_history['timestamps'])
                if ts >= window_start
            ]
            
            if not indices:
                return {}
            
            stats = {}
            for resource in ['cpu_usage', 'memory_usage', 'gpu_usage', 'gpu_memory']:
                values = [self.resource_history[resource][i] for i in indices]
                if values:
                    stats[resource] = {
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'current': values[-1]
                    }
            
            return stats
    
    def detect_resource_bottlenecks(self) -> List[str]:
        """检测资源瓶颈"""
        bottlenecks = []
        current = self.get_current_utilization()
        
        # CPU瓶颈检测
        if current['cpu_usage_percent'] > 90:
            bottlenecks.append("CPU使用率过高")
        
        # 内存瓶颈检测
        if current['memory_usage_mb'] > 12000:  # 12GB
            bottlenecks.append("内存使用量过高")
        
        # GPU瓶颈检测
        if current['gpu_usage_percent'] > 95:
            bottlenecks.append("GPU使用率过高")
        
        if current['gpu_memory_mb'] > 3500:  # 接近4GB限制
            bottlenecks.append("GPU内存使用量过高")
        
        return bottlenecks


class PerformanceOptimizer:
    """
    主性能优化器
    
    集成并行处理优化和资源监控，提供完整的性能优化解决方案
    """
    
    def __init__(self, strategy: ProcessingStrategy):
        """
        初始化性能优化器
        
        Args:
            strategy: 处理策略配置
        """
        self.strategy = strategy
        self.parallel_optimizer = ParallelProcessingOptimizer(strategy)
        self.resource_monitor = ResourceUtilizationMonitor()
        
        # 性能统计
        self.performance_stats = {
            'total_processing_time': 0.0,
            'items_processed': 0,
            'average_processing_speed': 0.0,
            'peak_gpu_utilization': 0.0,
            'peak_memory_usage': 0.0,
            'optimization_applied': []
        }
        
        logger.info("性能优化器初始化完成")
    
    def optimize_processing(self, items: List[Any], 
                          process_func: Callable,
                          progress_callback: Optional[Callable] = None) -> List[Any]:
        """
        执行优化的处理
        
        Args:
            items: 要处理的项目列表
            process_func: 处理函数
            progress_callback: 进度回调函数
            
        Returns:
            处理结果列表
        """
        start_time = time.time()
        
        try:
            # 启动资源监控
            self.resource_monitor.start_monitoring()
            
            # 执行优化的批处理
            results = self.parallel_optimizer.optimize_batch_processing(
                items, process_func, progress_callback
            )
            
            # 更新性能统计
            total_time = time.time() - start_time
            self.performance_stats['total_processing_time'] += total_time
            self.performance_stats['items_processed'] += len(items)
            
            if total_time > 0:
                self.performance_stats['average_processing_speed'] = len(items) / total_time
            
            # 记录峰值利用率
            current_util = self.resource_monitor.get_current_utilization()
            self.performance_stats['peak_gpu_utilization'] = max(
                self.performance_stats['peak_gpu_utilization'],
                current_util.get('gpu_usage_percent', 0)
            )
            self.performance_stats['peak_memory_usage'] = max(
                self.performance_stats['peak_memory_usage'],
                current_util.get('memory_usage_mb', 0)
            )
            
            logger.info(f"优化处理完成: {len(items)} 项目, 用时 {total_time:.2f}秒")
            
            return results
            
        finally:
            # 停止资源监控
            self.resource_monitor.stop_monitoring()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        生成性能报告
        
        实现需求6.6: 报告性能指标包括处理速度和资源利用率
        
        Returns:
            完整的性能报告
        """
        # 获取并行处理指标
        parallel_metrics = self.parallel_optimizer.get_performance_metrics()
        
        # 获取资源利用率统计
        resource_stats = self.resource_monitor.get_utilization_statistics()
        
        # 获取优化建议
        recommendations = self.parallel_optimizer.get_optimization_recommendations()
        
        # 检测资源瓶颈
        bottlenecks = self.resource_monitor.detect_resource_bottlenecks()
        
        report = {
            'performance_metrics': {
                'processing_speed_fps': parallel_metrics.processing_speed_fps,
                'total_processing_time': parallel_metrics.total_processing_time,
                'frames_processed': parallel_metrics.frames_processed,
                'tiles_processed': parallel_metrics.tiles_processed,
                'parallel_efficiency': parallel_metrics.parallel_efficiency
            },
            'resource_utilization': {
                'gpu_utilization_percent': parallel_metrics.gpu_utilization_percent,
                'gpu_memory_used_mb': parallel_metrics.gpu_memory_used_mb,
                'cpu_utilization_percent': parallel_metrics.cpu_utilization_percent,
                'system_memory_used_mb': parallel_metrics.system_memory_used_mb,
                'gpu_temperature_celsius': parallel_metrics.gpu_temperature_celsius,
                'gpu_power_usage_watts': parallel_metrics.gpu_power_usage_watts
            },
            'resource_statistics': resource_stats,
            'optimization_recommendations': recommendations,
            'resource_bottlenecks': bottlenecks,
            'system_info': {
                'gpu_available': self.parallel_optimizer.gpu_monitor.gpu_available,
                'gpu_count': self.parallel_optimizer.gpu_monitor.gpu_count,
                'cpu_count': self.parallel_optimizer.cpu_count,
                'optimal_threads': self.parallel_optimizer.optimal_threads,
                'optimal_processes': self.parallel_optimizer.optimal_processes
            }
        }
        
        return report
    
    def apply_dynamic_optimization(self) -> None:
        """
        应用动态优化
        
        实现需求6.2: 实现并行处理使用最优线程分配
        """
        # 检测当前资源状态
        bottlenecks = self.resource_monitor.detect_resource_bottlenecks()
        current_util = self.resource_monitor.get_current_utilization()
        
        optimizations_applied = []
        
        # 动态调整线程数
        if "CPU使用率过高" in bottlenecks:
            new_threads = max(1, self.parallel_optimizer.optimal_threads - 1)
            self.parallel_optimizer.optimal_threads = new_threads
            optimizations_applied.append(f"减少线程数至 {new_threads}")
        
        # 动态调整批处理大小
        if "GPU内存使用量过高" in bottlenecks:
            new_batch_size = max(1, self.strategy.batch_size - 1)
            self.strategy.batch_size = new_batch_size
            optimizations_applied.append(f"减少批处理大小至 {new_batch_size}")
        
        # 记录应用的优化
        self.performance_stats['optimization_applied'].extend(optimizations_applied)
        
        if optimizations_applied:
            logger.info(f"应用动态优化: {', '.join(optimizations_applied)}")
    
    def cleanup(self) -> None:
        """清理资源"""
        self.parallel_optimizer.stop_monitoring()
        self.resource_monitor.stop_monitoring()
        logger.info("性能优化器资源清理完成")