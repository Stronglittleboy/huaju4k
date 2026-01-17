"""
Performance Manager Implementation

Provides hardware optimization, resource allocation, and performance monitoring
for the video enhancement toolkit.
"""

import os
import platform
import subprocess
import psutil
import multiprocessing as mp
import threading
import time
import gc
import weakref
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, asdict
from collections import deque
from contextlib import contextmanager
import cv2

from ..core.interfaces import IPerformanceManager
from ..core.models import PerformanceConfig
from .interfaces import ILogger


@dataclass
class SystemResources:
    """System resource information."""
    cpu_cores: int
    cpu_model: str
    total_ram_gb: float
    available_ram_gb: float
    gpu_model: Optional[str]
    gpu_memory_gb: Optional[float]
    cuda_available: bool
    opencv_cuda_available: bool
    storage_available_gb: float
    os_info: str


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_used_gb: float
    gpu_usage_percent: Optional[float]
    gpu_memory_usage_percent: Optional[float]
    disk_io_read_mb_s: float
    disk_io_write_mb_s: float
    active_threads: int
    timestamp: float


@dataclass
class ResourceAllocation:
    """Resource allocation configuration."""
    cpu_threads: int
    memory_limit_gb: float
    gpu_enabled: bool
    batch_size: int
    tile_size: int
    parallel_processing: bool


@dataclass
class MemoryPool:
    """Memory pool for efficient memory management."""
    pool_id: str
    allocated_mb: float
    used_mb: float
    max_size_mb: float
    allocation_count: int
    last_access: float


class AdaptiveResourceManager:
    """Manages adaptive resource allocation based on workload patterns."""
    
    def __init__(self, logger: ILogger):
        self.logger = logger
        self._workload_history = deque(maxlen=50)
        self._resource_adjustments = {}
        self._performance_baseline = None
    
    def record_workload(self, task_type: str, duration: float, resources_used: Dict[str, Any], success: bool):
        """Record workload performance for adaptive learning."""
        workload_record = {
            "task_type": task_type,
            "duration": duration,
            "resources_used": resources_used,
            "success": success,
            "timestamp": time.time(),
            "efficiency": self._calculate_efficiency(duration, resources_used, success)
        }
        
        self._workload_history.append(workload_record)
        self._update_resource_adjustments(task_type, workload_record)
    
    def get_adaptive_config(self, task_type: str, base_config: PerformanceConfig) -> PerformanceConfig:
        """Get adaptively adjusted configuration based on historical performance."""
        if task_type not in self._resource_adjustments:
            return base_config
        
        adjustments = self._resource_adjustments[task_type]
        
        # Apply learned adjustments
        adjusted_config = PerformanceConfig(
            cpu_threads=max(1, int(base_config.cpu_threads * adjustments.get("thread_multiplier", 1.0))),
            gpu_memory_limit=base_config.gpu_memory_limit * adjustments.get("memory_multiplier", 1.0),
            memory_optimization=base_config.memory_optimization,
            parallel_processing=base_config.parallel_processing
        )
        
        self.logger.log_operation(
            "adaptive_config_applied",
            {
                "task_type": task_type,
                "base_config": asdict(base_config),
                "adjusted_config": asdict(adjusted_config),
                "adjustments": adjustments
            }
        )
        
        return adjusted_config
    
    def _calculate_efficiency(self, duration: float, resources_used: Dict[str, Any], success: bool) -> float:
        """Calculate efficiency score for a workload."""
        if not success:
            return 0.0
        
        # Simple efficiency metric: inverse of duration weighted by resource usage
        cpu_usage = resources_used.get("cpu_usage_percent", 50) / 100.0
        memory_usage = resources_used.get("memory_usage_percent", 50) / 100.0
        
        # Higher resource utilization with shorter duration = higher efficiency
        resource_efficiency = (cpu_usage + memory_usage) / 2.0
        time_efficiency = 1.0 / max(duration, 0.1)  # Avoid division by zero
        
        return resource_efficiency * time_efficiency
    
    def _update_resource_adjustments(self, task_type: str, workload_record: Dict[str, Any]):
        """Update resource adjustment factors based on workload performance."""
        if task_type not in self._resource_adjustments:
            self._resource_adjustments[task_type] = {
                "thread_multiplier": 1.0,
                "memory_multiplier": 1.0,
                "sample_count": 0,
                "avg_efficiency": 0.0
            }
        
        adjustments = self._resource_adjustments[task_type]
        adjustments["sample_count"] += 1
        
        # Update average efficiency
        current_efficiency = workload_record["efficiency"]
        adjustments["avg_efficiency"] = (
            (adjustments["avg_efficiency"] * (adjustments["sample_count"] - 1) + current_efficiency) /
            adjustments["sample_count"]
        )
        
        # Adaptive adjustment logic
        if adjustments["sample_count"] >= 5:  # Need minimum samples
            recent_records = [r for r in self._workload_history if r["task_type"] == task_type][-5:]
            avg_recent_efficiency = sum(r["efficiency"] for r in recent_records) / len(recent_records)
            
            # If recent efficiency is below average, try adjustments
            if avg_recent_efficiency < adjustments["avg_efficiency"] * 0.9:
                # Try reducing thread count if CPU seems to be bottleneck
                avg_cpu_usage = sum(r["resources_used"].get("cpu_usage_percent", 50) for r in recent_records) / len(recent_records)
                if avg_cpu_usage > 90:
                    adjustments["thread_multiplier"] = max(0.5, adjustments["thread_multiplier"] * 0.9)
                
                # Try reducing memory if memory seems to be bottleneck
                avg_memory_usage = sum(r["resources_used"].get("memory_usage_percent", 50) for r in recent_records) / len(recent_records)
                if avg_memory_usage > 85:
                    adjustments["memory_multiplier"] = max(0.5, adjustments["memory_multiplier"] * 0.9)


class MemoryManager:
    """Advanced memory management with pooling and optimization."""
    
    def __init__(self, logger: ILogger):
        self.logger = logger
        self._memory_pools: Dict[str, MemoryPool] = {}
        self._memory_limit_gb = 0.0
        self._current_usage_gb = 0.0
        self._allocation_callbacks: List[Callable] = []
        self._cleanup_threshold = 0.8  # Trigger cleanup at 80% usage
        self._lock = threading.Lock()
    
    def set_memory_limit(self, limit_gb: float):
        """Set global memory limit for the application."""
        self._memory_limit_gb = limit_gb
        self.logger.log_operation("memory_limit_set", {"limit_gb": limit_gb})
    
    def create_memory_pool(self, pool_id: str, max_size_mb: float) -> bool:
        """Create a memory pool for efficient allocation."""
        with self._lock:
            if pool_id in self._memory_pools:
                return False
            
            self._memory_pools[pool_id] = MemoryPool(
                pool_id=pool_id,
                allocated_mb=0.0,
                used_mb=0.0,
                max_size_mb=max_size_mb,
                allocation_count=0,
                last_access=time.time()
            )
            
            self.logger.log_operation(
                "memory_pool_created",
                {"pool_id": pool_id, "max_size_mb": max_size_mb}
            )
            return True
    
    @contextmanager
    def allocate_memory(self, pool_id: str, size_mb: float):
        """Context manager for safe memory allocation."""
        allocated = False
        try:
            with self._lock:
                if pool_id not in self._memory_pools:
                    raise ValueError(f"Memory pool {pool_id} does not exist")
                
                pool = self._memory_pools[pool_id]
                
                # Check if allocation would exceed pool limit
                if pool.used_mb + size_mb > pool.max_size_mb:
                    # Try to free unused memory first
                    self._cleanup_pool(pool_id)
                    
                    # Check again after cleanup
                    if pool.used_mb + size_mb > pool.max_size_mb:
                        raise MemoryError(f"Cannot allocate {size_mb}MB in pool {pool_id}")
                
                # Check global memory limit
                total_requested = self._current_usage_gb + (size_mb / 1024)
                if total_requested > self._memory_limit_gb:
                    self._trigger_global_cleanup()
                    
                    # Check again after global cleanup
                    total_requested = self._current_usage_gb + (size_mb / 1024)
                    if total_requested > self._memory_limit_gb:
                        raise MemoryError(f"Global memory limit exceeded: {total_requested:.2f}GB > {self._memory_limit_gb:.2f}GB")
                
                # Perform allocation
                pool.used_mb += size_mb
                pool.allocation_count += 1
                pool.last_access = time.time()
                self._current_usage_gb += size_mb / 1024
                allocated = True
                
                self.logger.log_operation(
                    "memory_allocated",
                    {
                        "pool_id": pool_id,
                        "size_mb": size_mb,
                        "pool_usage_mb": pool.used_mb,
                        "global_usage_gb": self._current_usage_gb
                    }
                )
            
            yield size_mb
            
        finally:
            if allocated:
                with self._lock:
                    pool = self._memory_pools[pool_id]
                    pool.used_mb = max(0, pool.used_mb - size_mb)
                    self._current_usage_gb = max(0, self._current_usage_gb - size_mb / 1024)
                    
                    self.logger.log_operation(
                        "memory_deallocated",
                        {
                            "pool_id": pool_id,
                            "size_mb": size_mb,
                            "pool_usage_mb": pool.used_mb,
                            "global_usage_gb": self._current_usage_gb
                        }
                    )
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        with self._lock:
            pool_stats = {}
            for pool_id, pool in self._memory_pools.items():
                pool_stats[pool_id] = {
                    "used_mb": pool.used_mb,
                    "max_size_mb": pool.max_size_mb,
                    "usage_percent": (pool.used_mb / pool.max_size_mb) * 100 if pool.max_size_mb > 0 else 0,
                    "allocation_count": pool.allocation_count,
                    "last_access": pool.last_access
                }
            
            return {
                "global_usage_gb": self._current_usage_gb,
                "global_limit_gb": self._memory_limit_gb,
                "global_usage_percent": (self._current_usage_gb / self._memory_limit_gb) * 100 if self._memory_limit_gb > 0 else 0,
                "pools": pool_stats,
                "system_memory": self._get_system_memory_info()
            }
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage by cleaning up unused allocations."""
        with self._lock:
            initial_usage = self._current_usage_gb
            cleanup_results = {}
            
            # Clean up each pool
            for pool_id in list(self._memory_pools.keys()):
                cleanup_results[pool_id] = self._cleanup_pool(pool_id)
            
            # Force garbage collection
            gc.collect()
            
            final_usage = self._current_usage_gb
            freed_gb = initial_usage - final_usage
            
            optimization_result = {
                "initial_usage_gb": initial_usage,
                "final_usage_gb": final_usage,
                "freed_gb": freed_gb,
                "pool_cleanups": cleanup_results
            }
            
            self.logger.log_operation("memory_optimization_complete", optimization_result)
            return optimization_result
    
    def _cleanup_pool(self, pool_id: str) -> Dict[str, Any]:
        """Clean up unused memory in a specific pool."""
        if pool_id not in self._memory_pools:
            return {"status": "pool_not_found"}
        
        pool = self._memory_pools[pool_id]
        initial_usage = pool.used_mb
        
        # For now, we can't actually free specific allocations without tracking them
        # This is a placeholder for more advanced memory tracking
        # In a real implementation, we'd track individual allocations and free unused ones
        
        return {
            "status": "cleanup_attempted",
            "initial_usage_mb": initial_usage,
            "final_usage_mb": pool.used_mb,
            "freed_mb": initial_usage - pool.used_mb
        }
    
    def _trigger_global_cleanup(self):
        """Trigger global memory cleanup when approaching limits."""
        self.logger.log_operation("global_memory_cleanup_triggered", {"usage_gb": self._current_usage_gb})
        
        # Run garbage collection
        gc.collect()
        
        # Clean up all pools
        for pool_id in self._memory_pools:
            self._cleanup_pool(pool_id)
        
        # Notify registered callbacks
        for callback in self._allocation_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.log_error(e, {"operation": "memory_cleanup_callback"})
    
    def _get_system_memory_info(self) -> Dict[str, float]:
        """Get current system memory information."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": (memory.total - memory.available) / (1024**3),
                "usage_percent": memory.percent
            }
        except Exception:
            return {}
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a callback to be called during memory cleanup."""
        self._allocation_callbacks.append(callback)


class BottleneckDetector:
    """Detects system bottlenecks and provides optimization recommendations."""
    
    def __init__(self, logger: ILogger):
        self.logger = logger
        self._metrics_history = deque(maxlen=30)  # Keep last 30 measurements
        self._bottleneck_thresholds = {
            "cpu_high": 90.0,
            "cpu_low": 20.0,
            "memory_high": 85.0,
            "memory_low": 30.0,
            "gpu_high": 95.0,
            "gpu_low": 15.0,
            "disk_io_high": 100.0  # MB/s
        }
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics for bottleneck detection."""
        self._metrics_history.append(metrics)
    
    def detect_current_bottlenecks(self) -> Dict[str, Any]:
        """Detect current system bottlenecks.
        
        Returns:
            Dictionary with detected bottlenecks and recommendations
        """
        if len(self._metrics_history) < 5:
            return {"status": "insufficient_data", "bottlenecks": [], "recommendations": []}
        
        recent_metrics = list(self._metrics_history)[-5:]
        bottlenecks = []
        recommendations = []
        
        # CPU bottleneck analysis
        avg_cpu = sum(m.get("cpu_usage_percent", 0) for m in recent_metrics) / len(recent_metrics)
        if avg_cpu > self._bottleneck_thresholds["cpu_high"]:
            bottlenecks.append("cpu_overload")
            recommendations.append("Reduce thread count or batch size to lower CPU usage")
        elif avg_cpu < self._bottleneck_thresholds["cpu_low"]:
            bottlenecks.append("cpu_underutilized")
            recommendations.append("Increase thread count or batch size to better utilize CPU")
        
        # Memory bottleneck analysis
        avg_memory = sum(m.get("memory_usage_percent", 0) for m in recent_metrics) / len(recent_metrics)
        if avg_memory > self._bottleneck_thresholds["memory_high"]:
            bottlenecks.append("memory_pressure")
            recommendations.append("Enable memory optimization or reduce batch size")
        elif avg_memory < self._bottleneck_thresholds["memory_low"]:
            bottlenecks.append("memory_underutilized")
            recommendations.append("Consider increasing batch size or tile size")
        
        # GPU bottleneck analysis (if GPU metrics available)
        gpu_metrics = [m for m in recent_metrics if m.get("gpu_usage_percent") is not None]
        if gpu_metrics:
            avg_gpu = sum(m.get("gpu_usage_percent", 0) for m in gpu_metrics) / len(gpu_metrics)
            if avg_gpu > self._bottleneck_thresholds["gpu_high"]:
                bottlenecks.append("gpu_overload")
                recommendations.append("Reduce GPU batch size or tile size")
            elif avg_gpu < self._bottleneck_thresholds["gpu_low"]:
                bottlenecks.append("gpu_underutilized")
                recommendations.append("Increase GPU utilization with larger batches")
        
        # Disk I/O bottleneck analysis
        avg_disk_read = sum(m.get("disk_io_read_mb_s", 0) for m in recent_metrics) / len(recent_metrics)
        avg_disk_write = sum(m.get("disk_io_write_mb_s", 0) for m in recent_metrics) / len(recent_metrics)
        
        if avg_disk_read > self._bottleneck_thresholds["disk_io_high"] or avg_disk_write > self._bottleneck_thresholds["disk_io_high"]:
            bottlenecks.append("disk_io_bottleneck")
            recommendations.append("Consider using faster storage or reducing I/O operations")
        
        result = {
            "status": "analysis_complete",
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "metrics_summary": {
                "avg_cpu_usage": avg_cpu,
                "avg_memory_usage": avg_memory,
                "avg_disk_read_mb_s": avg_disk_read,
                "avg_disk_write_mb_s": avg_disk_write
            }
        }
        
        if gpu_metrics:
            result["metrics_summary"]["avg_gpu_usage"] = sum(m.get("gpu_usage_percent", 0) for m in gpu_metrics) / len(gpu_metrics)
        
        return result


class PerformanceManager(IPerformanceManager):
    """
    Performance manager for hardware optimization and resource allocation.
    
    Provides system analysis, dynamic resource allocation, and performance monitoring
    capabilities for optimal video processing performance.
    """
    
    def __init__(self, logger: ILogger):
        """Initialize performance manager.
        
        Args:
            logger: Logger instance for performance logging
        """
        self.logger = logger
        self._system_resources: Optional[SystemResources] = None
        self._performance_monitor_active = False
        self._performance_history = []
        self._lock = threading.Lock()
        
        # Initialize advanced components
        self._adaptive_manager = AdaptiveResourceManager(logger)
        self._memory_manager = MemoryManager(logger)
        self._bottleneck_detector = BottleneckDetector(logger)
        
        # Initialize system analysis
        self._analyze_system_resources()
        
        # Set up memory management
        if self._system_resources:
            # Set memory limit to 80% of available RAM
            memory_limit = self._system_resources.available_ram_gb * 0.8
            self._memory_manager.set_memory_limit(memory_limit)
            
            # Create default memory pools
            self._create_default_memory_pools()
    
    def analyze_system_resources(self) -> Dict[str, Any]:
        """Analyze available hardware resources.
        
        Returns:
            Dictionary with comprehensive system resource information
        """
        if self._system_resources is None:
            self._analyze_system_resources()
        
        return asdict(self._system_resources)
    
    def optimize_processing_parameters(self, task_type: str) -> PerformanceConfig:
        """Optimize parameters based on available hardware.
        
        Args:
            task_type: Type of processing task ('video_extraction', 'ai_upscaling', 
                      'audio_processing', 'video_assembly')
            
        Returns:
            PerformanceConfig with optimized parameters
        """
        if self._system_resources is None:
            self._analyze_system_resources()
        
        # Base configuration
        cpu_threads = self._calculate_optimal_threads(task_type)
        memory_limit = self._calculate_memory_limit(task_type)
        gpu_enabled = self._should_use_gpu(task_type)
        
        base_config = PerformanceConfig(
            cpu_threads=cpu_threads,
            gpu_memory_limit=memory_limit,
            memory_optimization=True,
            parallel_processing=cpu_threads > 1
        )
        
        # Apply adaptive adjustments based on historical performance
        optimized_config = self._adaptive_manager.get_adaptive_config(task_type, base_config)
        
        # Check for current bottlenecks and adjust accordingly
        bottlenecks = self._bottleneck_detector.detect_current_bottlenecks()
        if bottlenecks:
            optimized_config = self._adjust_for_bottlenecks(optimized_config, bottlenecks)
        
        self.logger.log_operation(
            "optimize_processing_parameters",
            {
                "task_type": task_type,
                "base_config": asdict(base_config),
                "optimized_config": asdict(optimized_config),
                "bottlenecks": bottlenecks,
                "system_resources": asdict(self._system_resources)
            }
        )
        
        return optimized_config
    
    def allocate_resources(self, cpu_threads: int, gpu_usage: bool) -> Dict[str, Any]:
        """Allocate optimal system resources.
        
        Args:
            cpu_threads: Number of CPU threads to use
            gpu_usage: Whether to use GPU acceleration
            
        Returns:
            Dictionary with resource allocation details
        """
        if self._system_resources is None:
            self._analyze_system_resources()
        
        # Validate and adjust thread count
        max_threads = self._system_resources.cpu_cores
        adjusted_threads = min(cpu_threads, max_threads)
        
        # Calculate memory allocation
        memory_per_thread = self._system_resources.available_ram_gb / adjusted_threads
        safe_memory_limit = min(memory_per_thread * 0.8, 4.0)  # Cap at 4GB per thread
        
        # Determine batch size and tile size based on resources
        batch_size, tile_size = self._calculate_processing_parameters(
            adjusted_threads, safe_memory_limit, gpu_usage
        )
        
        allocation = ResourceAllocation(
            cpu_threads=adjusted_threads,
            memory_limit_gb=safe_memory_limit,
            gpu_enabled=gpu_usage and self._system_resources.cuda_available,
            batch_size=batch_size,
            tile_size=tile_size,
            parallel_processing=adjusted_threads > 1
        )
        
        # Create memory pool for this allocation if needed
        pool_id = f"allocation_{int(time.time())}"
        pool_size_mb = safe_memory_limit * 1024
        self._memory_manager.create_memory_pool(pool_id, pool_size_mb)
        
        allocation_dict = asdict(allocation)
        allocation_dict["memory_pool_id"] = pool_id
        
        self.logger.log_operation(
            "allocate_resources",
            {
                "requested_threads": cpu_threads,
                "allocated_threads": adjusted_threads,
                "gpu_requested": gpu_usage,
                "allocation": allocation_dict
            }
        )
        
        return allocation_dict
    
    def monitor_performance(self) -> Dict[str, Any]:
        """Monitor real-time performance metrics.
        
        Returns:
            Dictionary with current performance metrics
        """
        try:
            # CPU and memory metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent
            memory_used_gb = (memory.total - memory.available) / (1024**3)
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            disk_read_mb_s = 0.0
            disk_write_mb_s = 0.0
            
            if hasattr(self, '_last_disk_io'):
                time_delta = time.time() - self._last_disk_io_time
                if time_delta > 0:
                    read_delta = disk_io.read_bytes - self._last_disk_io.read_bytes
                    write_delta = disk_io.write_bytes - self._last_disk_io.write_bytes
                    disk_read_mb_s = (read_delta / time_delta) / (1024**2)
                    disk_write_mb_s = (write_delta / time_delta) / (1024**2)
            
            self._last_disk_io = disk_io
            self._last_disk_io_time = time.time()
            
            # GPU metrics (if available)
            gpu_usage_percent = None
            gpu_memory_usage_percent = None
            
            if self._system_resources and self._system_resources.gpu_model:
                gpu_metrics = self._get_gpu_metrics()
                gpu_usage_percent = gpu_metrics.get('usage_percent')
                gpu_memory_usage_percent = gpu_metrics.get('memory_usage_percent')
            
            # Thread count
            active_threads = threading.active_count()
            
            metrics = PerformanceMetrics(
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory_usage_percent,
                memory_used_gb=memory_used_gb,
                gpu_usage_percent=gpu_usage_percent,
                gpu_memory_usage_percent=gpu_memory_usage_percent,
                disk_io_read_mb_s=disk_read_mb_s,
                disk_io_write_mb_s=disk_write_mb_s,
                active_threads=active_threads,
                timestamp=time.time()
            )
            
            # Store in history (keep last 100 measurements)
            with self._lock:
                self._performance_history.append(metrics)
                if len(self._performance_history) > 100:
                    self._performance_history.pop(0)
            
            # Update bottleneck detector
            self._bottleneck_detector.update_metrics(asdict(metrics))
            
            return asdict(metrics)
            
        except Exception as e:
            self.logger.log_error(e, {"operation": "monitor_performance"})
            return {}
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics.
        
        Returns:
            Dictionary with memory statistics from memory manager
        """
        return self._memory_manager.get_memory_stats()
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage across all pools.
        
        Returns:
            Dictionary with optimization results
        """
        return self._memory_manager.optimize_memory_usage()
    
    def record_task_performance(self, task_type: str, duration: float, resources_used: Dict[str, Any], success: bool):
        """Record task performance for adaptive learning.
        
        Args:
            task_type: Type of task that was performed
            duration: Task duration in seconds
            resources_used: Dictionary with resource usage metrics
            success: Whether the task completed successfully
        """
        self._adaptive_manager.record_workload(task_type, duration, resources_used, success)
    
    def get_resource_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for resource optimization.
        
        Returns:
            Dictionary with optimization recommendations
        """
        bottlenecks = self._bottleneck_detector.detect_current_bottlenecks()
        memory_stats = self._memory_manager.get_memory_stats()
        
        recommendations = {
            "bottlenecks": bottlenecks,
            "memory_optimization": [],
            "performance_tuning": []
        }
        
        # Memory recommendations
        if memory_stats.get("global_usage_percent", 0) > 80:
            recommendations["memory_optimization"].append("Consider reducing batch size or enabling memory optimization")
        
        if memory_stats.get("global_usage_percent", 0) < 30:
            recommendations["memory_optimization"].append("Memory underutilized - consider increasing batch size")
        
        # Performance tuning recommendations
        if len(self._performance_history) > 10:
            recent_cpu = sum(m.cpu_usage_percent for m in self._performance_history[-10:]) / 10
            if recent_cpu > 90:
                recommendations["performance_tuning"].append("CPU overloaded - consider reducing thread count")
            elif recent_cpu < 30:
                recommendations["performance_tuning"].append("CPU underutilized - consider increasing thread count")
        
        return recommendations
    
    def _create_default_memory_pools(self):
        """Create default memory pools for common operations."""
        if not self._system_resources:
            return
        
        # Create pools based on available memory
        total_memory_mb = self._system_resources.available_ram_gb * 1024 * 0.8  # 80% of available
        
        # Video processing pool (40% of available memory)
        video_pool_mb = total_memory_mb * 0.4
        self._memory_manager.create_memory_pool("video_processing", video_pool_mb)
        
        # Audio processing pool (20% of available memory)
        audio_pool_mb = total_memory_mb * 0.2
        self._memory_manager.create_memory_pool("audio_processing", audio_pool_mb)
        
        # AI model pool (30% of available memory)
        ai_pool_mb = total_memory_mb * 0.3
        self._memory_manager.create_memory_pool("ai_models", ai_pool_mb)
        
        # Temporary operations pool (10% of available memory)
        temp_pool_mb = total_memory_mb * 0.1
        self._memory_manager.create_memory_pool("temporary", temp_pool_mb)
        
        self.logger.log_operation(
            "default_memory_pools_created",
            {
                "total_allocated_mb": total_memory_mb,
                "pools": {
                    "video_processing": video_pool_mb,
                    "audio_processing": audio_pool_mb,
                    "ai_models": ai_pool_mb,
                    "temporary": temp_pool_mb
                }
            }
        )
    
    def _adjust_for_bottlenecks(self, config: PerformanceConfig, bottlenecks: Dict[str, Any]) -> PerformanceConfig:
        """Adjust configuration based on detected bottlenecks."""
        adjusted_config = PerformanceConfig(
            cpu_threads=config.cpu_threads,
            gpu_memory_limit=config.gpu_memory_limit,
            memory_optimization=config.memory_optimization,
            parallel_processing=config.parallel_processing
        )
        
        detected_bottlenecks = bottlenecks.get("bottlenecks", [])
        
        # Adjust for CPU bottlenecks
        if "cpu_overload" in detected_bottlenecks:
            adjusted_config.cpu_threads = max(1, int(adjusted_config.cpu_threads * 0.8))
        elif "cpu_underutilized" in detected_bottlenecks:
            max_threads = self._system_resources.cpu_cores if self._system_resources else 4
            adjusted_config.cpu_threads = min(max_threads, int(adjusted_config.cpu_threads * 1.2))
        
        # Adjust for memory bottlenecks
        if "memory_pressure" in detected_bottlenecks:
            adjusted_config.gpu_memory_limit *= 0.8
            adjusted_config.memory_optimization = True
        elif "memory_underutilized" in detected_bottlenecks:
            adjusted_config.gpu_memory_limit *= 1.2
        
        # Adjust for GPU bottlenecks
        if "gpu_overload" in detected_bottlenecks:
            adjusted_config.gpu_memory_limit *= 0.7
        elif "gpu_underutilized" in detected_bottlenecks:
            adjusted_config.gpu_memory_limit *= 1.3
        
        return adjusted_config
    
    def get_performance_history(self) -> list:
        """Get performance monitoring history.
        
        Returns:
            List of performance metrics over time
        """
        with self._lock:
            return [asdict(metric) for metric in self._performance_history.copy()]
    
    def detect_bottlenecks(self) -> Dict[str, Any]:
        """Detect system bottlenecks and suggest optimizations.
        
        Returns:
            Dictionary with bottleneck analysis and recommendations
        """
        if len(self._performance_history) < 10:
            return {"status": "insufficient_data", "message": "Need more performance data"}
        
        with self._lock:
            recent_metrics = self._performance_history[-10:]
        
        # Analyze metrics
        avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_disk_read = sum(m.disk_io_read_mb_s for m in recent_metrics) / len(recent_metrics)
        avg_disk_write = sum(m.disk_io_write_mb_s for m in recent_metrics) / len(recent_metrics)
        
        bottlenecks = []
        recommendations = []
        
        # CPU bottleneck detection
        if avg_cpu > 90:
            bottlenecks.append("cpu_high_usage")
            recommendations.append("Consider reducing thread count or batch size")
        elif avg_cpu < 30:
            bottlenecks.append("cpu_underutilized")
            recommendations.append("Consider increasing thread count or batch size")
        
        # Memory bottleneck detection
        if avg_memory > 85:
            bottlenecks.append("memory_high_usage")
            recommendations.append("Reduce batch size or enable memory optimization")
        
        # Disk I/O bottleneck detection
        if avg_disk_read > 100 or avg_disk_write > 100:  # MB/s
            bottlenecks.append("disk_io_high")
            recommendations.append("Consider using SSD or reducing I/O operations")
        
        # GPU analysis (if available)
        gpu_bottlenecks = self._analyze_gpu_bottlenecks(recent_metrics)
        bottlenecks.extend(gpu_bottlenecks.get("bottlenecks", []))
        recommendations.extend(gpu_bottlenecks.get("recommendations", []))
        
        analysis = {
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "metrics_summary": {
                "avg_cpu_usage": avg_cpu,
                "avg_memory_usage": avg_memory,
                "avg_disk_read_mb_s": avg_disk_read,
                "avg_disk_write_mb_s": avg_disk_write
            }
        }
        
        self.logger.log_operation("detect_bottlenecks", analysis)
        return analysis
    
    def _analyze_system_resources(self) -> None:
        """Perform comprehensive system resource analysis."""
        try:
            # CPU information
            cpu_cores = mp.cpu_count()
            cpu_model = self._get_cpu_model()
            
            # Memory information
            memory = psutil.virtual_memory()
            total_ram_gb = memory.total / (1024**3)
            available_ram_gb = memory.available / (1024**3)
            
            # GPU information
            gpu_model, gpu_memory_gb = self._get_gpu_info()
            cuda_available = self._check_cuda_availability()
            opencv_cuda_available = self._check_opencv_cuda()
            
            # Storage information
            disk_usage = psutil.disk_usage('.')
            storage_available_gb = disk_usage.free / (1024**3)
            
            # OS information
            os_info = f"{platform.system()} {platform.release()}"
            
            self._system_resources = SystemResources(
                cpu_cores=cpu_cores,
                cpu_model=cpu_model,
                total_ram_gb=total_ram_gb,
                available_ram_gb=available_ram_gb,
                gpu_model=gpu_model,
                gpu_memory_gb=gpu_memory_gb,
                cuda_available=cuda_available,
                opencv_cuda_available=opencv_cuda_available,
                storage_available_gb=storage_available_gb,
                os_info=os_info
            )
            
            self.logger.log_operation(
                "system_analysis_complete",
                {"system_resources": asdict(self._system_resources)}
            )
            
        except Exception as e:
            self.logger.log_error(e, {"operation": "analyze_system_resources"})
            # Create minimal fallback configuration
            self._system_resources = SystemResources(
                cpu_cores=mp.cpu_count(),
                cpu_model="Unknown",
                total_ram_gb=8.0,
                available_ram_gb=4.0,
                gpu_model=None,
                gpu_memory_gb=None,
                cuda_available=False,
                opencv_cuda_available=False,
                storage_available_gb=10.0,
                os_info="Unknown"
            )
    
    def _get_cpu_model(self) -> str:
        """Get CPU model information."""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name", "/value"],
                    capture_output=True, text=True, check=True
                )
                for line in result.stdout.split('\n'):
                    if 'Name=' in line:
                        return line.split('=', 1)[1].strip()
            else:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            return line.split(':', 1)[1].strip()
        except Exception:
            pass
        return "Unknown CPU"
    
    def _get_gpu_info(self) -> Tuple[Optional[str], Optional[float]]:
        """Get GPU model and memory information."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            
            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        gpu_model = parts[0].strip()
                        gpu_memory_mb = float(parts[1].strip())
                        gpu_memory_gb = round(gpu_memory_mb / 1024, 2)
                        return gpu_model, gpu_memory_gb
        except Exception:
            pass
        return None, None
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _check_opencv_cuda(self) -> bool:
        """Check if OpenCV CUDA support is available."""
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            return False
    
    def _calculate_optimal_threads(self, task_type: str) -> int:
        """Calculate optimal thread count for task type."""
        if not self._system_resources:
            return mp.cpu_count()
        
        base_threads = self._system_resources.cpu_cores
        
        # Task-specific thread optimization
        if task_type == "video_extraction":
            # I/O bound task, can use more threads
            return min(base_threads * 2, 16)
        elif task_type == "ai_upscaling":
            # CPU/GPU intensive, use fewer threads to avoid contention
            return max(1, base_threads // 2)
        elif task_type == "audio_processing":
            # Moderate CPU usage
            return max(2, base_threads // 2)
        elif task_type == "video_assembly":
            # I/O and CPU intensive
            return max(2, base_threads - 1)
        else:
            # Default case
            return max(2, base_threads - 1)
    
    def _calculate_memory_limit(self, task_type: str) -> float:
        """Calculate memory limit for task type."""
        if not self._system_resources:
            return 2.0
        
        available_memory = self._system_resources.available_ram_gb
        
        # Reserve memory for system and other processes
        usable_memory = available_memory * 0.8
        
        # Task-specific memory allocation
        if task_type == "ai_upscaling":
            # AI models need more memory
            return min(usable_memory, 8.0)
        elif task_type == "video_extraction":
            # Frame extraction needs moderate memory
            return min(usable_memory, 4.0)
        else:
            # Default allocation
            return min(usable_memory, 2.0)
    
    def _should_use_gpu(self, task_type: str) -> bool:
        """Determine if GPU should be used for task type."""
        if not self._system_resources:
            return False
        
        # Only use GPU if CUDA is available and task benefits from it
        if not self._system_resources.cuda_available:
            return False
        
        # Based on the steering rules, GPU is mainly beneficial for AI models
        if task_type == "ai_upscaling":
            return True
        else:
            # For other tasks, CPU is often faster due to overhead
            return False
    
    def _calculate_processing_parameters(self, threads: int, memory_limit: float, gpu_enabled: bool) -> Tuple[int, int]:
        """Calculate optimal batch size and tile size."""
        # Base parameters
        base_batch_size = 4
        base_tile_size = 512
        
        # Adjust based on available resources
        if memory_limit > 6.0:
            batch_size = base_batch_size * 2
            tile_size = base_tile_size * 2
        elif memory_limit > 3.0:
            batch_size = base_batch_size
            tile_size = base_tile_size
        else:
            batch_size = max(1, base_batch_size // 2)
            tile_size = base_tile_size // 2
        
        # GPU-specific adjustments
        if gpu_enabled and self._system_resources and self._system_resources.gpu_memory_gb:
            if self._system_resources.gpu_memory_gb >= 6.0:
                tile_size = min(tile_size, 1024)
            elif self._system_resources.gpu_memory_gb >= 4.0:
                tile_size = min(tile_size, 512)
            else:
                tile_size = min(tile_size, 256)
        
        return batch_size, tile_size
    
    def _get_gpu_metrics(self) -> Dict[str, Optional[float]]:
        """Get current GPU usage metrics."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            
            if result.stdout.strip():
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 3:
                    gpu_usage = float(parts[0])
                    memory_used = float(parts[1])
                    memory_total = float(parts[2])
                    memory_usage_percent = (memory_used / memory_total) * 100
                    
                    return {
                        "usage_percent": gpu_usage,
                        "memory_usage_percent": memory_usage_percent
                    }
        except Exception:
            pass
        
        return {"usage_percent": None, "memory_usage_percent": None}
    
    def _analyze_gpu_bottlenecks(self, recent_metrics: list) -> Dict[str, list]:
        """Analyze GPU-specific bottlenecks."""
        bottlenecks = []
        recommendations = []
        
        # Check if we have GPU metrics
        gpu_metrics = [m for m in recent_metrics if m.gpu_usage_percent is not None]
        
        if not gpu_metrics:
            return {"bottlenecks": bottlenecks, "recommendations": recommendations}
        
        avg_gpu_usage = sum(m.gpu_usage_percent for m in gpu_metrics) / len(gpu_metrics)
        avg_gpu_memory = sum(m.gpu_memory_usage_percent for m in gpu_metrics if m.gpu_memory_usage_percent) / len(gpu_metrics)
        
        if avg_gpu_usage > 95:
            bottlenecks.append("gpu_high_usage")
            recommendations.append("GPU at maximum capacity - consider reducing batch size")
        elif avg_gpu_usage < 20:
            bottlenecks.append("gpu_underutilized")
            recommendations.append("GPU underutilized - consider increasing batch size or tile size")
        
        if avg_gpu_memory > 90:
            bottlenecks.append("gpu_memory_high")
            recommendations.append("GPU memory nearly full - reduce tile size or batch size")
        
        return {"bottlenecks": bottlenecks, "recommendations": recommendations}