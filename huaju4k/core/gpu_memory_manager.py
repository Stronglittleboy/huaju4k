"""
GPU显存管理器 - GPUMemoryManager

这个模块实现了智能的GPU显存管理，确保在6GB显存限制下稳定运行。
采用"锯齿型显存使用模式"，模型一次性加载后保持不变，帧处理完成后立即清理。

核心特性:
- 模型一次性加载，全程保持
- 每帧处理后立即清理张量
- 锯齿型显存使用模式（波动但不增长）
- 显存不足时的自动降级处理
- 实时显存监控

实现需求:
- 需求 3: 显存优化管理
- 需求 6: 错误恢复和中断处理
- 需求 7: 性能监控和资源管理
"""

import logging
import gc
import time
from typing import Dict, Optional, Any, List, Tuple, Union
import numpy as np

# GPU相关导入
try:
    import torch
    import torch.cuda
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

logger = logging.getLogger(__name__)


class GPUMemoryStats:
    """GPU内存统计类"""
    
    def __init__(self):
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.total_allocations = 0
        self.total_deallocations = 0
        self.cleanup_count = 0
        self.model_memory = 0


class GPUMemoryManager:
    """
    GPU显存管理器
    
    策略:
    - 模型一次性加载，全程保持
    - 每帧处理后立即清理张量
    - 锯齿型显存使用模式
    - 显存不足时自动降级
    """
    
    def __init__(self, max_vram_mb: int = 6144, 
                 cleanup_threshold: float = 0.85,
                 enable_mixed_precision: bool = True):
        """
        初始化GPU显存管理器
        
        Args:
            max_vram_mb: 最大显存限制（MB）
            cleanup_threshold: 自动清理阈值（0.0-1.0）
            enable_mixed_precision: 是否启用混合精度
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch不可用，无法使用GPU显存管理")
        
        self.max_vram_mb = max_vram_mb
        self.max_vram_bytes = max_vram_mb * 1024 * 1024
        self.cleanup_threshold = cleanup_threshold
        self.enable_mixed_precision = enable_mixed_precision
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_gpu_available = torch.cuda.is_available()
        
        # 模型管理
        self.loaded_models = {}
        self.current_model = None
        self.model_loaded = False
        
        # 统计信息
        self.stats = GPUMemoryStats()
        
        # 性能配置
        if self.is_gpu_available and self.enable_mixed_precision:
            # 启用混合精度训练
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        logger.info(f"GPUMemoryManager初始化完成")
        logger.info(f"设备: {self.device}")
        logger.info(f"GPU可用: {self.is_gpu_available}")
        logger.info(f"最大显存: {max_vram_mb}MB")
        logger.info(f"混合精度: {self.enable_mixed_precision}")
        
        if self.is_gpu_available:
            gpu_info = self._get_gpu_info()
            logger.info(f"GPU信息: {gpu_info['name']}, "
                       f"总显存: {gpu_info['total_memory_mb']:.0f}MB")
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """
        获取GPU信息
        
        Returns:
            Dict[str, Any]: GPU信息
        """
        if not self.is_gpu_available:
            return {"name": "CPU", "total_memory_mb": 0}
        
        gpu_props = torch.cuda.get_device_properties(0)
        return {
            "name": gpu_props.name,
            "total_memory_mb": gpu_props.total_memory / 1024 / 1024,
            "major": gpu_props.major,
            "minor": gpu_props.minor,
            "multi_processor_count": gpu_props.multi_processor_count
        }
    
    def load_model_once(self, model: torch.nn.Module, model_name: str = "default") -> bool:
        """
        一次性加载模型到显存，全程保持
        
        Args:
            model: PyTorch模型
            model_name: 模型名称
            
        Returns:
            bool: 加载是否成功
        """
        try:
            if not self.is_gpu_available:
                logger.info("GPU不可用，模型将在CPU上运行")
                self.loaded_models[model_name] = model
                self.current_model = model
                self.model_loaded = True
                return True
            
            # 检查显存是否足够
            if not self._check_memory_for_model(model):
                logger.error("显存不足，无法加载模型")
                return False
            
            # 将模型移动到GPU
            model = model.to(self.device)
            model.eval()  # 设置为评估模式
            
            # 记录模型显存使用
            if self.is_gpu_available:
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
                self.stats.model_memory = allocated
                logger.info(f"模型显存使用: {allocated / 1024 / 1024:.2f}MB")
            
            self.loaded_models[model_name] = model
            self.current_model = model
            self.model_loaded = True
            
            logger.info(f"模型 {model_name} 加载成功")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return False
    
    def _check_memory_for_model(self, model: torch.nn.Module) -> bool:
        """
        检查显存是否足够加载模型
        
        Args:
            model: PyTorch模型
            
        Returns:
            bool: 显存是否足够
        """
        if not self.is_gpu_available:
            return True
        
        # 估算模型大小
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size = param_size + buffer_size
        
        # 获取当前显存使用
        current_memory = torch.cuda.memory_allocated()
        available_memory = self.max_vram_bytes - current_memory
        
        logger.info(f"模型大小估算: {model_size / 1024 / 1024:.2f}MB")
        logger.info(f"可用显存: {available_memory / 1024 / 1024:.2f}MB")
        
        # 留50%缓冲区用于帧处理
        return model_size < available_memory * 0.5
    
    def process_frame_tensor(self, frame_tensor: torch.Tensor, 
                           model_name: str = "default") -> torch.Tensor:
        """
        处理帧张量，自动管理显存
        
        Args:
            frame_tensor: 输入帧张量
            model_name: 使用的模型名称
            
        Returns:
            torch.Tensor: 处理后的张量
        """
        if model_name not in self.loaded_models:
            raise ValueError(f"模型 {model_name} 未加载")
        
        model = self.loaded_models[model_name]
        
        try:
            # 确保张量在正确的设备上
            if frame_tensor.device != self.device:
                frame_tensor = frame_tensor.to(self.device)
            
            # 使用torch.no_grad()确保不保存计算图
            with torch.no_grad():
                if self.enable_mixed_precision and self.is_gpu_available:
                    # 使用混合精度
                    with torch.cuda.amp.autocast():
                        result = model(frame_tensor)
                else:
                    # 标准精度
                    result = model(frame_tensor)
            
            # 更新统计
            self.stats.total_allocations += 1
            
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"显存不足: {str(e)}")
            # 尝试清理显存后重试
            self.emergency_cleanup()
            
            # 重试一次
            try:
                with torch.no_grad():
                    result = model(frame_tensor)
                return result
            except torch.cuda.OutOfMemoryError:
                logger.error("显存清理后仍然不足，处理失败")
                raise
        
        except Exception as e:
            logger.error(f"帧处理出错: {str(e)}")
            raise
    
    def cleanup_frame_tensors(self, *tensors) -> None:
        """
        清理帧处理产生的张量
        
        Args:
            *tensors: 要清理的张量
        """
        for tensor in tensors:
            if tensor is not None:
                del tensor
        
        self.stats.total_deallocations += len(tensors)
        
        # 定期清理显存缓存
        if self.is_gpu_available:
            if self.stats.total_allocations % 10 == 0:  # 每10次分配清理一次
                torch.cuda.empty_cache()
                self.stats.cleanup_count += 1
    
    def emergency_cleanup(self) -> Dict[str, Any]:
        """
        紧急显存清理
        
        Returns:
            Dict[str, Any]: 清理结果
        """
        logger.warning("执行紧急显存清理")
        
        initial_memory = 0
        if self.is_gpu_available:
            initial_memory = torch.cuda.memory_allocated()
        
        # 1. 强制垃圾回收
        gc.collect()
        
        # 2. 清理PyTorch缓存
        if self.is_gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 3. 清理未使用的变量
        import sys
        frame = sys._getframe()
        while frame:
            frame.f_locals.clear()
            frame = frame.f_back
        
        final_memory = 0
        if self.is_gpu_available:
            final_memory = torch.cuda.memory_allocated()
        
        freed_memory = initial_memory - final_memory
        
        result = {
            "initial_memory_mb": initial_memory / 1024 / 1024,
            "final_memory_mb": final_memory / 1024 / 1024,
            "freed_memory_mb": freed_memory / 1024 / 1024,
            "cleanup_successful": freed_memory > 0
        }
        
        logger.info(f"紧急清理完成: 释放 {result['freed_memory_mb']:.2f}MB")
        return result
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取显存使用统计
        
        Returns:
            Dict[str, Any]: 显存统计信息
        """
        stats = {
            "device": str(self.device),
            "gpu_available": self.is_gpu_available,
            "max_vram_mb": self.max_vram_mb,
            "model_loaded": self.model_loaded,
            "total_allocations": self.stats.total_allocations,
            "total_deallocations": self.stats.total_deallocations,
            "cleanup_count": self.stats.cleanup_count
        }
        
        if self.is_gpu_available:
            # 当前显存使用
            current_allocated = torch.cuda.memory_allocated()
            current_reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
            max_reserved = torch.cuda.max_memory_reserved()
            
            stats.update({
                "current_allocated_mb": current_allocated / 1024 / 1024,
                "current_reserved_mb": current_reserved / 1024 / 1024,
                "max_allocated_mb": max_allocated / 1024 / 1024,
                "max_reserved_mb": max_reserved / 1024 / 1024,
                "model_memory_mb": self.stats.model_memory / 1024 / 1024,
                "memory_utilization": (current_allocated / self.max_vram_bytes) * 100,
                "gpu_info": self._get_gpu_info()
            })
            
            # 更新峰值统计
            if current_allocated > self.stats.peak_allocated:
                self.stats.peak_allocated = current_allocated
            if current_reserved > self.stats.peak_reserved:
                self.stats.peak_reserved = current_reserved
            
            stats.update({
                "peak_allocated_mb": self.stats.peak_allocated / 1024 / 1024,
                "peak_reserved_mb": self.stats.peak_reserved / 1024 / 1024
            })
        
        return stats
    
    def monitor_memory_usage(self) -> bool:
        """
        监控显存使用，如果超过阈值则自动清理
        
        Returns:
            bool: 是否需要采取行动
        """
        if not self.is_gpu_available:
            return True
        
        current_memory = torch.cuda.memory_allocated()
        usage_ratio = current_memory / self.max_vram_bytes
        
        if usage_ratio > self.cleanup_threshold:
            logger.warning(f"显存使用率过高: {usage_ratio*100:.1f}%")
            self.emergency_cleanup()
            return False
        
        return True
    
    def optimize_memory_settings(self) -> Dict[str, Any]:
        """
        优化显存设置
        
        Returns:
            Dict[str, Any]: 优化结果
        """
        if not self.is_gpu_available:
            return {"optimized": False, "reason": "GPU不可用"}
        
        # 1. 设置内存分配策略
        torch.cuda.empty_cache()
        
        # 2. 启用内存池
        if hasattr(torch.cuda, 'memory_pool'):
            torch.cuda.memory_pool.set_per_process_memory_fraction(0.9)
        
        # 3. 优化缓存策略
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # 4. 设置显存增长策略
        if hasattr(torch.cuda, 'set_memory_growth'):
            torch.cuda.set_memory_growth(True)
        
        return {
            "optimized": True,
            "memory_fraction": 0.9,
            "cudnn_benchmark": True,
            "memory_growth": True
        }
    
    def get_available_memory(self) -> int:
        """
        获取可用显存大小
        
        Returns:
            int: 可用显存大小（字节）
        """
        if not self.is_gpu_available:
            return 0
        
        current_memory = torch.cuda.memory_allocated()
        return max(0, self.max_vram_bytes - current_memory)
    
    def is_memory_available(self, required_size: int) -> bool:
        """
        检查是否有足够的显存
        
        Args:
            required_size: 需要的显存大小（字节）
            
        Returns:
            bool: 是否有足够显存
        """
        return self.get_available_memory() >= required_size
    
    def reset_peak_stats(self) -> None:
        """重置峰值统计"""
        if self.is_gpu_available:
            torch.cuda.reset_peak_memory_stats()
        
        self.stats.peak_allocated = 0
        self.stats.peak_reserved = 0
        
        logger.info("峰值统计已重置")
    
    def unload_model(self, model_name: str = "default") -> bool:
        """
        卸载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 卸载是否成功
        """
        if model_name not in self.loaded_models:
            return False
        
        # 删除模型
        del self.loaded_models[model_name]
        
        if self.current_model and model_name == "default":
            self.current_model = None
            self.model_loaded = False
        
        # 清理显存
        if self.is_gpu_available:
            torch.cuda.empty_cache()
        
        logger.info(f"模型 {model_name} 已卸载")
        return True
    
    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            # 卸载所有模型
            for model_name in list(self.loaded_models.keys()):
                self.unload_model(model_name)
            
            # 最终清理
            if self.is_gpu_available:
                torch.cuda.empty_cache()
        except:
            pass