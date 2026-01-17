"""
内存帧缓冲管理器 - MemoryFrameBuffer

这个模块实现了高效的内存帧缓冲管理，确保帧数据仅存在于内存中，
自动管理内存使用量，防止内存溢出。

核心特性:
- 自动内存使用量监控
- 智能帧生命周期管理
- 内存不足时的自动清理
- 高效的帧数据存储和检索

实现需求:
- 需求 2: 内存帧处理管道
- 需求 5: 磁盘空间约束遵循
- 需求 7: 性能监控和资源管理
"""

import logging
import gc
import time
from typing import Dict, Optional, Any, List, Tuple
import numpy as np
import psutil

logger = logging.getLogger(__name__)


class FrameMetadata:
    """帧元数据类"""
    
    def __init__(self, frame_id: int, frame_size: int, timestamp: float):
        self.frame_id = frame_id
        self.frame_size = frame_size  # 字节数
        self.timestamp = timestamp
        self.access_count = 0
        self.last_access = timestamp


class MemoryFrameBuffer:
    """
    内存帧缓冲管理器
    
    职责:
    - 管理帧在内存中的生命周期
    - 自动释放已处理帧
    - 监控内存使用量
    - 防止内存溢出
    """
    
    def __init__(self, max_buffer_size_mb: int = 1024, 
                 auto_cleanup_threshold: float = 0.8):
        """
        初始化内存帧缓冲器
        
        Args:
            max_buffer_size_mb: 最大缓冲区大小（MB）
            auto_cleanup_threshold: 自动清理阈值（0.0-1.0）
        """
        self.max_buffer_size = max_buffer_size_mb * 1024 * 1024  # 转换为字节
        self.auto_cleanup_threshold = auto_cleanup_threshold
        
        # 帧存储
        self.frames: Dict[int, np.ndarray] = {}
        self.metadata: Dict[int, FrameMetadata] = {}
        
        # 统计信息
        self.current_memory_usage = 0
        self.total_frames_added = 0
        self.total_frames_released = 0
        self.cleanup_count = 0
        
        # 性能监控
        self.peak_memory_usage = 0
        self.average_frame_size = 0
        
        logger.info(f"MemoryFrameBuffer初始化: 最大缓冲区 {max_buffer_size_mb}MB")
    
    def add_frame(self, frame_id: int, frame: np.ndarray) -> bool:
        """
        添加帧到缓冲区
        
        Args:
            frame_id: 帧ID
            frame: 帧数据（numpy数组）
            
        Returns:
            bool: 是否成功添加
        """
        frame_size = frame.nbytes
        timestamp = time.time()
        
        # 检查内存限制
        if self.current_memory_usage + frame_size > self.max_buffer_size:
            # 尝试自动清理
            if not self._auto_cleanup():
                logger.warning(f"内存不足，无法添加帧 {frame_id}")
                return False
        
        # 如果帧已存在，先释放旧的
        if frame_id in self.frames:
            self.release_frame(frame_id)
        
        # 添加新帧
        self.frames[frame_id] = frame.copy()  # 创建副本确保数据安全
        self.metadata[frame_id] = FrameMetadata(frame_id, frame_size, timestamp)
        
        # 更新统计
        self.current_memory_usage += frame_size
        self.total_frames_added += 1
        
        # 更新峰值内存使用
        if self.current_memory_usage > self.peak_memory_usage:
            self.peak_memory_usage = self.current_memory_usage
        
        # 更新平均帧大小
        self.average_frame_size = (
            (self.average_frame_size * (self.total_frames_added - 1) + frame_size) 
            / self.total_frames_added
        )
        
        logger.debug(f"添加帧 {frame_id}: {frame_size} 字节, "
                    f"总内存使用: {self.current_memory_usage / 1024 / 1024:.2f}MB")
        
        return True
    
    def get_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """
        获取帧数据
        
        Args:
            frame_id: 帧ID
            
        Returns:
            Optional[np.ndarray]: 帧数据，如果不存在则返回None
        """
        if frame_id not in self.frames:
            return None
        
        # 更新访问统计
        metadata = self.metadata[frame_id]
        metadata.access_count += 1
        metadata.last_access = time.time()
        
        return self.frames[frame_id]
    
    def release_frame(self, frame_id: int) -> bool:
        """
        立即释放帧内存
        
        Args:
            frame_id: 帧ID
            
        Returns:
            bool: 是否成功释放
        """
        if frame_id not in self.frames:
            return False
        
        # 获取帧大小
        frame_size = self.metadata[frame_id].frame_size
        
        # 删除帧数据
        del self.frames[frame_id]
        del self.metadata[frame_id]
        
        # 更新统计
        self.current_memory_usage -= frame_size
        self.total_frames_released += 1
        
        logger.debug(f"释放帧 {frame_id}: {frame_size} 字节, "
                    f"剩余内存使用: {self.current_memory_usage / 1024 / 1024:.2f}MB")
        
        return True
    
    def release_frames_batch(self, frame_ids: List[int]) -> int:
        """
        批量释放帧
        
        Args:
            frame_ids: 帧ID列表
            
        Returns:
            int: 成功释放的帧数量
        """
        released_count = 0
        for frame_id in frame_ids:
            if self.release_frame(frame_id):
                released_count += 1
        
        # 强制垃圾回收
        gc.collect()
        
        logger.info(f"批量释放 {released_count}/{len(frame_ids)} 帧")
        return released_count
    
    def _auto_cleanup(self) -> bool:
        """
        自动清理内存
        
        Returns:
            bool: 是否成功释放足够的内存
        """
        if len(self.frames) == 0:
            return False
        
        logger.info("开始自动内存清理")
        
        # 计算需要释放的内存量
        target_usage = int(self.max_buffer_size * (1 - self.auto_cleanup_threshold))
        memory_to_free = self.current_memory_usage - target_usage
        
        if memory_to_free <= 0:
            return True
        
        # 按访问时间排序，优先释放最久未访问的帧
        sorted_frames = sorted(
            self.metadata.items(),
            key=lambda x: x[1].last_access
        )
        
        freed_memory = 0
        frames_to_release = []
        
        for frame_id, metadata in sorted_frames:
            frames_to_release.append(frame_id)
            freed_memory += metadata.frame_size
            
            if freed_memory >= memory_to_free:
                break
        
        # 批量释放帧
        released_count = self.release_frames_batch(frames_to_release)
        self.cleanup_count += 1
        
        logger.info(f"自动清理完成: 释放 {released_count} 帧, "
                   f"释放内存 {freed_memory / 1024 / 1024:.2f}MB")
        
        return freed_memory >= memory_to_free
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取内存使用统计
        
        Returns:
            Dict[str, Any]: 内存统计信息
        """
        # 获取系统内存信息
        system_memory = psutil.virtual_memory()
        
        return {
            # 缓冲区统计
            "buffer_memory_mb": self.current_memory_usage / 1024 / 1024,
            "buffer_memory_percent": (self.current_memory_usage / self.max_buffer_size) * 100,
            "peak_memory_mb": self.peak_memory_usage / 1024 / 1024,
            "max_buffer_mb": self.max_buffer_size / 1024 / 1024,
            
            # 帧统计
            "active_frames": len(self.frames),
            "total_frames_added": self.total_frames_added,
            "total_frames_released": self.total_frames_released,
            "average_frame_size_mb": self.average_frame_size / 1024 / 1024,
            
            # 清理统计
            "cleanup_count": self.cleanup_count,
            
            # 系统内存统计
            "system_memory_total_gb": system_memory.total / 1024 / 1024 / 1024,
            "system_memory_available_gb": system_memory.available / 1024 / 1024 / 1024,
            "system_memory_percent": system_memory.percent
        }
    
    def get_frame_list(self) -> List[int]:
        """
        获取当前缓冲区中的所有帧ID
        
        Returns:
            List[int]: 帧ID列表
        """
        return list(self.frames.keys())
    
    def clear_all_frames(self) -> int:
        """
        清空所有帧
        
        Returns:
            int: 清理的帧数量
        """
        frame_count = len(self.frames)
        
        self.frames.clear()
        self.metadata.clear()
        self.current_memory_usage = 0
        
        # 强制垃圾回收
        gc.collect()
        
        logger.info(f"清空所有帧: {frame_count} 帧")
        return frame_count
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        优化内存使用
        
        Returns:
            Dict[str, Any]: 优化结果
        """
        initial_memory = self.current_memory_usage
        initial_frames = len(self.frames)
        
        # 1. 清理长时间未访问的帧
        current_time = time.time()
        old_frames = []
        
        for frame_id, metadata in self.metadata.items():
            # 超过60秒未访问的帧
            if current_time - metadata.last_access > 60:
                old_frames.append(frame_id)
        
        released_old = self.release_frames_batch(old_frames)
        
        # 2. 如果内存使用仍然过高，释放访问次数最少的帧
        if self.current_memory_usage > self.max_buffer_size * 0.7:
            sorted_by_access = sorted(
                self.metadata.items(),
                key=lambda x: x[1].access_count
            )
            
            low_access_frames = [
                frame_id for frame_id, _ in sorted_by_access[:len(sorted_by_access)//4]
            ]
            released_low_access = self.release_frames_batch(low_access_frames)
        else:
            released_low_access = 0
        
        # 3. 强制垃圾回收
        gc.collect()
        
        final_memory = self.current_memory_usage
        final_frames = len(self.frames)
        
        result = {
            "initial_memory_mb": initial_memory / 1024 / 1024,
            "final_memory_mb": final_memory / 1024 / 1024,
            "memory_freed_mb": (initial_memory - final_memory) / 1024 / 1024,
            "initial_frames": initial_frames,
            "final_frames": final_frames,
            "frames_released": initial_frames - final_frames,
            "old_frames_released": released_old,
            "low_access_frames_released": released_low_access
        }
        
        logger.info(f"内存优化完成: 释放 {result['memory_freed_mb']:.2f}MB, "
                   f"释放 {result['frames_released']} 帧")
        
        return result
    
    def is_memory_available(self, required_size: int) -> bool:
        """
        检查是否有足够的内存空间
        
        Args:
            required_size: 需要的内存大小（字节）
            
        Returns:
            bool: 是否有足够内存
        """
        return (self.current_memory_usage + required_size) <= self.max_buffer_size
    
    def get_available_memory(self) -> int:
        """
        获取可用内存大小
        
        Returns:
            int: 可用内存大小（字节）
        """
        return max(0, self.max_buffer_size - self.current_memory_usage)
    
    def __len__(self) -> int:
        """返回当前缓冲区中的帧数量"""
        return len(self.frames)
    
    def __contains__(self, frame_id: int) -> bool:
        """检查帧是否在缓冲区中"""
        return frame_id in self.frames
    
    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            self.clear_all_frames()
        except:
            pass