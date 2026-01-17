"""
GPU Stage - 真实 GPU 超分增强模块

这是一个独立的 GPU 处理阶段，设计原则：
- 可插拔：失败时自动回退到 CPU
- 可验证：nvidia-smi 必须显示负载
- 最小侵入：不改动现有架构

使用方式：
    from huaju4k.gpu_stage import GPUVideoSuperResolver
    
    resolver = GPUVideoSuperResolver()
    resolver.enhance_video("input.mp4", "output_sr.mp4")
"""

from .gpu_super_resolver import GPUVideoSuperResolver
from .model_manager import GPUModelManager

__all__ = ['GPUVideoSuperResolver', 'GPUModelManager']
