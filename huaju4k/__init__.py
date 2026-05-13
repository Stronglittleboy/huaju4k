"""
huaju4k - 话剧视频 4K 增强工具

专为话剧录像定制的 4K 增强工具。
针对固定机位、舞台灯光、对白为核心三大特征，
在 6GB 显存消费级显卡上实现高质量超分。
"""

__version__ = "1.0.0"
__author__ = "huaju4k"
__license__ = "MIT"

from .models.data_models import (
    VideoInfo,
    SceneSegment,
    StructureFeatures,
    EnhancementStrategy,
    ProcessingStrategy,
    ProcessResult,
    AudioResult,
    CheckpointData,
    TileConfiguration,
    TheaterFeatures,
    ResourceStatus,
)

__all__ = [
    "VideoInfo",
    "SceneSegment",
    "StructureFeatures",
    "EnhancementStrategy",
    "ProcessingStrategy",
    "ProcessResult",
    "AudioResult",
    "CheckpointData",
    "TileConfiguration",
    "TheaterFeatures",
    "ResourceStatus",
]
