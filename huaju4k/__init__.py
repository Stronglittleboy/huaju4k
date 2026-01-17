"""
huaju4k - 戏剧视频增强工具

专为戏剧内容优化的命令行视频增强工具。
将1080p/720p戏剧视频转换为4K分辨率，并提供专业的音频优化。

Features:
- AI驱动的4K视频升级
- 戏剧专用音频增强
- GPU加速支持（NVIDIA优化）
- 智能内存管理
- 批量处理功能
- 可配置的质量预设
"""

__version__ = "0.2.0"
__author__ = "huaju4k开发团队"
__description__ = "戏剧视频增强工具"
__license__ = "MIT"
__url__ = "https://github.com/huaju4k/huaju4k"

from .core.interfaces import (
    VideoProcessor,
    AudioEnhancer,
    MemoryManager,
    ProgressTracker,
    ConfigurationManager
)

from .core.integrated_video_processor import (
    IntegratedVideoProcessor,
    create_integrated_processor
)

from .media.ffmpeg_media_controller import (
    FFmpegMediaController,
    check_ffmpeg_availability
)

from .models.data_models import (
    VideoInfo,
    ProcessingStrategy,
    TheaterFeatures,
    ProcessResult,
    TileConfiguration
)

__all__ = [
    "VideoProcessor",
    "AudioEnhancer", 
    "MemoryManager",
    "ProgressTracker",
    "ConfigurationManager",
    "IntegratedVideoProcessor",
    "create_integrated_processor",
    "FFmpegMediaController",
    "check_ffmpeg_availability",
    "VideoInfo",
    "ProcessingStrategy",
    "TheaterFeatures",
    "ProcessResult",
    "TileConfiguration"
]