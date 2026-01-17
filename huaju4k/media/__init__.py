"""
Media processing layer for huaju4k video enhancement.

This module provides FFmpeg-based media control and orchestration,
replacing OpenCV-based video writing with professional media pipeline.
"""

from .ffmpeg_media_controller import FFmpegMediaController

__all__ = ["FFmpegMediaController"]