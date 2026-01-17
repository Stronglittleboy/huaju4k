"""
Audio processing module for Huaju4K theater enhancement.

This module provides master-grade audio enhancement capabilities including
source separation, dialogue enhancement, and professional remixing.
"""

from .master_grade_enhancer import MasterGradeAudioEnhancer
from .audio_source_separator import AudioSourceSeparator

__all__ = [
    'MasterGradeAudioEnhancer',
    'AudioSourceSeparator'
]