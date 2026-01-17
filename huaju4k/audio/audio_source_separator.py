"""
Audio Source Separator for Theater Enhancement

This module provides audio source separation capabilities using Spleeter
and fallback methods for separating vocals and accompaniment tracks.
"""

import os
import logging
import tempfile
from typing import Dict, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class AudioSourceSeparator:
    """
    Audio source separator with Spleeter and fallback methods.
    
    Provides professional-grade audio source separation for theater
    enhancement, with graceful fallback to simpler methods when
    advanced libraries are not available.
    """
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize audio source separator.
        
        Args:
            temp_dir: Temporary directory for processing files
        """
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "huaju4k_audio"
        self.temp_dir.mkdir(exist_ok=True)
        self.spleeter_available = self._check_spleeter_availability()
        
        logger.info(f"Audio source separator initialized (Spleeter: {self.spleeter_available})")
    
    def separate_audio_sources(self, audio_path: str) -> Dict[str, str]:
        """
        Separate audio into vocals and accompaniment tracks.
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Dictionary with 'vocals' and 'accompaniment' file paths
        """
        logger.info(f"Separating audio sources: {audio_path}")
        
        if self.spleeter_available:
            try:
                return self._separate_with_spleeter(audio_path)
            except Exception as e:
                logger.warning(f"Spleeter separation failed: {e}, using fallback")
        
        # Fallback to simple separation
        return self._simple_vocal_separation(audio_path)
    
    def _check_spleeter_availability(self) -> bool:
        """
        Check if Spleeter is available for use.
        
        Returns:
            True if Spleeter is available, False otherwise
        """
        try:
            from spleeter.separator import Separator
            return True
        except ImportError:
            logger.warning("Spleeter not available, will use fallback separation")
            return False
    
    def _separate_with_spleeter(self, audio_path: str) -> Dict[str, str]:
        """
        Separate audio using Spleeter (advanced method).
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Dictionary with separated track paths
        """
        from spleeter.separator import Separator
        
        # Initialize separator (vocals + accompaniment)
        separator = Separator('spleeter:2stems-16kHz')
        
        # Create output directory
        output_dir = self.temp_dir / "separated"
        output_dir.mkdir(exist_ok=True)
        
        # Perform separation
        separator.separate_to_file(audio_path, str(output_dir))
        
        # Build result paths
        audio_name = Path(audio_path).stem
        result = {
            'vocals': str(output_dir / audio_name / "vocals.wav"),
            'accompaniment': str(output_dir / audio_name / "accompaniment.wav")
        }
        
        # Verify output files exist
        for track_type, track_path in result.items():
            if not os.path.exists(track_path):
                raise RuntimeError(f"Spleeter failed to create {track_type} track: {track_path}")
        
        logger.info("Spleeter separation completed successfully")
        return result
    
    def _simple_vocal_separation(self, audio_path: str) -> Dict[str, str]:
        """
        Simple vocal separation using stereo channel processing (fallback method).
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Dictionary with separated track paths
        """
        try:
            import librosa
            import soundfile as sf
        except ImportError:
            logger.error("librosa and soundfile required for fallback separation")
            raise RuntimeError("Audio separation failed: no available method")
        
        logger.info("Using simple vocal separation (stereo channel processing)")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=44100, mono=False)
        
        if y.ndim == 1:
            # Mono audio - cannot separate effectively
            logger.warning("Mono audio detected, creating pseudo-separation")
            
            vocals_path = self.temp_dir / "vocals_mono.wav"
            accompaniment_path = self.temp_dir / "accompaniment_mono.wav"
            
            # Save original as vocals, attenuated version as accompaniment
            sf.write(str(vocals_path), y, sr)
            sf.write(str(accompaniment_path), y * 0.3, sr)
            
        else:
            # Stereo audio - use center channel extraction
            logger.info("Stereo audio detected, using center channel extraction")
            
            # Extract vocals (center channel difference)
            vocals = y[0] - y[1]  # L-R difference (rough vocal extraction)
            
            # Extract accompaniment (center channel average)
            accompaniment = (y[0] + y[1]) / 2  # L+R average (background music)
            
            # Save separated tracks
            vocals_path = self.temp_dir / "vocals_separated.wav"
            accompaniment_path = self.temp_dir / "accompaniment_separated.wav"
            
            sf.write(str(vocals_path), vocals, sr)
            sf.write(str(accompaniment_path), accompaniment, sr)
        
        result = {
            'vocals': str(vocals_path),
            'accompaniment': str(accompaniment_path)
        }
        
        logger.info("Simple vocal separation completed")
        return result
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """
        Check availability of audio processing dependencies.
        
        Returns:
            Dictionary mapping dependency names to availability status
        """
        dependencies = {}
        
        # Check librosa
        try:
            import librosa
            dependencies['librosa'] = True
        except ImportError:
            dependencies['librosa'] = False
        
        # Check soundfile
        try:
            import soundfile
            dependencies['soundfile'] = True
        except ImportError:
            dependencies['soundfile'] = False
        
        # Check Spleeter
        try:
            from spleeter.separator import Separator
            dependencies['spleeter'] = True
        except ImportError:
            dependencies['spleeter'] = False
        
        return dependencies
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("Audio separator temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup audio separator temp files: {e}")