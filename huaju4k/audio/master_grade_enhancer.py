"""
Master Grade Audio Enhancer for Theater Enhancement

This module provides professional-grade audio enhancement capabilities
including source separation, dialogue enhancement, and master-grade remixing.
"""

import os
import logging
import tempfile
import subprocess
from typing import Dict, Optional, Any
from pathlib import Path

import numpy as np

from ..models.data_models import EnhancementStrategy, AudioResult
from .audio_source_separator import AudioSourceSeparator

logger = logging.getLogger(__name__)


class MasterGradeAudioEnhancer:
    """
    Master-grade audio enhancer for theater video enhancement.
    
    Provides professional audio processing including source separation,
    dialogue enhancement, music processing, and master-grade remixing
    with graceful fallback when advanced libraries are unavailable.
    """
    
    def __init__(self):
        """Initialize master grade audio enhancer."""
        self.temp_dir = Path(tempfile.gettempdir()) / "huaju4k_audio"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize audio source separator
        self.source_separator = AudioSourceSeparator(self.temp_dir)
        
        # Check dependency availability
        self.dependencies = self.check_dependencies()
        
        logger.info(f"Master grade audio enhancer initialized")
        logger.info(f"Available dependencies: {[k for k, v in self.dependencies.items() if v]}")
    
    def enhance_audio(self, video_path: str, 
                     strategy: EnhancementStrategy) -> AudioResult:
        """
        Master-grade audio enhancement main pipeline.
        
        Args:
            video_path: Path to input video file
            strategy: Enhancement strategy configuration
            
        Returns:
            AudioResult with processing outcome
        """
        logger.info(f"Starting master-grade audio enhancement: {video_path}")
        
        try:
            # Check if source separation is enabled in strategy
            if not hasattr(strategy.audio_strategy, 'source_separation_enabled'):
                # Add source separation flag if not present
                strategy.audio_strategy.source_separation_enabled = True
            
            if not strategy.audio_strategy.source_separation_enabled:
                logger.info("Source separation disabled, using simple enhancement")
                return self._simple_audio_enhancement(video_path, strategy)
            
            # 1. Audio track extraction
            logger.info("Step 1: Extracting audio track with FFmpeg")
            audio_path = self._extract_audio_with_ffmpeg(video_path)
            
            # 2. Audio source separation
            logger.info("Step 2: Separating audio sources")
            separated_tracks = self.source_separator.separate_audio_sources(audio_path)
            
            # 3. Track-specific processing
            logger.info("Step 3: Processing separated tracks")
            enhanced_dialogue = self._enhance_dialogue(
                separated_tracks['vocals'], 
                getattr(strategy.audio_strategy, 'dialogue_enhancement', 0.7)
            )
            
            enhanced_music = self._process_music(
                separated_tracks['accompaniment'],
                getattr(strategy.audio_strategy, 'music_processing', 0.5)
            )
            
            # 4. Master-grade remixing
            logger.info("Step 4: Master-grade remixing")
            master_settings = getattr(strategy.audio_strategy, 'master_settings', {
                'dialogue_gain': 0,
                'music_gain': -6
            })
            
            master_audio_path = self._master_grade_remix(
                enhanced_dialogue, enhanced_music, master_settings
            )
            
            # 5. Calculate quality metrics
            quality_improvements = self._calculate_audio_quality_metrics(master_audio_path)
            
            logger.info("Master-grade audio enhancement completed successfully")
            return AudioResult(
                success=True,
                output_path=master_audio_path,
                quality_improvements=quality_improvements
            )
            
        except Exception as e:
            logger.error(f"Master grade audio enhancement failed: {e}")
            return AudioResult(success=False, error=str(e))
    
    def _extract_audio_with_ffmpeg(self, video_path: str) -> str:
        """
        Extract audio track using FFmpeg.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Path to extracted audio file
        """
        audio_path = self.temp_dir / "extracted_audio.wav"
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video stream
            '-acodec', 'pcm_s16le',  # 16-bit PCM encoding
            '-ar', '44100',  # 44.1kHz sample rate
            '-ac', '2',  # Stereo
            '-y',  # Overwrite output file
            str(audio_path)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Audio extracted successfully: {audio_path}")
            return str(audio_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg audio extraction failed: {e.stderr}")
            raise RuntimeError(f"Audio extraction failed: {e.stderr}")
        except FileNotFoundError:
            logger.error("FFmpeg not found in system PATH")
            raise RuntimeError("FFmpeg not available for audio extraction")
    
    def _enhance_dialogue(self, vocals_path: str, enhancement_strength: float) -> str:
        """
        Enhance dialogue track using advanced audio processing.
        
        Args:
            vocals_path: Path to vocals/dialogue track
            enhancement_strength: Enhancement strength (0.0-1.0)
            
        Returns:
            Path to enhanced dialogue track
        """
        logger.info(f"Enhancing dialogue with strength: {enhancement_strength}")
        
        try:
            import librosa
            import soundfile as sf
            
            # Load audio
            y, sr = librosa.load(vocals_path, sr=44100)
            
            # 1. Noise reduction (if noisereduce available)
            if self.dependencies.get('noisereduce', False):
                import noisereduce as nr
                logger.info("Applying noise reduction")
                y = nr.reduce_noise(
                    y=y, sr=sr, 
                    prop_decrease=enhancement_strength * 0.8
                )
            else:
                logger.info("Noise reduction not available, skipping")
            
            # 2. Pre-emphasis (enhance high frequency clarity)
            if enhancement_strength > 0.5:
                logger.info("Applying pre-emphasis for high frequency enhancement")
                y = librosa.effects.preemphasis(y, coef=0.97)
            
            # 3. Dynamic range compression (soft limiter)
            logger.info("Applying dynamic range compression")
            threshold = 0.8
            ratio = 4.0
            y = np.where(
                np.abs(y) > threshold,
                np.sign(y) * (threshold + (np.abs(y) - threshold) / ratio),
                y
            )
            
            # Save enhanced dialogue
            enhanced_path = self.temp_dir / "enhanced_dialogue.wav"
            sf.write(str(enhanced_path), y, sr)
            
            logger.info(f"Dialogue enhanced: {enhanced_path}")
            return str(enhanced_path)
            
        except ImportError as e:
            logger.error(f"Audio enhancement libraries not available: {e}")
            logger.info("Returning original vocals track")
            return vocals_path  # Return original file
    
    def _process_music(self, music_path: str, processing_strength: float) -> str:
        """
        Process music/accompaniment track.
        
        Args:
            music_path: Path to music/accompaniment track
            processing_strength: Processing strength (0.0-1.0)
            
        Returns:
            Path to processed music track
        """
        logger.info(f"Processing music with strength: {processing_strength}")
        
        try:
            import librosa
            import soundfile as sf
            
            # Load audio
            y, sr = librosa.load(music_path, sr=44100)
            
            # Apply gentle processing to preserve musicality
            if processing_strength > 0.3:
                # Light compression for music
                threshold = 0.9
                ratio = 2.0
                y = np.where(
                    np.abs(y) > threshold,
                    np.sign(y) * (threshold + (np.abs(y) - threshold) / ratio),
                    y
                )
            
            # Save processed music
            processed_path = self.temp_dir / "processed_music.wav"
            sf.write(str(processed_path), y, sr)
            
            logger.info(f"Music processed: {processed_path}")
            return str(processed_path)
            
        except ImportError as e:
            logger.error(f"Audio processing libraries not available: {e}")
            logger.info("Returning original music track")
            return music_path  # Return original file
    
    def _master_grade_remix(self, dialogue_path: str, music_path: str, 
                           master_settings: Dict[str, Any]) -> str:
        """
        Master-grade remixing using professional audio processing.
        
        Args:
            dialogue_path: Path to enhanced dialogue track
            music_path: Path to processed music track
            master_settings: Master remixing settings
            
        Returns:
            Path to final master audio
        """
        logger.info("Performing master-grade remixing")
        
        try:
            from pydub import AudioSegment
            
            # Load audio tracks
            dialogue = AudioSegment.from_wav(dialogue_path)
            music = AudioSegment.from_wav(music_path)
            
            # Apply volume balancing
            dialogue_gain = master_settings.get('dialogue_gain', 0)  # dB
            music_gain = master_settings.get('music_gain', -6)       # dB
            
            logger.info(f"Applying gains - Dialogue: {dialogue_gain}dB, Music: {music_gain}dB")
            dialogue = dialogue + dialogue_gain
            music = music + music_gain
            
            # Length alignment
            max_length = max(len(dialogue), len(music))
            dialogue = dialogue[:max_length]
            music = music[:max_length]
            
            # Overlay mixing
            logger.info("Mixing dialogue and music tracks")
            master_audio = dialogue.overlay(music)
            
            # Master-grade processing
            logger.info("Applying master-grade normalization")
            master_audio = master_audio.normalize(headroom=1.0)  # Prevent clipping
            
            # Export final audio
            master_path = self.temp_dir / "master_audio.wav"
            master_audio.export(str(master_path), format="wav")
            
            logger.info(f"Master audio created: {master_path}")
            return str(master_path)
            
        except ImportError as e:
            logger.error(f"pydub not available: {e}")
            logger.info("Returning dialogue track as fallback")
            return dialogue_path  # Return dialogue audio as fallback
    
    def _simple_audio_enhancement(self, video_path: str, 
                                 strategy: EnhancementStrategy) -> AudioResult:
        """
        Simple audio enhancement without source separation.
        
        Args:
            video_path: Path to input video file
            strategy: Enhancement strategy
            
        Returns:
            AudioResult with simple enhancement outcome
        """
        logger.info("Performing simple audio enhancement")
        
        try:
            # Extract audio
            audio_path = self._extract_audio_with_ffmpeg(video_path)
            
            # Apply basic enhancement
            enhanced_path = self._enhance_dialogue(audio_path, 0.5)
            
            return AudioResult(
                success=True,
                output_path=enhanced_path,
                quality_improvements={'simple_enhancement': 1.0}
            )
            
        except Exception as e:
            logger.error(f"Simple audio enhancement failed: {e}")
            return AudioResult(success=False, error=str(e))
    
    def _calculate_audio_quality_metrics(self, audio_path: str) -> Dict[str, float]:
        """
        Calculate audio quality improvement metrics.
        
        Args:
            audio_path: Path to processed audio file
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        try:
            import librosa
            
            # Load audio for analysis
            y, sr = librosa.load(audio_path, sr=44100)
            
            # Calculate basic metrics
            metrics['rms_energy'] = float(np.sqrt(np.mean(y**2)))
            metrics['peak_amplitude'] = float(np.max(np.abs(y)))
            metrics['dynamic_range'] = float(20 * np.log10(np.max(np.abs(y)) / (np.mean(np.abs(y)) + 1e-10)))
            
            # Spectral metrics
            stft = librosa.stft(y)
            spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(stft))
            metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            
            logger.info(f"Audio quality metrics calculated: {metrics}")
            
        except ImportError:
            logger.warning("librosa not available for quality metrics")
            metrics = {'basic_enhancement': 1.0}
        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {e}")
            metrics = {'processing_completed': 1.0}
        
        return metrics
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """
        Check availability of all audio processing dependencies.
        
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
        
        # Check noisereduce
        try:
            import noisereduce
            dependencies['noisereduce'] = True
        except ImportError:
            dependencies['noisereduce'] = False
        
        # Check pydub
        try:
            from pydub import AudioSegment
            dependencies['pydub'] = True
        except ImportError:
            dependencies['pydub'] = False
        
        # Check Spleeter
        try:
            from spleeter.separator import Separator
            dependencies['spleeter'] = True
        except ImportError:
            dependencies['spleeter'] = False
        
        # Check FFmpeg availability
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            dependencies['ffmpeg'] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            dependencies['ffmpeg'] = False
        
        return dependencies
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("Master grade audio enhancer temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup master grade audio enhancer temp files: {e}")
        
        # Also cleanup source separator temp files
        self.source_separator.cleanup_temp_files()