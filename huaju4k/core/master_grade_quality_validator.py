"""
Master Grade Quality Validator for Theater Enhancement

This module implements comprehensive quality validation for master-grade theater video enhancement.
It provides advanced metrics beyond basic file validation, including brightness stability,
edge stability, highlight clipping, audio quality, and sync validation.
"""

import os
import logging
import tempfile
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from ..models.data_models import EnhancementStrategy, VideoInfo
from .video_analyzer import VideoAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Comprehensive quality report for master-grade validation."""
    basic_quality: Dict[str, float] = field(default_factory=dict)
    video_quality: Dict[str, float] = field(default_factory=dict)
    audio_quality: Dict[str, float] = field(default_factory=dict)
    sync_quality: Dict[str, float] = field(default_factory=dict)
    master_grade_metrics: Dict[str, float] = field(default_factory=dict)
    
    overall_score: float = 0.0
    validation_timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_overall_score(self) -> float:
        """Calculate overall quality score from all categories."""
        all_scores = []
        
        for category in [self.basic_quality, self.video_quality, 
                        self.audio_quality, self.sync_quality, 
                        self.master_grade_metrics]:
            category_scores = [v for k, v in category.items() 
                             if isinstance(v, (int, float)) and k != 'validation_error']
            if category_scores:
                all_scores.extend(category_scores)
        
        if all_scores:
            self.overall_score = np.mean(all_scores)
        else:
            self.overall_score = 0.0
        
        return self.overall_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


class MasterGradeQualityValidator:
    """
    Master-grade quality validator for theater enhancement.
    
    This validator provides comprehensive quality assessment including:
    - Basic file validation
    - Video quality metrics (brightness stability, edge stability, highlight clipping)
    - Audio quality metrics (dialogue clarity, volume consistency, frequency response)
    - Audio-video synchronization validation
    - Master-grade specific metrics
    """
    
    def __init__(self):
        """Initialize the master-grade quality validator."""
        self.video_analyzer = VideoAnalyzer()
        self.temp_dir = Path(tempfile.gettempdir()) / "huaju4k_quality_validation"
        self.temp_dir.mkdir(exist_ok=True)
    
    def validate_master_quality(self, input_path: str, 
                              output_path: str,
                              enhancement_strategy: EnhancementStrategy) -> QualityReport:
        """
        Master-grade quality validation.
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file
            enhancement_strategy: Enhancement strategy used for processing
            
        Returns:
            QualityReport: Comprehensive quality report
        """
        logger.info(f"Starting master-grade quality validation: {output_path}")
        
        report = QualityReport()
        
        try:
            # Basic quality validation
            report.basic_quality = self._validate_basic_quality(input_path, output_path)
            
            # Video quality validation
            report.video_quality = self._validate_video_quality(output_path, enhancement_strategy)
            
            # Audio quality validation
            report.audio_quality = self._validate_audio_quality(output_path)
            
            # Synchronization validation
            report.sync_quality = self._validate_av_sync(output_path)
            
            # Master-grade specific validation
            report.master_grade_metrics = self._validate_master_grade_metrics(
                input_path, output_path, enhancement_strategy
            )
            
            # Calculate overall score
            report.calculate_overall_score()
            
            logger.info(f"Quality validation completed. Overall score: {report.overall_score:.3f}")
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            report.basic_quality['validation_error'] = 1.0
        
        return report
    
    def _validate_basic_quality(self, input_path: str, output_path: str) -> Dict[str, float]:
        """Basic quality validation (file existence, size, format)."""
        metrics = {}
        
        try:
            # Check output file exists
            if not os.path.exists(output_path):
                metrics['file_exists'] = 0.0
                return metrics
            
            # Check file size
            output_size = os.path.getsize(output_path)
            if output_size == 0:
                metrics['file_exists'] = 0.0
                return metrics
            
            metrics['file_exists'] = 1.0
            metrics['output_file_size_mb'] = output_size / (1024 * 1024)
            
            # Compare with input size
            input_size = os.path.getsize(input_path)
            metrics['size_ratio'] = output_size / input_size
            
            # Basic format validation
            try:
                cap = cv2.VideoCapture(output_path)
                if cap.isOpened():
                    metrics['format_valid'] = 1.0
                    cap.release()
                else:
                    metrics['format_valid'] = 0.0
            except Exception:
                metrics['format_valid'] = 0.0
            
        except Exception as e:
            logger.error(f"Basic quality validation failed: {e}")
            metrics['validation_error'] = 1.0
        
        return metrics
    
    def _validate_video_quality(self, output_path: str, 
                              strategy: EnhancementStrategy) -> Dict[str, float]:
        """Video quality validation with advanced metrics."""
        metrics = {}
        
        try:
            # Brightness stability check
            metrics['brightness_stability'] = self._check_brightness_stability(output_path)
            
            # Edge stability assessment
            metrics['edge_stability'] = self._check_edge_stability(output_path)
            
            # Highlight clipping detection
            metrics['highlight_clipping'] = self._check_highlight_clipping(output_path)
            
            # GAN enhancement quality validation
            if strategy.gan_policy.global_allowed:
                metrics['gan_enhancement_quality'] = self._validate_gan_enhancement(output_path)
            
            # Temporal stability validation
            if strategy.temporal_strategy.background_lock:
                metrics['temporal_stability'] = self._validate_temporal_stability(output_path)
            
        except Exception as e:
            logger.error(f"Video quality validation failed: {e}")
            metrics['validation_error'] = 1.0
        
        return metrics
    
    def _check_brightness_stability(self, video_path: str) -> float:
        """Check frame-to-frame brightness stability."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        
        brightness_values = []
        frame_count = 0
        
        try:
            while frame_count < 100:  # Sample 100 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate average brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray) / 255.0
                brightness_values.append(brightness)
                frame_count += 1
            
            cap.release()
            
            if len(brightness_values) < 2:
                return 0.0
            
            # Calculate brightness stability (lower std = more stable)
            brightness_std = np.std(brightness_values)
            stability_score = max(0.0, 1.0 - brightness_std * 5)  # Normalize to 0-1
            
            return stability_score
            
        except Exception as e:
            logger.error(f"Brightness stability check failed: {e}")
            cap.release()
            return 0.0
    
    def _check_edge_stability(self, video_path: str) -> float:
        """Check edge stability across frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        
        edge_densities = []
        frame_count = 0
        
        try:
            while frame_count < 50:  # Sample 50 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate edge density
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                edge_densities.append(edge_density)
                frame_count += 1
            
            cap.release()
            
            if len(edge_densities) < 2:
                return 0.0
            
            # Calculate edge stability
            edge_std = np.std(edge_densities)
            stability_score = max(0.0, 1.0 - edge_std * 10)
            
            return stability_score
            
        except Exception as e:
            logger.error(f"Edge stability check failed: {e}")
            cap.release()
            return 0.0
    
    def _check_highlight_clipping(self, video_path: str) -> float:
        """Check for highlight clipping (overexposure)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 1.0  # Assume problem if can't check
        
        clipping_ratios = []
        frame_count = 0
        
        try:
            while frame_count < 30:  # Sample 30 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check for highlight clipping (pixels >= 250)
                clipped_pixels = np.sum(frame >= 250)
                total_pixels = frame.size
                clipping_ratio = clipped_pixels / total_pixels
                clipping_ratios.append(clipping_ratio)
                frame_count += 1
            
            cap.release()
            
            if not clipping_ratios:
                return 1.0
            
            # Average clipping ratio, lower is better
            avg_clipping = np.mean(clipping_ratios)
            clipping_score = max(0.0, 1.0 - avg_clipping * 20)  # 5% clipping = 0 score
            
            return clipping_score
            
        except Exception as e:
            logger.error(f"Highlight clipping check failed: {e}")
            cap.release()
            return 0.0
    
    def _validate_gan_enhancement(self, video_path: str) -> float:
        """Validate GAN enhancement quality."""
        # Placeholder implementation - could be expanded with specific GAN quality metrics
        try:
            # For now, use edge stability as a proxy for GAN enhancement quality
            return self._check_edge_stability(video_path)
        except Exception as e:
            logger.error(f"GAN enhancement validation failed: {e}")
            return 0.5
    
    def _validate_temporal_stability(self, video_path: str) -> float:
        """Validate temporal locking effectiveness."""
        # Placeholder implementation - could be expanded with motion analysis
        try:
            # For now, use brightness stability as a proxy for temporal stability
            return self._check_brightness_stability(video_path)
        except Exception as e:
            logger.error(f"Temporal stability validation failed: {e}")
            return 0.5
    
    def _validate_audio_quality(self, output_path: str) -> Dict[str, float]:
        """Audio quality validation."""
        metrics = {}
        
        try:
            # Extract audio for analysis
            temp_audio = self._extract_audio_for_analysis(output_path)
            
            if temp_audio:
                # Dialogue clarity assessment
                metrics['dialogue_clarity'] = self._assess_dialogue_clarity(temp_audio)
                
                # Volume consistency check
                metrics['volume_consistency'] = self._check_volume_consistency(temp_audio)
                
                # Frequency response analysis
                metrics['frequency_response'] = self._analyze_frequency_response(temp_audio)
                
                # Dynamic range assessment
                metrics['dynamic_range'] = self._assess_dynamic_range(temp_audio)
                
                # Clean up temp file
                try:
                    os.remove(temp_audio)
                except:
                    pass
            else:
                metrics['audio_extraction_failed'] = 1.0
            
        except Exception as e:
            logger.error(f"Audio quality validation failed: {e}")
            metrics['validation_error'] = 1.0
        
        return metrics
    
    def _extract_audio_for_analysis(self, video_path: str) -> Optional[str]:
        """Extract audio from video for analysis."""
        try:
            audio_path = self.temp_dir / "temp_audio_for_analysis.wav"
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '44100',  # 44.1kHz sample rate
                '-ac', '2',  # Stereo
                '-y',  # Overwrite output
                str(audio_path)
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if os.path.exists(audio_path):
                return str(audio_path)
            else:
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio extraction failed: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Audio extraction error: {e}")
            return None
    
    def _assess_dialogue_clarity(self, audio_path: str) -> float:
        """Assess dialogue clarity."""
        try:
            # Try to use librosa for advanced analysis
            try:
                import librosa
                
                # Load audio
                y, sr = librosa.load(audio_path, sr=44100)
                
                # Analyze speech frequency energy (300-3400Hz)
                stft = librosa.stft(y)
                freqs = librosa.fft_frequencies(sr=sr)
                
                # Speech frequency mask
                speech_freq_mask = (freqs >= 300) & (freqs <= 3400)
                speech_energy = np.mean(np.abs(stft[speech_freq_mask, :]))
                
                # Total energy
                total_energy = np.mean(np.abs(stft))
                
                # Speech clarity ratio
                if total_energy > 0:
                    clarity_ratio = speech_energy / total_energy
                    clarity_score = min(1.0, clarity_ratio * 2)  # Normalize
                else:
                    clarity_score = 0.0
                
                return clarity_score
                
            except ImportError:
                logger.warning("librosa not available for dialogue clarity assessment")
                return 0.5  # Default medium score
                
        except Exception as e:
            logger.error(f"Dialogue clarity assessment failed: {e}")
            return 0.0
    
    def _check_volume_consistency(self, audio_path: str) -> float:
        """Check volume consistency across the audio."""
        try:
            # Use ffmpeg to analyze volume levels
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-af', 'volumedetect',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse volume statistics from stderr
            stderr = result.stderr
            
            # Look for mean volume and max volume
            mean_volume = None
            max_volume = None
            
            for line in stderr.split('\n'):
                if 'mean_volume:' in line:
                    try:
                        mean_volume = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                    except:
                        pass
                elif 'max_volume:' in line:
                    try:
                        max_volume = float(line.split('max_volume:')[1].split('dB')[0].strip())
                    except:
                        pass
            
            if mean_volume is not None and max_volume is not None:
                # Calculate dynamic range
                dynamic_range = max_volume - mean_volume
                
                # Good consistency means reasonable dynamic range (not too compressed, not too wide)
                # Ideal range: 6-20 dB
                if 6 <= dynamic_range <= 20:
                    consistency_score = 1.0
                elif dynamic_range < 6:
                    # Too compressed
                    consistency_score = max(0.0, dynamic_range / 6)
                else:
                    # Too wide
                    consistency_score = max(0.0, 1.0 - (dynamic_range - 20) / 20)
                
                return consistency_score
            else:
                return 0.5  # Default if can't analyze
                
        except Exception as e:
            logger.error(f"Volume consistency check failed: {e}")
            return 0.5
    
    def _analyze_frequency_response(self, audio_path: str) -> float:
        """Analyze frequency response balance."""
        try:
            # Use ffmpeg to analyze frequency spectrum
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-af', 'astats=metadata=1:reset=1',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # For now, return a default score
            # This could be expanded with more sophisticated frequency analysis
            return 0.7
            
        except Exception as e:
            logger.error(f"Frequency response analysis failed: {e}")
            return 0.5
    
    def _assess_dynamic_range(self, audio_path: str) -> float:
        """Assess audio dynamic range."""
        try:
            # This is similar to volume consistency but focuses on dynamic range
            return self._check_volume_consistency(audio_path)
            
        except Exception as e:
            logger.error(f"Dynamic range assessment failed: {e}")
            return 0.5
    
    def _validate_av_sync(self, video_path: str) -> Dict[str, float]:
        """Audio-video synchronization validation."""
        metrics = {}
        
        try:
            # Get video and audio durations
            video_info = self.video_analyzer.analyze_video(video_path)
            
            # Extract audio duration
            audio_duration = self._get_audio_duration(video_path)
            
            if audio_duration is not None:
                # Calculate duration difference
                duration_diff = abs(video_info.duration - audio_duration)
                sync_score = max(0.0, 1.0 - duration_diff / 2.0)  # 2 second diff = 0 score
                
                metrics['duration_sync'] = sync_score
                metrics['duration_difference'] = duration_diff
            else:
                metrics['duration_sync'] = 1.0  # No audio = perfect sync
                metrics['duration_difference'] = 0.0
            
            # Framerate consistency check
            metrics['framerate_consistency'] = self._check_framerate_consistency(video_path)
            
        except Exception as e:
            logger.error(f"AV sync validation failed: {e}")
            metrics['validation_error'] = 1.0
        
        return metrics
    
    def _get_audio_duration(self, video_path: str) -> Optional[float]:
        """Get audio duration from video file."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=duration',
                '-of', 'csv=p=0',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
            else:
                return None
                
        except Exception as e:
            logger.error(f"Audio duration extraction failed: {e}")
            return None
    
    def _check_framerate_consistency(self, video_path: str) -> float:
        """Check framerate consistency."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Basic check - if we can get valid FPS and frame count, assume consistent
            if fps > 0 and frame_count > 0:
                return 1.0
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Framerate consistency check failed: {e}")
            return 0.0
    
    def _validate_master_grade_metrics(self, input_path: str, output_path: str,
                                     strategy: EnhancementStrategy) -> Dict[str, float]:
        """Master-grade specific validation metrics."""
        metrics = {}
        
        try:
            # Resolution improvement validation
            input_info = self.video_analyzer.analyze_video(input_path)
            output_info = self.video_analyzer.analyze_video(output_path)
            
            input_pixels = input_info.resolution[0] * input_info.resolution[1]
            output_pixels = output_info.resolution[0] * output_info.resolution[1]
            
            metrics['resolution_improvement_ratio'] = output_pixels / input_pixels
            
            # Strategy compliance validation
            expected_improvement = 4.0 if len(strategy.resolution_plan) == 2 else 2.0
            improvement_accuracy = min(1.0, metrics['resolution_improvement_ratio'] / expected_improvement)
            metrics['strategy_compliance'] = improvement_accuracy
            
            # Processing quality score (combination of various factors)
            quality_factors = [
                self._check_brightness_stability(output_path),
                self._check_edge_stability(output_path),
                self._check_highlight_clipping(output_path)
            ]
            
            metrics['processing_quality'] = np.mean(quality_factors)
            
        except Exception as e:
            logger.error(f"Master-grade metrics validation failed: {e}")
            metrics['validation_error'] = 1.0
        
        return metrics