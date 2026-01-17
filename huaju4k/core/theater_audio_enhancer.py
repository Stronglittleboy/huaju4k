"""
Theater Audio Enhancement System for huaju4k.

This module provides specialized audio processing for theater drama content,
including venue-specific optimization, noise reduction, and dialogue enhancement.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# Optional imports with fallbacks
try:
    import librosa
    import soundfile as sf
    from scipy import signal
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    librosa = None
    sf = None
    signal = None

from ..models.data_models import TheaterFeatures, AudioResult
from ..configs.config_manager import ConfigManager
from .interfaces import AudioEnhancer
from .dialogue_enhancer import AdvancedDialogueEnhancer
from .spatial_audio_optimizer import SpatialAudioOptimizer

logger = logging.getLogger(__name__)


class TheaterAudioEnhancer(AudioEnhancer):
    """
    Theater-specific audio enhancement system.
    
    Provides specialized audio processing optimized for theater drama content,
    including venue size detection, noise reduction, dialogue enhancement,
    and spatial audio optimization.
    """
    
    def __init__(self, theater_preset: str = "medium", config: Optional[Dict] = None):
        """
        Initialize theater audio enhancer.
        
        Args:
            theater_preset: Theater size preset (small, medium, large)
            config: Optional configuration dictionary
        """
        if not HAS_AUDIO_LIBS:
            raise ImportError(
                "Audio processing libraries not available. "
                "Please install: librosa, soundfile, scipy"
            )
        
        self.theater_preset = theater_preset
        self.config = config or self._load_default_config()
        
        # Initialize processing components
        self.noise_reducer = SpectralGateNoiseReducer(self.config.get('noise_reduction', {}))
        self.dialogue_enhancer = AdvancedDialogueEnhancer(self.config.get('dialogue_enhancement', {}))
        self.spatial_optimizer = SpatialAudioOptimizer(self.config.get('spatial_audio', {}))
        
        logger.info(f"TheaterAudioEnhancer initialized with preset: {theater_preset}")
    
    def analyze_theater_characteristics(self, audio_path: str) -> TheaterFeatures:
        """
        Analyze theater-specific audio characteristics from file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            TheaterFeatures object containing analysis results
        """
        try:
            # Load audio file
            audio, sr = self._load_audio(audio_path)
            if audio is None:
                return self._get_default_theater_features()
            
            # Analyze characteristics
            return self._analyze_theater_characteristics(audio, sr)
            
        except Exception as e:
            logger.error(f"Theater characteristics analysis failed for {audio_path}: {e}")
            return self._get_default_theater_features()
    
    def synchronize_with_video(self, audio_path: str, output_path: str,
                              video_fps: float, target_delay_ms: float = 0.0) -> AudioResult:
        """
        Synchronize audio with video timing.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for synchronized audio output
            video_fps: Video frame rate for synchronization
            target_delay_ms: Target delay in milliseconds
            
        Returns:
            AudioResult containing synchronization outcome
        """
        start_time = time.time()
        
        try:
            logger.info(f"Synchronizing audio with video: {audio_path}")
            
            # Load audio file
            audio, sr = self._load_audio(audio_path)
            if audio is None:
                return AudioResult(
                    success=False,
                    error=f"Failed to load audio file: {audio_path}"
                )
            
            # Apply synchronization
            synchronized_audio = self.spatial_optimizer.synchronize_audio_video(
                audio, video_fps, sr, target_delay_ms
            )
            
            # Save synchronized audio
            success = self._save_audio(synchronized_audio, sr, output_path)
            if not success:
                return AudioResult(
                    success=False,
                    error=f"Failed to save synchronized audio: {output_path}"
                )
            
            processing_time = time.time() - start_time
            
            return AudioResult(
                success=True,
                output_path=output_path,
                processing_time=processing_time,
                quality_improvements={"sync_correction_ms": target_delay_ms}
            )
            
        except Exception as e:
            logger.error(f"Audio-video synchronization failed: {e}")
            return AudioResult(
                success=False,
                error=str(e)
            )
    
    def enhance_with_video_sync(self, audio_path: str, output_path: str,
                               video_fps: float, target_delay_ms: float = 0.0,
                               theater_preset: str = None) -> AudioResult:
        """
        Apply theater enhancement with video synchronization.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for output audio file
            video_fps: Video frame rate for synchronization
            target_delay_ms: Target delay in milliseconds
            theater_preset: Optional theater size preset override
            
        Returns:
            AudioResult containing enhancement and sync outcome
        """
        start_time = time.time()
        preset = theater_preset or self.theater_preset
        
        try:
            logger.info(f"Starting enhanced audio processing with sync: {audio_path}")
            
            # Load audio file
            audio, sr = self._load_audio(audio_path)
            if audio is None:
                return AudioResult(
                    success=False,
                    error=f"Failed to load audio file: {audio_path}"
                )
            
            # Analyze theater characteristics
            theater_features = self._analyze_theater_characteristics(audio, sr)
            logger.debug(f"Detected theater features: {theater_features}")
            
            # Apply venue-specific optimization (pass sample rate)
            enhanced_audio = self._apply_venue_optimization_with_sr(audio, theater_features, sr)
            
            # Apply audio-video synchronization
            if abs(target_delay_ms) >= 1.0:
                enhanced_audio = self.spatial_optimizer.synchronize_audio_video(
                    enhanced_audio, video_fps, sr, target_delay_ms
                )
            
            # Save enhanced and synchronized audio
            success = self._save_audio(enhanced_audio, sr, output_path)
            if not success:
                return AudioResult(
                    success=False,
                    error=f"Failed to save enhanced audio: {output_path}"
                )
            
            # Calculate quality improvements
            quality_metrics = self._calculate_quality_metrics(audio, enhanced_audio, sr)
            quality_metrics["sync_correction_ms"] = target_delay_ms
            
            processing_time = time.time() - start_time
            
            return AudioResult(
                success=True,
                output_path=output_path,
                processing_time=processing_time,
                quality_improvements=quality_metrics,
                theater_features=theater_features
            )
            
        except Exception as e:
            logger.error(f"Enhanced audio processing with sync failed: {e}")
            return AudioResult(
                success=False,
                error=str(e)
            )
    
    def enhance(self, audio_path: str, output_path: str, 
                theater_preset: str = None) -> AudioResult:
        """
        Apply theater-specific audio enhancement.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for output audio file
            theater_preset: Optional theater size preset override
            
        Returns:
            AudioResult containing enhancement outcome
        """
        # Call enhanced processing without video sync
        return self.enhance_with_video_sync(
            audio_path, output_path, 25.0, 0.0, theater_preset
        )
    
    def _analyze_theater_characteristics(self, audio: np.ndarray, sr: int) -> TheaterFeatures:
        """
        Analyze theater-specific audio characteristics.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            
        Returns:
            TheaterFeatures containing venue characteristics
        """
        try:
            # Detect venue size based on reverb characteristics
            venue_size = self._detect_venue_size(audio, sr)
            
            # Calculate reverb time (RT60)
            reverb_time = self._calculate_reverb_time(audio, sr)
            
            # Analyze noise floor
            noise_floor = self._analyze_noise_floor(audio)
            
            # Detect dialogue frequency range
            dialogue_freq_range = self._detect_dialogue_frequency_range(audio, sr)
            
            # Calculate dynamic range
            dynamic_range = self._calculate_dynamic_range(audio)
            
            # Analyze spatial characteristics
            spatial_chars = self._analyze_spatial_characteristics(audio)
            
            # Create ambient noise profile
            ambient_profile = self._create_ambient_noise_profile(audio, sr)
            
            return TheaterFeatures(
                venue_size=venue_size,
                reverb_time=reverb_time,
                noise_floor=noise_floor,
                dialogue_frequency_range=dialogue_freq_range,
                dynamic_range=dynamic_range,
                spatial_characteristics=spatial_chars,
                ambient_noise_profile=ambient_profile
            )
            
        except Exception as e:
            logger.error(f"Theater characteristics analysis failed: {e}")
            # Return default characteristics
            return self._get_default_theater_features()
    
    def apply_venue_optimization(self, audio: np.ndarray, 
                                features: TheaterFeatures) -> np.ndarray:
        """
        Apply venue-specific audio processing.
        
        Args:
            audio: Audio data as numpy array
            features: Theater characteristics
            
        Returns:
            Processed audio data
        """
        # Default sample rate for backward compatibility
        return self._apply_venue_optimization_with_sr(audio, features, 22050)
    
    def _apply_venue_optimization_with_sr(self, audio: np.ndarray, 
                                         features: TheaterFeatures, sr: int) -> np.ndarray:
        """
        Apply venue-specific audio processing with sample rate.
        
        Args:
            audio: Audio data as numpy array
            features: Theater characteristics
            sr: Sample rate
            
        Returns:
            Processed audio data
        """
        try:
            logger.debug(f"Applying venue optimization for {features.venue_size} theater")
            
            # Apply noise reduction based on venue characteristics
            audio = self.noise_reducer.reduce_noise(audio, features)
            
            # Apply dialogue enhancement
            if len(audio.shape) > 1:
                # Process each channel
                for ch in range(audio.shape[1]):
                    audio[:, ch] = self.dialogue_enhancer.enhance_dialogue(
                        audio[:, ch], sr, features
                    )
            else:
                audio = self.dialogue_enhancer.enhance_dialogue(audio, sr, features)
            
            # Apply venue-specific EQ
            audio = self._apply_venue_eq(audio, features)
            
            # Apply dynamic range optimization
            audio = self._optimize_dynamic_range(audio, features)
            
            # Apply spatial audio optimization
            audio = self.spatial_optimizer.optimize_spatial_audio(audio, sr, features)
            
            return audio
            
        except Exception as e:
            logger.error(f"Venue optimization failed: {e}")
            return audio  # Return original audio on failure
    
    def _load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load audio file using librosa."""
        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=False)
            logger.debug(f"Loaded audio: {audio.shape}, sr={sr}")
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return None, None
    
    def _save_audio(self, audio: np.ndarray, sr: int, output_path: str) -> bool:
        """Save audio file using soundfile."""
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio)) * 0.95
            
            # Handle different audio shapes
            if len(audio.shape) == 1:
                # Mono audio
                sf.write(output_path, audio, sr)
            elif len(audio.shape) == 2:
                # Stereo audio - ensure correct orientation
                if audio.shape[0] < audio.shape[1]:
                    # Transpose if channels are in first dimension
                    audio = audio.T
                sf.write(output_path, audio, sr)
            else:
                logger.error(f"Unsupported audio shape: {audio.shape}")
                return False
            
            logger.debug(f"Saved enhanced audio: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save audio {output_path}: {e}")
            return False
    
    def _detect_venue_size(self, audio: np.ndarray, sr: int) -> str:
        """Detect venue size based on reverb characteristics."""
        try:
            # Calculate reverb time as a proxy for venue size
            rt60 = self._calculate_reverb_time(audio, sr)
            
            # Classify based on reverb time
            if rt60 < 1.0:
                return "small"
            elif rt60 < 2.5:
                return "medium"
            else:
                return "large"
                
        except Exception:
            return self.theater_preset  # Fallback to preset
    
    def _calculate_reverb_time(self, audio: np.ndarray, sr: int) -> float:
        """Calculate RT60 reverb time."""
        try:
            # Use a simple energy decay method
            # This is a simplified implementation
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)  # Convert to mono
            
            # Calculate energy envelope
            hop_length = 512
            frame_length = 2048
            energy = librosa.feature.rms(y=audio, frame_length=frame_length, 
                                       hop_length=hop_length)[0]
            
            # Find decay time (simplified)
            peak_idx = np.argmax(energy)
            if peak_idx < len(energy) - 1:
                decay_energy = energy[peak_idx:]
                # Find where energy drops to -60dB (1/1000 of peak)
                threshold = np.max(decay_energy) / 1000
                decay_idx = np.where(decay_energy < threshold)[0]
                if len(decay_idx) > 0:
                    rt60 = (decay_idx[0] * hop_length) / sr
                    return min(rt60, 5.0)  # Cap at 5 seconds
            
            return 1.5  # Default RT60
            
        except Exception:
            return 1.5  # Default fallback
    
    def _analyze_noise_floor(self, audio: np.ndarray) -> float:
        """Analyze background noise floor in dB."""
        try:
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Find quiet segments (bottom 10% of energy)
            rms_energy = librosa.feature.rms(y=audio)[0]
            quiet_threshold = np.percentile(rms_energy, 10)
            quiet_segments = rms_energy[rms_energy <= quiet_threshold]
            
            if len(quiet_segments) > 0:
                noise_floor = 20 * np.log10(np.mean(quiet_segments) + 1e-10)
                return max(noise_floor, -80.0)  # Cap at -80dB
            
            return -60.0  # Default noise floor
            
        except Exception:
            return -60.0
    
    def _detect_dialogue_frequency_range(self, audio: np.ndarray, sr: int) -> Tuple[float, float]:
        """Detect the frequency range containing dialogue."""
        try:
            # Human speech typically ranges from 85Hz to 8kHz
            # Focus on the most important range: 300Hz to 3.4kHz
            return (300.0, 3400.0)
        except Exception:
            return (300.0, 3400.0)
    
    def _calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range in dB."""
        try:
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Calculate RMS energy
            rms_energy = librosa.feature.rms(y=audio)[0]
            
            # Dynamic range = difference between peak and quiet levels
            peak_level = np.percentile(rms_energy, 95)
            quiet_level = np.percentile(rms_energy, 5)
            
            dynamic_range = 20 * np.log10((peak_level + 1e-10) / (quiet_level + 1e-10))
            return min(dynamic_range, 60.0)  # Cap at 60dB
            
        except Exception:
            return 30.0  # Default dynamic range
    
    def _analyze_spatial_characteristics(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze stereo imaging characteristics."""
        try:
            if len(audio.shape) < 2 or audio.shape[1] < 2:
                return {"stereo_width": 0.0, "center_focus": 1.0}
            
            left = audio[:, 0]
            right = audio[:, 1]
            
            # Calculate stereo width (correlation between channels)
            correlation = np.corrcoef(left, right)[0, 1]
            stereo_width = 1.0 - abs(correlation)
            
            # Calculate center focus (energy in center vs sides)
            center = (left + right) / 2
            sides = (left - right) / 2
            
            center_energy = np.mean(center ** 2)
            sides_energy = np.mean(sides ** 2)
            
            center_focus = center_energy / (center_energy + sides_energy + 1e-10)
            
            return {
                "stereo_width": float(stereo_width),
                "center_focus": float(center_focus)
            }
            
        except Exception:
            return {"stereo_width": 0.5, "center_focus": 0.7}
    
    def _create_ambient_noise_profile(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Create ambient noise profile for noise reduction."""
        try:
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Compute spectral characteristics of quiet segments
            rms_energy = librosa.feature.rms(y=audio)[0]
            quiet_threshold = np.percentile(rms_energy, 20)
            
            # Find quiet segments
            hop_length = 512
            quiet_frames = rms_energy <= quiet_threshold
            
            if np.sum(quiet_frames) > 0:
                # Compute average spectrum of quiet segments
                stft = librosa.stft(audio, hop_length=hop_length)
                quiet_spectrum = np.mean(np.abs(stft[:, quiet_frames]), axis=1)
                
                # Create frequency bins
                freqs = librosa.fft_frequencies(sr=sr)
                
                # Sample key frequency bands
                profile = {}
                bands = [(0, 200), (200, 500), (500, 1000), (1000, 2000), 
                        (2000, 4000), (4000, 8000), (8000, sr//2)]
                
                for i, (low, high) in enumerate(bands):
                    mask = (freqs >= low) & (freqs < high)
                    if np.sum(mask) > 0:
                        profile[f"band_{i}"] = float(np.mean(quiet_spectrum[mask]))
                
                return profile
            
            return {"band_0": 0.001}  # Default minimal noise
            
        except Exception:
            return {"band_0": 0.001}
    
    def _apply_venue_eq(self, audio: np.ndarray, features: TheaterFeatures) -> np.ndarray:
        """Apply venue-specific equalization."""
        try:
            # Get venue-specific EQ settings
            eq_settings = self._get_venue_eq_settings(features.venue_size)
            
            # Apply EQ (simplified implementation)
            # In a full implementation, this would use proper filter design
            return audio  # Placeholder - return original for now
            
        except Exception:
            return audio
    
    def _optimize_dynamic_range(self, audio: np.ndarray, features: TheaterFeatures) -> np.ndarray:
        """Optimize dynamic range for theater playback."""
        try:
            # Apply gentle compression to optimize dynamic range
            # This is a simplified implementation
            
            if len(audio.shape) > 1:
                # Process each channel
                processed = np.zeros_like(audio)
                for ch in range(audio.shape[1]):
                    processed[:, ch] = self._apply_compression(audio[:, ch])
                return processed
            else:
                return self._apply_compression(audio)
                
        except Exception:
            return audio
    
    def _apply_compression(self, audio: np.ndarray, 
                          threshold: float = 0.7, ratio: float = 3.0) -> np.ndarray:
        """Apply dynamic range compression."""
        try:
            # Simple soft-knee compression
            abs_audio = np.abs(audio)
            compressed = np.copy(audio)
            
            # Find samples above threshold
            above_threshold = abs_audio > threshold
            
            if np.any(above_threshold):
                # Apply compression to samples above threshold
                excess = abs_audio[above_threshold] - threshold
                compressed_excess = excess / ratio
                
                # Apply compression while preserving sign
                sign = np.sign(audio[above_threshold])
                compressed[above_threshold] = sign * (threshold + compressed_excess)
            
            return compressed
            
        except Exception:
            return audio
    
    def _enhance_spatial_audio(self, audio: np.ndarray, features: TheaterFeatures) -> np.ndarray:
        """Enhance spatial audio characteristics."""
        try:
            if audio.shape[1] != 2:
                return audio  # Only process stereo audio
            
            # Enhance stereo width based on venue size
            width_factor = self._get_spatial_width_factor(features.venue_size)
            
            left = audio[:, 0]
            right = audio[:, 1]
            
            # Calculate mid and side signals
            mid = (left + right) / 2
            side = (left - right) / 2
            
            # Adjust side signal based on width factor
            side_enhanced = side * width_factor
            
            # Reconstruct stereo signal
            left_enhanced = mid + side_enhanced
            right_enhanced = mid - side_enhanced
            
            return np.column_stack([left_enhanced, right_enhanced])
            
        except Exception:
            return audio
    
    def _get_spatial_width_factor(self, venue_size: str) -> float:
        """Get spatial width enhancement factor based on venue size."""
        factors = {
            "small": 0.8,   # Reduce width for intimate venues
            "medium": 1.0,  # Maintain natural width
            "large": 1.2    # Enhance width for large venues
        }
        return factors.get(venue_size, 1.0)
    
    def _get_venue_eq_settings(self, venue_size: str) -> Dict[str, float]:
        """Get EQ settings for venue size."""
        settings = {
            "small": {"low_cut": 80, "presence_boost": 2.0, "high_cut": 12000},
            "medium": {"low_cut": 60, "presence_boost": 1.5, "high_cut": 15000},
            "large": {"low_cut": 40, "presence_boost": 1.0, "high_cut": 18000}
        }
        return settings.get(venue_size, settings["medium"])
    
    def _calculate_quality_metrics(self, original: np.ndarray, 
                                  enhanced: np.ndarray, sr: int) -> Dict[str, float]:
        """Calculate quality improvement metrics."""
        try:
            metrics = {}
            
            # Convert to mono for analysis if needed
            if len(original.shape) > 1:
                orig_mono = np.mean(original, axis=1)
                enh_mono = np.mean(enhanced, axis=1)
            else:
                orig_mono = original
                enh_mono = enhanced
            
            # Calculate SNR improvement (simplified)
            orig_rms = np.sqrt(np.mean(orig_mono ** 2))
            enh_rms = np.sqrt(np.mean(enh_mono ** 2))
            
            if orig_rms > 0:
                snr_improvement = 20 * np.log10(enh_rms / orig_rms)
                metrics["snr_improvement"] = float(snr_improvement)
            
            # Calculate spectral centroid change (brightness)
            orig_centroid = np.mean(librosa.feature.spectral_centroid(y=orig_mono, sr=sr))
            enh_centroid = np.mean(librosa.feature.spectral_centroid(y=enh_mono, sr=sr))
            
            metrics["spectral_centroid_change"] = float(enh_centroid - orig_centroid)
            
            # Calculate dynamic range improvement
            orig_dynamic = self._calculate_dynamic_range(orig_mono)
            enh_dynamic = self._calculate_dynamic_range(enh_mono)
            
            metrics["dynamic_range_change"] = float(enh_dynamic - orig_dynamic)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e}")
            return {"snr_improvement": 0.0}
    
    def _get_default_theater_features(self) -> TheaterFeatures:
        """Get default theater features as fallback."""
        return TheaterFeatures(
            venue_size=self.theater_preset,
            reverb_time=1.5,
            noise_floor=-60.0,
            dialogue_frequency_range=(300.0, 3400.0),
            dynamic_range=30.0,
            spatial_characteristics={"stereo_width": 0.5, "center_focus": 0.7},
            ambient_noise_profile={"band_0": 0.001}
        )
    
    def _load_default_config(self) -> Dict:
        """Load default configuration."""
        return {
            "noise_reduction": {
                "strength": 0.5,
                "preserve_speech": True
            },
            "dialogue_enhancement": {
                "boost_db": 3.0,
                "frequency_range": (300, 3400)
            }
        }


class SpectralGateNoiseReducer:
    """
    Spectral gate-based noise reduction for theater audio.
    
    Uses spectral gating to reduce background noise while preserving
    dialogue and important audio content.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize noise reducer with configuration."""
        self.config = config or {}
        self.strength = self.config.get("strength", 0.5)
        self.preserve_speech = self.config.get("preserve_speech", True)
        
    def reduce_noise(self, audio: np.ndarray, features: TheaterFeatures) -> np.ndarray:
        """
        Apply spectral gate noise reduction.
        
        Args:
            audio: Input audio signal
            features: Theater characteristics for adaptive processing
            
        Returns:
            Noise-reduced audio signal
        """
        try:
            if len(audio.shape) > 1:
                # Process each channel separately
                processed = np.zeros_like(audio)
                for ch in range(audio.shape[1]):
                    processed[:, ch] = self._reduce_noise_mono(audio[:, ch], features)
                return processed
            else:
                return self._reduce_noise_mono(audio, features)
                
        except Exception as e:
            logger.error(f"Noise reduction failed: {e}")
            return audio
    
    def _reduce_noise_mono(self, audio: np.ndarray, features: TheaterFeatures) -> np.ndarray:
        """Apply noise reduction to mono audio."""
        try:
            # Check minimum audio length
            if len(audio) < 2048:
                logger.warning(f"Audio too short for noise reduction: {len(audio)} samples")
                return audio
            
            # Compute STFT with appropriate parameters
            n_fft = min(2048, len(audio) // 4)
            hop_length = n_fft // 4
            
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Check if STFT produced valid results
            if magnitude.size == 0:
                logger.warning("STFT produced empty magnitude spectrum")
                return audio
            
            # Estimate noise threshold from quiet segments
            noise_threshold = self._estimate_noise_threshold(magnitude, features)
            
            # Create spectral gate
            gate = self._create_spectral_gate(magnitude, noise_threshold)
            
            # Apply gate to magnitude spectrum
            gated_magnitude = magnitude * gate
            
            # Reconstruct audio
            gated_stft = gated_magnitude * np.exp(1j * phase)
            processed_audio = librosa.istft(gated_stft, hop_length=hop_length, length=len(audio))
            
            return processed_audio
            
        except Exception as e:
            logger.error(f"Mono noise reduction failed: {e}")
            return audio
    
    def _estimate_noise_threshold(self, magnitude: np.ndarray, 
                                 features: TheaterFeatures) -> np.ndarray:
        """Estimate noise threshold from magnitude spectrum."""
        try:
            # Use noise floor from features
            noise_floor_linear = 10 ** (features.noise_floor / 20)
            
            # Estimate noise threshold as percentile of magnitude
            threshold = np.percentile(magnitude, 20, axis=1, keepdims=True)
            
            # Adjust based on noise floor
            threshold = np.maximum(threshold, noise_floor_linear)
            
            return threshold
            
        except Exception:
            # Fallback threshold
            return np.percentile(magnitude, 15, axis=1, keepdims=True)
    
    def _create_spectral_gate(self, magnitude: np.ndarray, 
                             threshold: np.ndarray) -> np.ndarray:
        """Create spectral gate based on magnitude and threshold."""
        try:
            # Create soft gate
            ratio = magnitude / (threshold + 1e-10)
            
            # Soft gating function
            gate = np.where(ratio > 1.0, 
                           1.0, 
                           ratio ** (1.0 + self.strength))
            
            # Ensure minimum gate value to preserve naturalness
            min_gate = 0.1 if self.preserve_speech else 0.05
            gate = np.maximum(gate, min_gate)
            
            return gate
            
        except Exception:
            return np.ones_like(magnitude)


class DialogueEnhancer:
    """
    Dialogue enhancement for theater audio.
    
    Enhances speech clarity and intelligibility while preserving
    naturalness and avoiding artifacts.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize dialogue enhancer with configuration."""
        self.config = config or {}
        self.boost_db = self.config.get("boost_db", 3.0)
        self.frequency_range = self.config.get("frequency_range", (300, 3400))
        
    def enhance_dialogue(self, audio: np.ndarray, features: TheaterFeatures) -> np.ndarray:
        """
        Apply dialogue enhancement.
        
        Args:
            audio: Input audio signal
            features: Theater characteristics for adaptive processing
            
        Returns:
            Dialogue-enhanced audio signal
        """
        try:
            if len(audio.shape) > 1:
                # Process each channel separately
                processed = np.zeros_like(audio)
                for ch in range(audio.shape[1]):
                    processed[:, ch] = self._enhance_dialogue_mono(audio[:, ch], features)
                return processed
            else:
                return self._enhance_dialogue_mono(audio, features)
                
        except Exception as e:
            logger.error(f"Dialogue enhancement failed: {e}")
            return audio
    
    def _enhance_dialogue_mono(self, audio: np.ndarray, features: TheaterFeatures) -> np.ndarray:
        """Apply dialogue enhancement to mono audio."""
        try:
            # Use dialogue frequency range from features
            freq_range = features.dialogue_frequency_range
            
            # Compute STFT
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=22050)  # Default sr
            
            # Create dialogue frequency mask
            dialogue_mask = self._create_dialogue_mask(freqs, freq_range)
            
            # Apply frequency-selective enhancement
            enhanced_magnitude = self._apply_dialogue_boost(magnitude, dialogue_mask, features)
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            processed_audio = librosa.istft(enhanced_stft)
            
            return processed_audio
            
        except Exception as e:
            logger.error(f"Mono dialogue enhancement failed: {e}")
            return audio
    
    def _create_dialogue_mask(self, freqs: np.ndarray, 
                             freq_range: Tuple[float, float]) -> np.ndarray:
        """Create frequency mask for dialogue enhancement."""
        try:
            low_freq, high_freq = freq_range
            
            # Create smooth mask with soft edges
            mask = np.ones_like(freqs)
            
            # Soft low-frequency rolloff
            low_mask = freqs < low_freq
            if np.any(low_mask):
                rolloff = np.exp(-(low_freq - freqs[low_mask]) / (low_freq * 0.3))
                mask[low_mask] = rolloff
            
            # Soft high-frequency rolloff
            high_mask = freqs > high_freq
            if np.any(high_mask):
                rolloff = np.exp(-(freqs[high_mask] - high_freq) / (high_freq * 0.3))
                mask[high_mask] = rolloff
            
            return mask.reshape(-1, 1)
            
        except Exception:
            return np.ones((len(freqs), 1))
    
    def _apply_dialogue_boost(self, magnitude: np.ndarray, 
                             dialogue_mask: np.ndarray, 
                             features: TheaterFeatures) -> np.ndarray:
        """Apply dialogue frequency boost."""
        try:
            # Adjust boost based on venue size
            venue_boost_factor = {
                "small": 0.8,   # Less boost for intimate venues
                "medium": 1.0,  # Standard boost
                "large": 1.2    # More boost for large venues
            }.get(features.venue_size, 1.0)
            
            effective_boost = self.boost_db * venue_boost_factor
            boost_linear = 10 ** (effective_boost / 20)
            
            # Apply boost with dialogue mask
            boost_factor = 1.0 + (boost_linear - 1.0) * dialogue_mask
            enhanced_magnitude = magnitude * boost_factor
            
            return enhanced_magnitude
            
        except Exception:
            return magnitude


# Add missing import
import time