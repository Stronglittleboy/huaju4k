"""
Spatial Audio Optimization for Theater Enhancement.

This module provides advanced spatial audio processing including stereo imaging
enhancement and audio-video synchronization for theater drama content.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Optional imports with fallbacks
try:
    import librosa
    import scipy.signal
    from scipy import ndimage
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    librosa = None
    scipy = None

from ..models.data_models import TheaterFeatures

logger = logging.getLogger(__name__)


class SpatialAudioOptimizer:
    """
    Advanced spatial audio optimization for theater content.
    
    Provides stereo imaging enhancement, spatial positioning,
    and audio-video synchronization capabilities.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize spatial audio optimizer.
        
        Args:
            config: Configuration dictionary with spatial processing parameters
        """
        if not HAS_AUDIO_LIBS:
            raise ImportError(
                "Audio processing libraries not available. "
                "Please install: librosa, scipy"
            )
        
        self.config = config or self._get_default_config()
        
        # Stereo imaging parameters
        self.imaging_config = self.config.get('stereo_imaging', {})
        
        # Synchronization parameters
        self.sync_config = self.config.get('synchronization', {})
        
        logger.info("SpatialAudioOptimizer initialized")
    
    def optimize_spatial_audio(self, audio: np.ndarray, sr: int,
                              features: TheaterFeatures) -> np.ndarray:
        """
        Apply comprehensive spatial audio optimization.
        
        Args:
            audio: Input audio signal (mono or stereo)
            sr: Sample rate
            features: Theater characteristics for adaptive processing
            
        Returns:
            Spatially optimized audio signal
        """
        try:
            logger.debug("Starting spatial audio optimization")
            
            # Ensure stereo format for spatial processing
            if len(audio.shape) == 1:
                # Convert mono to stereo
                audio = self._mono_to_stereo(audio)
            elif audio.shape[1] != 2:
                logger.warning(f"Unsupported audio format: {audio.shape}")
                return audio
            
            # Step 1: Enhance stereo imaging
            audio = self._enhance_stereo_imaging(audio, sr, features)
            
            # Step 2: Optimize spatial positioning
            audio = self._optimize_spatial_positioning(audio, features)
            
            # Step 3: Apply venue-specific spatial processing
            audio = self._apply_venue_spatial_processing(audio, features)
            
            # Step 4: Ensure phase coherence
            audio = self._ensure_phase_coherence(audio)
            
            logger.debug("Spatial audio optimization completed")
            return audio
            
        except Exception as e:
            logger.error(f"Spatial audio optimization failed: {e}")
            return audio  # Return original on failure
    
    def synchronize_audio_video(self, audio: np.ndarray, video_fps: float,
                               audio_sr: int, target_delay_ms: float = 0.0) -> np.ndarray:
        """
        Synchronize audio with video timing.
        
        Args:
            audio: Input audio signal
            video_fps: Video frame rate
            audio_sr: Audio sample rate
            target_delay_ms: Target delay in milliseconds (positive = audio delayed)
            
        Returns:
            Synchronized audio signal
        """
        try:
            logger.debug(f"Synchronizing audio-video (delay: {target_delay_ms}ms)")
            
            if abs(target_delay_ms) < 1.0:
                return audio  # No significant delay to correct
            
            # Calculate delay in samples
            delay_samples = int(target_delay_ms * audio_sr / 1000.0)
            
            if delay_samples > 0:
                # Audio is delayed - add silence at beginning
                if len(audio.shape) == 1:
                    padding = np.zeros(delay_samples)
                    synchronized_audio = np.concatenate([padding, audio])
                else:
                    padding = np.zeros((delay_samples, audio.shape[1]))
                    synchronized_audio = np.concatenate([padding, audio], axis=0)
            else:
                # Audio is ahead - remove samples from beginning
                delay_samples = abs(delay_samples)
                if delay_samples < len(audio):
                    synchronized_audio = audio[delay_samples:]
                else:
                    logger.warning("Delay correction larger than audio length")
                    synchronized_audio = audio
            
            logger.debug("Audio-video synchronization completed")
            return synchronized_audio
            
        except Exception as e:
            logger.error(f"Audio-video synchronization failed: {e}")
            return audio
    
    def _enhance_stereo_imaging(self, audio: np.ndarray, sr: int,
                               features: TheaterFeatures) -> np.ndarray:
        """
        Enhance stereo imaging for better spatial perception.
        
        Args:
            audio: Stereo audio signal
            sr: Sample rate
            features: Theater characteristics
            
        Returns:
            Stereo imaging enhanced audio
        """
        try:
            logger.debug("Enhancing stereo imaging")
            
            left = audio[:, 0]
            right = audio[:, 1]
            
            # Calculate mid and side signals
            mid = (left + right) / 2
            side = (left - right) / 2
            
            # Enhance stereo width based on venue size
            width_enhancement = self._get_stereo_width_enhancement(features.venue_size)
            
            # Apply frequency-dependent stereo enhancement
            enhanced_side = self._apply_frequency_dependent_stereo(side, sr, features)
            
            # Scale side signal
            enhanced_side *= width_enhancement
            
            # Apply stereo imaging enhancement
            enhanced_side = self._apply_stereo_imaging_enhancement(enhanced_side, sr)
            
            # Reconstruct stereo signal
            enhanced_left = mid + enhanced_side
            enhanced_right = mid - enhanced_side
            
            # Ensure no clipping
            enhanced_audio = np.column_stack([enhanced_left, enhanced_right])
            enhanced_audio = self._normalize_stereo(enhanced_audio)
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Stereo imaging enhancement failed: {e}")
            return audio
    
    def _apply_frequency_dependent_stereo(self, side: np.ndarray, sr: int,
                                         features: TheaterFeatures) -> np.ndarray:
        """Apply frequency-dependent stereo enhancement."""
        try:
            # Check minimum audio length
            if len(side) < 2048:
                logger.warning(f"Audio too short for frequency-dependent stereo: {len(side)} samples")
                return side
            
            # Compute STFT
            n_fft = min(2048, len(side) // 4)
            hop_length = n_fft // 4
            
            stft = librosa.stft(side, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Check if STFT produced valid results
            if magnitude.size == 0:
                logger.warning("STFT produced empty magnitude spectrum")
                return side
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # Define frequency-dependent enhancement
            enhancement_curve = self._get_frequency_enhancement_curve(freqs, features)
            
            # Apply enhancement - ensure proper broadcasting
            if enhancement_curve.shape[0] == magnitude.shape[0]:
                enhanced_magnitude = magnitude * enhancement_curve.reshape(-1, 1)
            else:
                logger.warning("Enhancement curve shape mismatch, skipping frequency enhancement")
                enhanced_magnitude = magnitude
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_side = librosa.istft(enhanced_stft, hop_length=hop_length, length=len(side))
            
            return enhanced_side
            
        except Exception as e:
            logger.error(f"Frequency-dependent stereo enhancement failed: {e}")
            return side
    
    def _apply_stereo_imaging_enhancement(self, side: np.ndarray, sr: int) -> np.ndarray:
        """Apply stereo imaging enhancement using psychoacoustic principles."""
        try:
            # Apply subtle delay to create enhanced spatial perception
            delay_samples = int(0.001 * sr)  # 1ms delay
            
            if delay_samples > 0 and delay_samples < len(side):
                # Create delayed version
                delayed_side = np.zeros_like(side)
                delayed_side[delay_samples:] = side[:-delay_samples]
                
                # Mix with original for enhanced imaging
                mix_factor = 0.1  # 10% mix
                enhanced_side = side + mix_factor * delayed_side
                
                return enhanced_side
            
            return side
            
        except Exception as e:
            logger.error(f"Stereo imaging enhancement failed: {e}")
            return side
    
    def _optimize_spatial_positioning(self, audio: np.ndarray,
                                     features: TheaterFeatures) -> np.ndarray:
        """
        Optimize spatial positioning for theater acoustics.
        
        Args:
            audio: Stereo audio signal
            features: Theater characteristics
            
        Returns:
            Spatially positioned audio
        """
        try:
            logger.debug("Optimizing spatial positioning")
            
            left = audio[:, 0]
            right = audio[:, 1]
            
            # Apply venue-specific spatial positioning
            positioned_left, positioned_right = self._apply_venue_positioning(
                left, right, features
            )
            
            # Apply center focus adjustment
            center_focus = features.spatial_characteristics.get('center_focus', 0.7)
            positioned_left, positioned_right = self._adjust_center_focus(
                positioned_left, positioned_right, center_focus
            )
            
            # Apply spatial depth enhancement
            positioned_left, positioned_right = self._enhance_spatial_depth(
                positioned_left, positioned_right, features
            )
            
            return np.column_stack([positioned_left, positioned_right])
            
        except Exception as e:
            logger.error(f"Spatial positioning optimization failed: {e}")
            return audio
    
    def _apply_venue_positioning(self, left: np.ndarray, right: np.ndarray,
                                features: TheaterFeatures) -> Tuple[np.ndarray, np.ndarray]:
        """Apply venue-specific spatial positioning."""
        try:
            venue_size = features.venue_size
            
            # Get positioning parameters
            positioning = self._get_venue_positioning_parameters(venue_size)
            
            # Apply cross-talk simulation for venue acoustics
            crosstalk_delay = positioning['crosstalk_delay']
            crosstalk_level = positioning['crosstalk_level']
            
            if crosstalk_delay > 0 and crosstalk_level > 0:
                delay_samples = int(crosstalk_delay * 22050)  # Assuming 22050 Hz
                
                if delay_samples < len(left):
                    # Add delayed crosstalk
                    left_with_crosstalk = left.copy()
                    right_with_crosstalk = right.copy()
                    
                    # Left ear receives delayed right signal
                    left_with_crosstalk[delay_samples:] += (
                        crosstalk_level * right[:-delay_samples]
                    )
                    
                    # Right ear receives delayed left signal
                    right_with_crosstalk[delay_samples:] += (
                        crosstalk_level * left[:-delay_samples]
                    )
                    
                    return left_with_crosstalk, right_with_crosstalk
            
            return left, right
            
        except Exception as e:
            logger.error(f"Venue positioning failed: {e}")
            return left, right
    
    def _adjust_center_focus(self, left: np.ndarray, right: np.ndarray,
                            center_focus: float) -> Tuple[np.ndarray, np.ndarray]:
        """Adjust center focus of stereo image."""
        try:
            # Calculate center and side components
            center = (left + right) / 2
            side_left = left - center
            side_right = right - center
            
            # Adjust center focus
            focus_factor = center_focus
            adjusted_center = center * focus_factor
            
            # Reconstruct with adjusted center
            adjusted_left = adjusted_center + side_left
            adjusted_right = adjusted_center + side_right
            
            return adjusted_left, adjusted_right
            
        except Exception as e:
            logger.error(f"Center focus adjustment failed: {e}")
            return left, right
    
    def _enhance_spatial_depth(self, left: np.ndarray, right: np.ndarray,
                              features: TheaterFeatures) -> Tuple[np.ndarray, np.ndarray]:
        """Enhance spatial depth perception."""
        try:
            # Apply subtle reverb to enhance depth
            reverb_time = features.reverb_time
            depth_factor = min(reverb_time / 2.0, 1.0)  # Scale based on reverb time
            
            if depth_factor > 0.1:
                # Create simple reverb effect
                reverb_left = self._apply_simple_reverb(left, depth_factor)
                reverb_right = self._apply_simple_reverb(right, depth_factor)
                
                # Mix with original
                reverb_mix = 0.15  # 15% reverb mix
                enhanced_left = left + reverb_mix * reverb_left
                enhanced_right = right + reverb_mix * reverb_right
                
                return enhanced_left, enhanced_right
            
            return left, right
            
        except Exception as e:
            logger.error(f"Spatial depth enhancement failed: {e}")
            return left, right
    
    def _apply_simple_reverb(self, audio: np.ndarray, depth_factor: float) -> np.ndarray:
        """Apply simple reverb effect for spatial depth."""
        try:
            # Create multiple delayed versions with decreasing amplitude
            reverb_signal = np.zeros_like(audio)
            
            delays = [0.02, 0.05, 0.08, 0.12]  # Delay times in seconds
            gains = [0.3, 0.2, 0.15, 0.1]     # Corresponding gains
            
            sr = 22050  # Assumed sample rate
            
            for delay_time, gain in zip(delays, gains):
                delay_samples = int(delay_time * sr)
                if delay_samples < len(audio):
                    delayed_audio = np.zeros_like(audio)
                    delayed_audio[delay_samples:] = audio[:-delay_samples]
                    reverb_signal += gain * depth_factor * delayed_audio
            
            return reverb_signal
            
        except Exception as e:
            logger.error(f"Simple reverb failed: {e}")
            return np.zeros_like(audio)
    
    def _apply_venue_spatial_processing(self, audio: np.ndarray,
                                       features: TheaterFeatures) -> np.ndarray:
        """Apply venue-specific spatial processing."""
        try:
            logger.debug(f"Applying venue spatial processing for {features.venue_size}")
            
            venue_size = features.venue_size
            
            # Get venue-specific processing parameters
            spatial_params = self._get_venue_spatial_parameters(venue_size)
            
            # Apply venue-specific EQ to spatial components
            audio = self._apply_spatial_eq(audio, spatial_params)
            
            # Apply venue-specific stereo width adjustment
            audio = self._apply_venue_stereo_width(audio, spatial_params)
            
            return audio
            
        except Exception as e:
            logger.error(f"Venue spatial processing failed: {e}")
            return audio
    
    def _apply_spatial_eq(self, audio: np.ndarray, spatial_params: Dict) -> np.ndarray:
        """Apply EQ specifically for spatial components."""
        try:
            left = audio[:, 0]
            right = audio[:, 1]
            
            # Calculate side signal for spatial EQ
            side = (left - right) / 2
            mid = (left + right) / 2
            
            # Apply EQ to side signal only
            eq_params = spatial_params.get('spatial_eq', {})
            
            if eq_params:
                # Simple high-frequency boost for spatial clarity
                high_boost = eq_params.get('high_boost', 1.0)
                if high_boost != 1.0:
                    # Apply simple high-frequency emphasis
                    # This is a simplified implementation
                    side *= high_boost
            
            # Reconstruct stereo
            enhanced_left = mid + side
            enhanced_right = mid - side
            
            return np.column_stack([enhanced_left, enhanced_right])
            
        except Exception as e:
            logger.error(f"Spatial EQ failed: {e}")
            return audio
    
    def _apply_venue_stereo_width(self, audio: np.ndarray, spatial_params: Dict) -> np.ndarray:
        """Apply venue-specific stereo width adjustment."""
        try:
            width_factor = spatial_params.get('stereo_width_factor', 1.0)
            
            if width_factor != 1.0:
                left = audio[:, 0]
                right = audio[:, 1]
                
                # Calculate mid and side
                mid = (left + right) / 2
                side = (left - right) / 2
                
                # Adjust side signal
                adjusted_side = side * width_factor
                
                # Reconstruct
                adjusted_left = mid + adjusted_side
                adjusted_right = mid - adjusted_side
                
                return np.column_stack([adjusted_left, adjusted_right])
            
            return audio
            
        except Exception as e:
            logger.error(f"Venue stereo width adjustment failed: {e}")
            return audio
    
    def _ensure_phase_coherence(self, audio: np.ndarray) -> np.ndarray:
        """Ensure phase coherence between stereo channels."""
        try:
            logger.debug("Ensuring phase coherence")
            
            left = audio[:, 0]
            right = audio[:, 1]
            
            # Check for phase issues using correlation
            correlation = np.corrcoef(left, right)[0, 1]
            
            if correlation < -0.5:  # Significant phase issues
                logger.warning("Phase coherence issues detected, applying correction")
                
                # Simple phase correction: invert one channel if severely out of phase
                right = -right
                
                return np.column_stack([left, right])
            
            return audio
            
        except Exception as e:
            logger.error(f"Phase coherence check failed: {e}")
            return audio
    
    def _mono_to_stereo(self, audio: np.ndarray) -> np.ndarray:
        """Convert mono audio to stereo."""
        try:
            # Simple duplication for stereo
            stereo_audio = np.column_stack([audio, audio])
            return stereo_audio
            
        except Exception as e:
            logger.error(f"Mono to stereo conversion failed: {e}")
            return np.column_stack([audio, audio])
    
    def _normalize_stereo(self, audio: np.ndarray, target_level: float = 0.95) -> np.ndarray:
        """Normalize stereo audio to prevent clipping."""
        try:
            peak_level = np.max(np.abs(audio))
            
            if peak_level > target_level:
                scale_factor = target_level / peak_level
                normalized_audio = audio * scale_factor
                return normalized_audio
            
            return audio
            
        except Exception as e:
            logger.error(f"Stereo normalization failed: {e}")
            return audio
    
    # Configuration and parameter methods
    
    def _get_stereo_width_enhancement(self, venue_size: str) -> float:
        """Get stereo width enhancement factor based on venue size."""
        enhancements = {
            'small': 0.8,   # Reduce width for intimate venues
            'medium': 1.0,  # Maintain natural width
            'large': 1.3    # Enhance width for large venues
        }
        return enhancements.get(venue_size, 1.0)
    
    def _get_frequency_enhancement_curve(self, freqs: np.ndarray,
                                        features: TheaterFeatures) -> np.ndarray:
        """Get frequency-dependent enhancement curve."""
        try:
            venue_size = features.venue_size
            
            # Define frequency-dependent enhancement
            enhancement = np.ones_like(freqs)
            
            # Enhance mid frequencies for better spatial perception
            mid_freq_mask = (freqs >= 1000) & (freqs <= 4000)
            if np.any(mid_freq_mask):
                mid_enhancement = {
                    'small': 1.1,
                    'medium': 1.2,
                    'large': 1.3
                }.get(venue_size, 1.2)
                
                enhancement[mid_freq_mask] = mid_enhancement
            
            # Gentle high-frequency enhancement for spatial clarity
            high_freq_mask = freqs >= 4000
            if np.any(high_freq_mask):
                high_enhancement = {
                    'small': 1.05,
                    'medium': 1.1,
                    'large': 1.15
                }.get(venue_size, 1.1)
                
                enhancement[high_freq_mask] = high_enhancement
            
            return enhancement
            
        except Exception as e:
            logger.error(f"Frequency enhancement curve generation failed: {e}")
            return np.ones_like(freqs)
    
    def _get_venue_positioning_parameters(self, venue_size: str) -> Dict[str, float]:
        """Get venue-specific positioning parameters."""
        params = {
            'small': {
                'crosstalk_delay': 0.0002,  # 0.2ms
                'crosstalk_level': 0.05     # 5% crosstalk
            },
            'medium': {
                'crosstalk_delay': 0.0005,  # 0.5ms
                'crosstalk_level': 0.03     # 3% crosstalk
            },
            'large': {
                'crosstalk_delay': 0.001,   # 1ms
                'crosstalk_level': 0.02     # 2% crosstalk
            }
        }
        return params.get(venue_size, params['medium'])
    
    def _get_venue_spatial_parameters(self, venue_size: str) -> Dict:
        """Get venue-specific spatial processing parameters."""
        params = {
            'small': {
                'stereo_width_factor': 0.8,
                'spatial_eq': {'high_boost': 1.05}
            },
            'medium': {
                'stereo_width_factor': 1.0,
                'spatial_eq': {'high_boost': 1.1}
            },
            'large': {
                'stereo_width_factor': 1.3,
                'spatial_eq': {'high_boost': 1.15}
            }
        }
        return params.get(venue_size, params['medium'])
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for spatial audio optimization."""
        return {
            'stereo_imaging': {
                'enable_width_enhancement': True,
                'enable_frequency_dependent': True,
                'enable_imaging_enhancement': True
            },
            'synchronization': {
                'enable_av_sync': True,
                'max_correction_ms': 100.0,
                'detection_threshold': 5.0
            },
            'spatial_processing': {
                'enable_venue_positioning': True,
                'enable_center_focus': True,
                'enable_depth_enhancement': True,
                'enable_phase_coherence': True
            }
        }