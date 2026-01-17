"""
Advanced Dialogue Enhancement for Theater Audio.

This module provides sophisticated dialogue enhancement algorithms
specifically designed for theater drama content, focusing on speech
clarity while preserving naturalness.
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


class AdvancedDialogueEnhancer:
    """
    Advanced dialogue enhancement system for theater audio.
    
    Provides sophisticated speech frequency optimization, clarity enhancement,
    and dynamic range preservation while avoiding artifacts.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize advanced dialogue enhancer.
        
        Args:
            config: Configuration dictionary with enhancement parameters
        """
        if not HAS_AUDIO_LIBS:
            raise ImportError(
                "Audio processing libraries not available. "
                "Please install: librosa, scipy"
            )
        
        self.config = config or self._get_default_config()
        
        # Speech frequency optimization parameters
        self.speech_bands = self._define_speech_frequency_bands()
        
        # Clarity enhancement parameters
        self.clarity_config = self.config.get('clarity', {})
        
        # Dynamic range preservation parameters
        self.dynamics_config = self.config.get('dynamics', {})
        
        logger.info("AdvancedDialogueEnhancer initialized")
    
    def enhance_dialogue(self, audio: np.ndarray, sr: int, 
                        features: TheaterFeatures) -> np.ndarray:
        """
        Apply comprehensive dialogue enhancement.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            features: Theater characteristics for adaptive processing
            
        Returns:
            Enhanced audio signal
        """
        try:
            logger.debug("Starting advanced dialogue enhancement")
            
            # Step 1: Speech frequency optimization
            audio = self._optimize_speech_frequencies(audio, sr, features)
            
            # Step 2: Dialogue clarity enhancement
            audio = self._enhance_dialogue_clarity(audio, sr, features)
            
            # Step 3: Dynamic range preservation
            audio = self._preserve_dynamic_range(audio, features)
            
            # Step 4: Artifact prevention
            audio = self._prevent_artifacts(audio, sr)
            
            logger.debug("Advanced dialogue enhancement completed")
            return audio
            
        except Exception as e:
            logger.error(f"Advanced dialogue enhancement failed: {e}")
            return audio  # Return original on failure
    
    def _optimize_speech_frequencies(self, audio: np.ndarray, sr: int,
                                   features: TheaterFeatures) -> np.ndarray:
        """
        Optimize speech frequency bands for better intelligibility.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            features: Theater characteristics
            
        Returns:
            Frequency-optimized audio
        """
        try:
            logger.debug("Optimizing speech frequencies")
            
            # Check minimum audio length
            if len(audio) < 2048:
                logger.warning(f"Audio too short for speech optimization: {len(audio)} samples")
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
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # Apply frequency-specific enhancements
            enhanced_magnitude = self._apply_speech_band_enhancement(
                magnitude, freqs, features
            )
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length, length=len(audio))
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Speech frequency optimization failed: {e}")
            return audio
    
    def _apply_speech_band_enhancement(self, magnitude: np.ndarray, 
                                     freqs: np.ndarray,
                                     features: TheaterFeatures) -> np.ndarray:
        """Apply enhancement to specific speech frequency bands."""
        try:
            enhanced_magnitude = magnitude.copy()
            
            # Get venue-specific enhancement factors
            venue_factors = self._get_venue_enhancement_factors(features.venue_size)
            
            for band_name, band_config in self.speech_bands.items():
                freq_range = band_config['range']
                base_gain = band_config['gain']
                
                # Adjust gain based on venue
                venue_gain = venue_factors.get(band_name, 1.0)
                effective_gain = base_gain * venue_gain
                
                # Create frequency mask
                mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
                
                if np.any(mask):
                    # Apply smooth gain transition
                    gain_linear = 10 ** (effective_gain / 20)
                    enhanced_magnitude[mask] *= gain_linear
            
            return enhanced_magnitude
            
        except Exception as e:
            logger.error(f"Speech band enhancement failed: {e}")
            return magnitude
    
    def _enhance_dialogue_clarity(self, audio: np.ndarray, sr: int,
                                 features: TheaterFeatures) -> np.ndarray:
        """
        Enhance dialogue clarity using advanced signal processing.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            features: Theater characteristics
            
        Returns:
            Clarity-enhanced audio
        """
        try:
            logger.debug("Enhancing dialogue clarity")
            
            # Apply spectral enhancement
            audio = self._apply_spectral_enhancement(audio, sr, features)
            
            # Apply transient enhancement for consonants
            audio = self._enhance_transients(audio, sr)
            
            # Apply formant enhancement
            audio = self._enhance_formants(audio, sr, features)
            
            return audio
            
        except Exception as e:
            logger.error(f"Dialogue clarity enhancement failed: {e}")
            return audio
    
    def _apply_spectral_enhancement(self, audio: np.ndarray, sr: int,
                                   features: TheaterFeatures) -> np.ndarray:
        """Apply spectral enhancement for better speech clarity."""
        try:
            # Check minimum audio length
            if len(audio) < 2048:
                logger.warning(f"Audio too short for spectral enhancement: {len(audio)} samples")
                return audio
            
            # Compute spectral features
            n_fft = min(2048, len(audio) // 4)
            hop_length = n_fft // 4
            
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Check if STFT produced valid results
            if magnitude.size == 0:
                logger.warning("STFT produced empty magnitude spectrum")
                return audio
            
            # Apply spectral sharpening
            sharpened_magnitude = self._apply_spectral_sharpening(magnitude, features)
            
            # Apply harmonic enhancement
            enhanced_magnitude = self._enhance_harmonics(sharpened_magnitude, sr, n_fft)
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length, length=len(audio))
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Spectral enhancement failed: {e}")
            return audio
    
    def _apply_spectral_sharpening(self, magnitude: np.ndarray,
                                  features: TheaterFeatures) -> np.ndarray:
        """Apply spectral sharpening to enhance speech clarity."""
        try:
            # Get sharpening strength based on venue size
            strength = self._get_sharpening_strength(features.venue_size)
            
            # Apply spectral contrast enhancement
            # This enhances the difference between peaks and valleys in the spectrum
            enhanced_magnitude = magnitude.copy()
            
            # Smooth the magnitude spectrum
            smoothed = ndimage.gaussian_filter1d(magnitude, sigma=2.0, axis=0)
            
            # Calculate spectral contrast
            contrast = magnitude / (smoothed + 1e-10)
            
            # Apply contrast enhancement
            enhanced_contrast = contrast ** (1.0 + strength)
            enhanced_magnitude = smoothed * enhanced_contrast
            
            return enhanced_magnitude
            
        except Exception as e:
            logger.error(f"Spectral sharpening failed: {e}")
            return magnitude
    
    def _enhance_harmonics(self, magnitude: np.ndarray, sr: int, n_fft: int) -> np.ndarray:
        """Enhance harmonic content for better speech intelligibility."""
        try:
            # Find harmonic peaks in the spectrum
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # Focus on speech fundamental frequency range (80-400 Hz)
            f0_range = (80, 400)
            f0_mask = (freqs >= f0_range[0]) & (freqs <= f0_range[1])
            
            if np.any(f0_mask):
                # Enhance harmonic structure
                harmonic_boost = 1.2  # 20% boost for harmonics
                magnitude[f0_mask] *= harmonic_boost
            
            return magnitude
            
        except Exception as e:
            logger.error(f"Harmonic enhancement failed: {e}")
            return magnitude
    
    def _enhance_transients(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Enhance transient components for better consonant clarity."""
        try:
            # Detect transients using onset detection
            onset_frames = librosa.onset.onset_detect(
                y=audio, sr=sr, units='frames', hop_length=512
            )
            
            if len(onset_frames) == 0:
                return audio
            
            # Enhance regions around transients
            enhanced_audio = audio.copy()
            hop_length = 512
            
            for onset_frame in onset_frames:
                # Convert frame to sample index
                onset_sample = onset_frame * hop_length
                
                # Define enhancement window around transient
                window_size = int(0.02 * sr)  # 20ms window
                start_idx = max(0, onset_sample - window_size // 2)
                end_idx = min(len(audio), onset_sample + window_size // 2)
                
                # Apply gentle enhancement to transient region
                enhancement_factor = 1.15  # 15% boost
                enhanced_audio[start_idx:end_idx] *= enhancement_factor
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Transient enhancement failed: {e}")
            return audio
    
    def _enhance_formants(self, audio: np.ndarray, sr: int,
                         features: TheaterFeatures) -> np.ndarray:
        """Enhance formant frequencies for better vowel clarity."""
        try:
            # Check minimum audio length
            if len(audio) < 2048:
                logger.warning(f"Audio too short for formant enhancement: {len(audio)} samples")
                return audio
            
            # Define typical formant frequency ranges
            formant_ranges = [
                (300, 800),   # F1 - first formant
                (800, 2500),  # F2 - second formant
                (2500, 3500)  # F3 - third formant
            ]
            
            # Compute STFT
            n_fft = min(2048, len(audio) // 4)
            hop_length = n_fft // 4
            
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Check if STFT produced valid results
            if magnitude.size == 0:
                logger.warning("STFT produced empty magnitude spectrum")
                return audio
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # Enhance each formant region
            enhanced_magnitude = magnitude.copy()
            
            for i, (f_low, f_high) in enumerate(formant_ranges):
                formant_mask = (freqs >= f_low) & (freqs <= f_high)
                
                if np.any(formant_mask):
                    # Apply formant-specific enhancement
                    formant_boost = self._get_formant_boost(i, features.venue_size)
                    boost_linear = 10 ** (formant_boost / 20)
                    enhanced_magnitude[formant_mask] *= boost_linear
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length, length=len(audio))
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Formant enhancement failed: {e}")
            return audio
    
    def _preserve_dynamic_range(self, audio: np.ndarray,
                               features: TheaterFeatures) -> np.ndarray:
        """
        Preserve natural dynamic range while enhancing dialogue.
        
        Args:
            audio: Input audio signal
            features: Theater characteristics
            
        Returns:
            Dynamic range preserved audio
        """
        try:
            logger.debug("Preserving dynamic range")
            
            # Calculate original dynamic characteristics
            original_rms = np.sqrt(np.mean(audio ** 2))
            original_peak = np.max(np.abs(audio))
            original_crest_factor = original_peak / (original_rms + 1e-10)
            
            # Apply adaptive compression to preserve dynamics
            compressed_audio = self._apply_adaptive_compression(audio, features)
            
            # Restore natural crest factor
            restored_audio = self._restore_crest_factor(
                compressed_audio, original_crest_factor
            )
            
            return restored_audio
            
        except Exception as e:
            logger.error(f"Dynamic range preservation failed: {e}")
            return audio
    
    def _apply_adaptive_compression(self, audio: np.ndarray,
                                   features: TheaterFeatures) -> np.ndarray:
        """Apply adaptive compression based on venue characteristics."""
        try:
            # Get compression parameters based on venue size
            comp_params = self._get_compression_parameters(features.venue_size)
            
            threshold = comp_params['threshold']
            ratio = comp_params['ratio']
            attack_time = comp_params['attack_time']
            release_time = comp_params['release_time']
            
            # Apply envelope-following compression
            compressed_audio = self._envelope_compression(
                audio, threshold, ratio, attack_time, release_time
            )
            
            return compressed_audio
            
        except Exception as e:
            logger.error(f"Adaptive compression failed: {e}")
            return audio
    
    def _envelope_compression(self, audio: np.ndarray, threshold: float,
                             ratio: float, attack_time: float,
                             release_time: float, sr: int = 22050) -> np.ndarray:
        """Apply envelope-following compression."""
        try:
            # Calculate envelope
            envelope = np.abs(audio)
            
            # Smooth envelope with attack/release characteristics
            smoothed_envelope = self._smooth_envelope(
                envelope, attack_time, release_time, sr
            )
            
            # Apply compression to envelope
            compressed_envelope = np.where(
                smoothed_envelope > threshold,
                threshold + (smoothed_envelope - threshold) / ratio,
                smoothed_envelope
            )
            
            # Apply gain reduction
            gain_reduction = compressed_envelope / (smoothed_envelope + 1e-10)
            compressed_audio = audio * gain_reduction
            
            return compressed_audio
            
        except Exception as e:
            logger.error(f"Envelope compression failed: {e}")
            return audio
    
    def _smooth_envelope(self, envelope: np.ndarray, attack_time: float,
                        release_time: float, sr: int) -> np.ndarray:
        """Smooth envelope with attack/release characteristics."""
        try:
            # Convert time constants to filter coefficients
            attack_coeff = np.exp(-1.0 / (attack_time * sr))
            release_coeff = np.exp(-1.0 / (release_time * sr))
            
            # Apply asymmetric smoothing
            smoothed = np.zeros_like(envelope)
            smoothed[0] = envelope[0]
            
            for i in range(1, len(envelope)):
                if envelope[i] > smoothed[i-1]:
                    # Attack phase
                    smoothed[i] = attack_coeff * smoothed[i-1] + (1 - attack_coeff) * envelope[i]
                else:
                    # Release phase
                    smoothed[i] = release_coeff * smoothed[i-1] + (1 - release_coeff) * envelope[i]
            
            return smoothed
            
        except Exception as e:
            logger.error(f"Envelope smoothing failed: {e}")
            return envelope
    
    def _restore_crest_factor(self, audio: np.ndarray, 
                             target_crest_factor: float) -> np.ndarray:
        """Restore natural crest factor to preserve dynamics."""
        try:
            current_rms = np.sqrt(np.mean(audio ** 2))
            current_peak = np.max(np.abs(audio))
            current_crest_factor = current_peak / (current_rms + 1e-10)
            
            if current_crest_factor > 0:
                # Adjust to match target crest factor
                scale_factor = target_crest_factor / current_crest_factor
                scale_factor = np.clip(scale_factor, 0.8, 1.2)  # Limit adjustment
                
                restored_audio = audio * scale_factor
                return restored_audio
            
            return audio
            
        except Exception as e:
            logger.error(f"Crest factor restoration failed: {e}")
            return audio
    
    def _prevent_artifacts(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Prevent processing artifacts and ensure natural sound."""
        try:
            logger.debug("Preventing artifacts")
            
            # Apply gentle low-pass filtering to remove high-frequency artifacts
            audio = self._apply_anti_aliasing_filter(audio, sr)
            
            # Apply gentle smoothing to reduce harsh transitions
            audio = self._smooth_harsh_transitions(audio)
            
            # Normalize to prevent clipping
            audio = self._normalize_audio(audio)
            
            return audio
            
        except Exception as e:
            logger.error(f"Artifact prevention failed: {e}")
            return audio
    
    def _apply_anti_aliasing_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply anti-aliasing filter to prevent high-frequency artifacts."""
        try:
            # Check minimum audio length for filtering
            if len(audio) < 32:  # Minimum length for filtering
                logger.warning(f"Audio too short for anti-aliasing filter: {len(audio)} samples")
                return audio
            
            # Design low-pass filter at 80% of Nyquist frequency
            cutoff_freq = 0.8 * sr / 2
            normalized_cutoff = cutoff_freq / (sr / 2)
            
            # Ensure normalized cutoff is valid
            if normalized_cutoff >= 1.0:
                normalized_cutoff = 0.95
            
            # Design Butterworth filter
            b, a = scipy.signal.butter(4, normalized_cutoff, btype='low')
            
            # Check if we have enough samples for filtfilt
            min_padlen = 3 * max(len(a), len(b))
            if len(audio) <= min_padlen:
                logger.warning(f"Audio too short for filtfilt (need >{min_padlen}, got {len(audio)})")
                # Use simple filter instead
                filtered_audio = scipy.signal.lfilter(b, a, audio)
            else:
                # Apply filter
                filtered_audio = scipy.signal.filtfilt(b, a, audio)
            
            return filtered_audio
            
        except Exception as e:
            logger.error(f"Anti-aliasing filter failed: {e}")
            return audio
    
    def _smooth_harsh_transitions(self, audio: np.ndarray) -> np.ndarray:
        """Smooth harsh transitions to reduce artifacts."""
        try:
            # Check minimum audio length
            if len(audio) < 3:
                logger.warning(f"Audio too short for transition smoothing: {len(audio)} samples")
                return audio
            
            # Apply gentle smoothing using a small moving average
            window_size = min(3, len(audio))
            kernel = np.ones(window_size) / window_size
            
            # Pad audio to handle edges
            pad_width = window_size // 2
            if len(audio) > pad_width * 2:
                padded_audio = np.pad(audio, pad_width, mode='edge')
                smoothed_audio = np.convolve(padded_audio, kernel, mode='valid')
                
                # Ensure output length matches input
                if len(smoothed_audio) != len(audio):
                    smoothed_audio = smoothed_audio[:len(audio)]
                
                # Blend with original to preserve transients
                blend_factor = 0.1  # 10% smoothing
                blended_audio = (1 - blend_factor) * audio + blend_factor * smoothed_audio
                
                return blended_audio
            else:
                return audio
            
        except Exception as e:
            logger.error(f"Transition smoothing failed: {e}")
            return audio
    
    def _normalize_audio(self, audio: np.ndarray, target_level: float = 0.95) -> np.ndarray:
        """Normalize audio to prevent clipping while preserving dynamics."""
        try:
            # Check for empty or invalid audio
            if len(audio) == 0:
                logger.warning("Cannot normalize empty audio array")
                return audio
            
            peak_level = np.max(np.abs(audio))
            
            # Check for zero or very small peak level
            if peak_level < 1e-10:
                logger.warning("Audio peak level too small for normalization")
                return audio
            
            if peak_level > target_level:
                scale_factor = target_level / peak_level
                normalized_audio = audio * scale_factor
                return normalized_audio
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio normalization failed: {e}")
            return audio
    
    # Configuration and parameter methods
    
    def _define_speech_frequency_bands(self) -> Dict[str, Dict]:
        """Define speech frequency bands and their enhancement parameters."""
        return {
            'fundamental': {
                'range': (80, 250),
                'gain': 1.0,  # dB
                'description': 'Fundamental frequency range'
            },
            'vowel_clarity': {
                'range': (250, 800),
                'gain': 2.0,  # dB
                'description': 'Primary vowel formant range'
            },
            'consonant_clarity': {
                'range': (800, 2500),
                'gain': 3.0,  # dB
                'description': 'Consonant clarity range'
            },
            'presence': {
                'range': (2500, 4000),
                'gain': 2.5,  # dB
                'description': 'Speech presence and intelligibility'
            },
            'brilliance': {
                'range': (4000, 8000),
                'gain': 1.5,  # dB
                'description': 'Speech brilliance and air'
            }
        }
    
    def _get_venue_enhancement_factors(self, venue_size: str) -> Dict[str, float]:
        """Get venue-specific enhancement factors."""
        factors = {
            'small': {
                'fundamental': 0.8,
                'vowel_clarity': 0.9,
                'consonant_clarity': 1.0,
                'presence': 1.1,
                'brilliance': 1.2
            },
            'medium': {
                'fundamental': 1.0,
                'vowel_clarity': 1.0,
                'consonant_clarity': 1.0,
                'presence': 1.0,
                'brilliance': 1.0
            },
            'large': {
                'fundamental': 1.2,
                'vowel_clarity': 1.1,
                'consonant_clarity': 1.0,
                'presence': 0.9,
                'brilliance': 0.8
            }
        }
        return factors.get(venue_size, factors['medium'])
    
    def _get_sharpening_strength(self, venue_size: str) -> float:
        """Get spectral sharpening strength based on venue size."""
        strengths = {
            'small': 0.1,   # Gentle sharpening for intimate venues
            'medium': 0.2,  # Moderate sharpening
            'large': 0.3    # Stronger sharpening for large venues
        }
        return strengths.get(venue_size, 0.2)
    
    def _get_formant_boost(self, formant_index: int, venue_size: str) -> float:
        """Get formant-specific boost in dB."""
        boosts = {
            'small': [1.0, 1.5, 1.0],   # F1, F2, F3 boosts
            'medium': [1.5, 2.0, 1.5],
            'large': [2.0, 2.5, 2.0]
        }
        venue_boosts = boosts.get(venue_size, boosts['medium'])
        return venue_boosts[min(formant_index, len(venue_boosts) - 1)]
    
    def _get_compression_parameters(self, venue_size: str) -> Dict[str, float]:
        """Get compression parameters based on venue size."""
        params = {
            'small': {
                'threshold': 0.8,
                'ratio': 2.0,
                'attack_time': 0.003,  # 3ms
                'release_time': 0.1    # 100ms
            },
            'medium': {
                'threshold': 0.7,
                'ratio': 3.0,
                'attack_time': 0.005,  # 5ms
                'release_time': 0.15   # 150ms
            },
            'large': {
                'threshold': 0.6,
                'ratio': 4.0,
                'attack_time': 0.01,   # 10ms
                'release_time': 0.2    # 200ms
            }
        }
        return params.get(venue_size, params['medium'])
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for dialogue enhancement."""
        return {
            'clarity': {
                'spectral_sharpening': True,
                'harmonic_enhancement': True,
                'transient_enhancement': True,
                'formant_enhancement': True
            },
            'dynamics': {
                'preserve_crest_factor': True,
                'adaptive_compression': True,
                'envelope_smoothing': True
            },
            'artifact_prevention': {
                'anti_aliasing': True,
                'transition_smoothing': True,
                'normalization': True
            }
        }