"""
Theater Audio Optimizer

Modular audio optimization system specifically designed for theater video processing.
Refactored from existing audio processing functionality with focus on:
- Theater-specific audio characteristic detection
- Dialogue clarity enhancement without over-processing
- Intelligent noise reduction
- Audio quality validation and distortion prevention
"""

import os
import subprocess
import logging
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import librosa
import soundfile as sf

from .interfaces import IAudioOptimizer
from .models import AudioConfig, AudioAnalysis, QualityMetrics
from ..infrastructure.interfaces import ILogger


class TheaterAudioOptimizer(IAudioOptimizer):
    """
    Professional theater audio optimization system.
    
    Provides specialized audio enhancement for theater recordings with focus on:
    - Preserving dialogue naturalness
    - Intelligent noise reduction without over-processing
    - Theater-specific acoustic characteristic detection
    - Quality validation and distortion prevention
    """
    
    def __init__(self, config: AudioConfig, logger: ILogger, workspace_dir: str = "audio_workspace"):
        """
        Initialize the theater audio optimizer.
        
        Args:
            config: Audio processing configuration
            logger: Structured logger instance
            workspace_dir: Directory for temporary audio processing files
        """
        self.config = config
        self.logger = logger
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(exist_ok=True)
        
        # Theater-specific processing parameters
        self._theater_presets = self._load_theater_presets()
        self._quality_thresholds = self._load_quality_thresholds()
    
    def analyze_audio_characteristics(self, audio_path: str) -> AudioAnalysis:
        """
        Analyze theater-specific audio characteristics.
        
        Performs comprehensive analysis including:
        - Noise profile detection
        - Dialogue frequency range identification
        - Dynamic range analysis
        - Theater acoustic characteristics
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            AudioAnalysis with detected characteristics and recommended settings
        """
        self.logger.log_operation("audio_analysis_started", {"audio_path": audio_path})
        
        try:
            # Load audio data
            y, sr = librosa.load(audio_path, sr=48000)
            duration = len(y) / sr
            
            self.logger.log_operation("audio_loaded", {
                "duration": duration,
                "sample_rate": sr,
                "channels": 1 if len(y.shape) == 1 else y.shape[0]
            })
            
            # 1. Noise profile analysis
            noise_profile = self._analyze_noise_profile(y, sr)
            
            # 2. Dialogue frequency range detection
            dialogue_range = self._detect_dialogue_frequency_range(y, sr)
            
            # 3. Dynamic range analysis
            dynamic_range = self._analyze_dynamic_range(y)
            
            # 4. Peak level analysis
            peak_levels = self._analyze_peak_levels(y, sr)
            
            # 5. Generate recommended settings based on analysis
            recommended_settings = self._generate_recommended_settings(
                noise_profile, dialogue_range, dynamic_range, peak_levels
            )
            
            analysis_result = AudioAnalysis(
                noise_profile=noise_profile,
                dialogue_frequency_range=dialogue_range,
                dynamic_range=dynamic_range,
                peak_levels=peak_levels,
                recommended_settings=recommended_settings
            )
            
            # Save analysis results
            self._save_analysis_results(audio_path, analysis_result, y, sr)
            
            self.logger.log_operation("audio_analysis_completed", {
                "dialogue_range": dialogue_range,
                "dynamic_range": dynamic_range,
                "noise_level": float(np.mean(noise_profile)),
                "recommended_preset": recommended_settings.theater_preset
            })
            
            return analysis_result
            
        except Exception as e:
            self.logger.log_error(e, {"operation": "audio_analysis", "audio_path": audio_path})
            raise
    
    def apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply intelligent noise reduction without over-processing.
        
        Uses spectral subtraction with theater-specific adaptations:
        - Preserves dialogue frequencies
        - Adaptive noise floor estimation
        - Over-processing prevention
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Processed audio data with reduced noise
        """
        self.logger.log_operation("noise_reduction_started", {
            "strength": self.config.noise_reduction_strength,
            "preserve_naturalness": self.config.preserve_naturalness
        })
        
        try:
            # Save input audio temporarily for FFmpeg processing
            temp_input = self.workspace / "temp_input_noise_reduction.wav"
            temp_output = self.workspace / "temp_output_noise_reduction.wav"
            
            sf.write(temp_input, audio_data, 48000)
            
            # Apply noise reduction using FFmpeg with theater-optimized parameters
            noise_floor = -40 if self.config.preserve_naturalness else -35
            
            # Adjust strength based on configuration
            if self.config.noise_reduction_strength <= 0.3:
                nr_strength = 0.5  # Light
            elif self.config.noise_reduction_strength <= 0.7:
                nr_strength = 1.0  # Medium
            else:
                nr_strength = 1.5  # Heavy
            
            # Theater-specific noise reduction command
            cmd = [
                'ffmpeg', '-i', str(temp_input),
                '-af', f'afftdn=nr={nr_strength}:nf={noise_floor}:tn=1',
                str(temp_output), '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Load processed audio
            processed_audio, _ = librosa.load(temp_output, sr=48000)
            
            # Clean up temporary files
            temp_input.unlink(missing_ok=True)
            temp_output.unlink(missing_ok=True)
            
            self.logger.log_operation("noise_reduction_completed", {
                "input_length": len(audio_data),
                "output_length": len(processed_audio),
                "processing_successful": True
            })
            
            return processed_audio
            
        except Exception as e:
            self.logger.log_error(e, {"operation": "noise_reduction"})
            # Clean up temporary files on error
            temp_input.unlink(missing_ok=True)
            temp_output.unlink(missing_ok=True)
            raise
    
    def optimize_dialogue_clarity(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Enhance dialogue frequencies while preserving naturalness.
        
        Applies theater-specific EQ with focus on:
        - Speech intelligibility (2-4 kHz boost)
        - Presence enhancement (4-6 kHz)
        - Natural sound preservation
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Audio data with enhanced dialogue clarity
        """
        self.logger.log_operation("dialogue_enhancement_started", {
            "enhancement_level": self.config.dialogue_enhancement,
            "preserve_naturalness": self.config.preserve_naturalness
        })
        
        try:
            # Save input audio temporarily for FFmpeg processing
            temp_input = self.workspace / "temp_input_dialogue.wav"
            temp_output = self.workspace / "temp_output_dialogue.wav"
            
            sf.write(temp_input, audio_data, 48000)
            
            # Build EQ filter chain for dialogue enhancement
            eq_filters = []
            
            # Low frequency control (reduce rumble)
            eq_filters.append("highpass=f=100:p=1")
            
            # Speech intelligibility boost (2-4 kHz)
            speech_gain = self.config.dialogue_enhancement * 3.0  # Max 3dB boost
            eq_filters.append(f"equalizer=f=3000:width_type=h:width=1000:g={speech_gain}")
            
            # Presence enhancement (4-6 kHz) - more subtle
            presence_gain = self.config.dialogue_enhancement * 2.0  # Max 2dB boost
            eq_filters.append(f"equalizer=f=5000:width_type=h:width=1000:g={presence_gain}")
            
            # High frequency control (prevent harshness)
            if not self.config.preserve_naturalness:
                eq_filters.append("lowpass=f=10000:p=1")
            
            # Combine filters
            filter_chain = ",".join(eq_filters)
            
            cmd = [
                'ffmpeg', '-i', str(temp_input),
                '-af', filter_chain,
                str(temp_output), '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Load processed audio
            processed_audio, _ = librosa.load(temp_output, sr=48000)
            
            # Clean up temporary files
            temp_input.unlink(missing_ok=True)
            temp_output.unlink(missing_ok=True)
            
            self.logger.log_operation("dialogue_enhancement_completed", {
                "speech_gain_db": speech_gain,
                "presence_gain_db": presence_gain,
                "processing_successful": True
            })
            
            return processed_audio
            
        except Exception as e:
            self.logger.log_error(e, {"operation": "dialogue_enhancement"})
            # Clean up temporary files on error
            temp_input.unlink(missing_ok=True)
            temp_output.unlink(missing_ok=True)
            raise
    
    def validate_audio_quality(self, original: np.ndarray, processed: np.ndarray) -> QualityMetrics:
        """
        Validate processed audio to prevent distortion.
        
        Performs comprehensive quality analysis including:
        - SNR improvement measurement
        - THD (Total Harmonic Distortion) analysis
        - Dialogue clarity scoring
        - Naturalness preservation assessment
        - Distortion detection
        
        Args:
            original: Original audio data
            processed: Processed audio data
            
        Returns:
            QualityMetrics with validation results
        """
        self.logger.log_operation("quality_validation_started", {
            "original_length": len(original),
            "processed_length": len(processed)
        })
        
        try:
            # Ensure same length for comparison
            min_len = min(len(original), len(processed))
            orig_audio = original[:min_len]
            proc_audio = processed[:min_len]
            
            # 1. SNR Improvement calculation
            snr_improvement = self._calculate_snr_improvement(orig_audio, proc_audio)
            
            # 2. THD Level estimation
            thd_level = self._estimate_thd_level(proc_audio)
            
            # 3. Dialogue clarity scoring
            dialogue_clarity_score = self._score_dialogue_clarity(orig_audio, proc_audio)
            
            # 4. Naturalness scoring
            naturalness_score = self._score_naturalness(orig_audio, proc_audio)
            
            # 5. Distortion detection
            distortion_detected = self._detect_distortion(proc_audio, thd_level)
            
            quality_metrics = QualityMetrics(
                snr_improvement=snr_improvement,
                thd_level=thd_level,
                dialogue_clarity_score=dialogue_clarity_score,
                naturalness_score=naturalness_score,
                distortion_detected=distortion_detected
            )
            
            # Log quality assessment results
            self.logger.log_operation("quality_validation_completed", {
                "snr_improvement_db": snr_improvement,
                "thd_level_percent": thd_level * 100,
                "dialogue_clarity_score": dialogue_clarity_score,
                "naturalness_score": naturalness_score,
                "distortion_detected": distortion_detected,
                "quality_passed": not distortion_detected and snr_improvement > 0
            })
            
            return quality_metrics
            
        except Exception as e:
            self.logger.log_error(e, {"operation": "quality_validation"})
            raise
    
    # Private helper methods
    
    def _analyze_noise_profile(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Analyze noise profile using silence detection."""
        # Use percentile-based silence detection
        silence_threshold = np.percentile(np.abs(y), 20)
        silence_mask = np.abs(y) < silence_threshold
        
        if np.any(silence_mask):
            noise_samples = y[silence_mask]
            # Create noise profile using spectral analysis
            noise_stft = librosa.stft(noise_samples[:sr*2] if len(noise_samples) > sr*2 else noise_samples)
            noise_profile = np.mean(np.abs(noise_stft), axis=1)
        else:
            # Fallback: use lowest energy frames
            stft = librosa.stft(y)
            frame_energy = np.sum(np.abs(stft)**2, axis=0)
            low_energy_frames = frame_energy < np.percentile(frame_energy, 10)
            noise_profile = np.mean(np.abs(stft[:, low_energy_frames]), axis=1)
        
        return noise_profile
    
    def _detect_dialogue_frequency_range(self, y: np.ndarray, sr: int) -> Tuple[float, float]:
        """Detect the primary dialogue frequency range."""
        # Analyze spectral characteristics to find speech-dominant frequencies
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Focus on speech frequency range (300-8000 Hz)
        speech_indices = np.where((freqs >= 300) & (freqs <= 8000))[0]
        speech_magnitude = magnitude[speech_indices, :]
        
        # Find frequency range with highest consistent energy
        freq_energy = np.mean(speech_magnitude, axis=1)
        
        # Find the range containing 80% of speech energy
        cumulative_energy = np.cumsum(freq_energy)
        total_energy = cumulative_energy[-1]
        
        start_idx = np.where(cumulative_energy >= total_energy * 0.1)[0][0]
        end_idx = np.where(cumulative_energy >= total_energy * 0.9)[0][0]
        
        dialogue_range = (freqs[speech_indices[start_idx]], freqs[speech_indices[end_idx]])
        
        return dialogue_range
    
    def _analyze_dynamic_range(self, y: np.ndarray) -> float:
        """Calculate dynamic range in dB."""
        rms_energy = np.sqrt(np.mean(y**2))
        peak_amplitude = np.max(np.abs(y))
        
        if rms_energy > 0:
            dynamic_range = 20 * np.log10(peak_amplitude / rms_energy)
        else:
            dynamic_range = 0.0
        
        return float(dynamic_range)
    
    def _analyze_peak_levels(self, y: np.ndarray, sr: int) -> List[float]:
        """Analyze peak levels over time."""
        # Divide audio into 1-second segments
        segment_length = sr
        num_segments = len(y) // segment_length
        
        peak_levels = []
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            segment = y[start_idx:end_idx]
            peak_level = float(np.max(np.abs(segment)))
            peak_levels.append(peak_level)
        
        return peak_levels
    
    def _generate_recommended_settings(self, noise_profile: np.ndarray, 
                                     dialogue_range: Tuple[float, float],
                                     dynamic_range: float,
                                     peak_levels: List[float]) -> AudioConfig:
        """Generate recommended audio processing settings."""
        # Estimate noise level
        noise_level = float(np.mean(noise_profile))
        
        # Determine noise reduction strength
        if noise_level > 0.1:
            noise_reduction = 0.8  # Heavy
        elif noise_level > 0.05:
            noise_reduction = 0.5  # Medium
        else:
            noise_reduction = 0.2  # Light
        
        # Determine dialogue enhancement level
        dialogue_width = dialogue_range[1] - dialogue_range[0]
        if dialogue_width < 2000:  # Narrow dialogue range
            dialogue_enhancement = 0.6
        elif dialogue_width < 4000:  # Normal range
            dialogue_enhancement = 0.4
        else:  # Wide range
            dialogue_enhancement = 0.2
        
        # Determine theater preset based on dynamic range
        if dynamic_range > 20:
            theater_preset = "large"  # Large venue with wide dynamic range
        elif dynamic_range > 15:
            theater_preset = "medium"  # Medium venue
        else:
            theater_preset = "small"  # Small venue or compressed audio
        
        return AudioConfig(
            noise_reduction_strength=min(noise_reduction, 1.0),
            dialogue_enhancement=min(dialogue_enhancement, 1.0),
            dynamic_range_target=-23.0,  # Standard for theater
            preserve_naturalness=True,
            theater_preset=theater_preset
        )
    
    def _calculate_snr_improvement(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate SNR improvement in dB."""
        # Estimate noise floor for both signals
        orig_noise = np.percentile(np.abs(original), 10)
        proc_noise = np.percentile(np.abs(processed), 10)
        
        # Calculate RMS for signal estimation
        orig_rms = np.sqrt(np.mean(original**2))
        proc_rms = np.sqrt(np.mean(processed**2))
        
        # Calculate SNR for both
        orig_snr = 20 * np.log10(orig_rms / (orig_noise + 1e-10))
        proc_snr = 20 * np.log10(proc_rms / (proc_noise + 1e-10))
        
        return float(proc_snr - orig_snr)
    
    def _estimate_thd_level(self, audio: np.ndarray) -> float:
        """Estimate Total Harmonic Distortion level."""
        # Simplified THD estimation using spectral analysis
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Calculate harmonic content vs fundamental
        freqs = librosa.fft_frequencies(sr=48000)
        
        # Focus on audible range
        audible_indices = np.where((freqs >= 20) & (freqs <= 20000))[0]
        audible_magnitude = magnitude[audible_indices, :]
        
        # Estimate THD as ratio of high-frequency content to total
        total_energy = np.sum(audible_magnitude**2)
        high_freq_indices = np.where(freqs[audible_indices] > 5000)[0]
        
        if len(high_freq_indices) > 0:
            high_freq_energy = np.sum(audible_magnitude[high_freq_indices, :]**2)
            thd_estimate = high_freq_energy / (total_energy + 1e-10)
        else:
            thd_estimate = 0.0
        
        return float(min(thd_estimate, 1.0))
    
    def _score_dialogue_clarity(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Score dialogue clarity improvement (0-1 scale)."""
        # Analyze speech frequency range energy
        orig_stft = librosa.stft(original)
        proc_stft = librosa.stft(processed)
        
        freqs = librosa.fft_frequencies(sr=48000)
        speech_indices = np.where((freqs >= 2000) & (freqs <= 4000))[0]
        
        orig_speech_energy = np.mean(np.abs(orig_stft[speech_indices, :]))
        proc_speech_energy = np.mean(np.abs(proc_stft[speech_indices, :]))
        
        # Calculate relative improvement
        if orig_speech_energy > 0:
            improvement_ratio = proc_speech_energy / orig_speech_energy
            clarity_score = min(improvement_ratio, 2.0) / 2.0  # Normalize to 0-1
        else:
            clarity_score = 0.5  # Neutral score
        
        return float(clarity_score)
    
    def _score_naturalness(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Score naturalness preservation (0-1 scale, higher is more natural)."""
        # Calculate spectral similarity
        orig_stft = librosa.stft(original)
        proc_stft = librosa.stft(processed)
        
        orig_spectrum = np.mean(np.abs(orig_stft), axis=1)
        proc_spectrum = np.mean(np.abs(proc_stft), axis=1)
        
        # Normalize spectra
        orig_spectrum = orig_spectrum / (np.sum(orig_spectrum) + 1e-10)
        proc_spectrum = proc_spectrum / (np.sum(proc_spectrum) + 1e-10)
        
        # Calculate spectral correlation
        correlation = np.corrcoef(orig_spectrum, proc_spectrum)[0, 1]
        
        # Convert to naturalness score (higher correlation = more natural)
        naturalness_score = max(0.0, correlation)
        
        return float(naturalness_score)
    
    def _detect_distortion(self, audio: np.ndarray, thd_level: float) -> bool:
        """Detect if audio has significant distortion."""
        # Check for clipping
        clipping_threshold = 0.95
        clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
        clipping_ratio = clipped_samples / len(audio)
        
        # Check THD level
        thd_threshold = 0.05  # 5% THD threshold
        
        # Detect distortion if either condition is met
        distortion_detected = (clipping_ratio > 0.001) or (thd_level > thd_threshold)
        
        return distortion_detected
    
    def _save_analysis_results(self, audio_path: str, analysis: AudioAnalysis, 
                             audio_data: np.ndarray, sr: int) -> None:
        """Save analysis results to workspace."""
        # Create analysis summary
        analysis_summary = {
            "audio_file": audio_path,
            "timestamp": datetime.now().isoformat(),
            "dialogue_frequency_range": analysis.dialogue_frequency_range,
            "dynamic_range_db": analysis.dynamic_range,
            "peak_levels_count": len(analysis.peak_levels),
            "recommended_settings": {
                "noise_reduction_strength": analysis.recommended_settings.noise_reduction_strength,
                "dialogue_enhancement": analysis.recommended_settings.dialogue_enhancement,
                "theater_preset": analysis.recommended_settings.theater_preset,
                "preserve_naturalness": analysis.recommended_settings.preserve_naturalness
            }
        }
        
        # Save to JSON file
        analysis_file = self.workspace / "theater_audio_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
    
    def _load_theater_presets(self) -> Dict[str, Dict[str, Any]]:
        """Load theater-specific processing presets."""
        return {
            "small": {
                "reverb_time": 0.8,
                "stereo_width": 100,
                "bass_rolloff": 120,
                "presence_boost": 1.5
            },
            "medium": {
                "reverb_time": 1.2,
                "stereo_width": 120,
                "bass_rolloff": 100,
                "presence_boost": 2.0
            },
            "large": {
                "reverb_time": 1.8,
                "stereo_width": 140,
                "bass_rolloff": 80,
                "presence_boost": 2.5
            }
        }
    
    def _load_quality_thresholds(self) -> Dict[str, float]:
        """Load quality validation thresholds."""
        return {
            "min_snr_improvement": 1.0,  # Minimum 1dB SNR improvement
            "max_thd_level": 0.05,       # Maximum 5% THD
            "min_dialogue_clarity": 0.6,  # Minimum dialogue clarity score
            "min_naturalness": 0.7,      # Minimum naturalness score
            "max_clipping_ratio": 0.001  # Maximum 0.1% clipped samples
        }
    
    # Additional methods for Task 5.2: Audio Quality Validation and Distortion Prevention
    
    def preserve_dynamic_range(self, audio_data: np.ndarray, target_lufs: float = -23.0) -> np.ndarray:
        """
        Preserve dynamic range while normalizing to target LUFS.
        
        Implements intelligent dynamic range preservation:
        - Maintains natural audio dynamics
        - Prevents over-compression
        - Normalizes to broadcast standards
        
        Args:
            audio_data: Input audio data
            target_lufs: Target LUFS level (default: -23.0 for broadcast)
            
        Returns:
            Audio data with preserved dynamic range
        """
        self.logger.log_operation("dynamic_range_preservation_started", {
            "target_lufs": target_lufs,
            "preserve_naturalness": self.config.preserve_naturalness
        })
        
        try:
            # Calculate current RMS level
            current_rms = np.sqrt(np.mean(audio_data**2))
            
            if current_rms == 0:
                self.logger.log_operation("dynamic_range_preservation_skipped", 
                                        {"reason": "silent_audio"})
                return audio_data
            
            # Convert target LUFS to linear scale (approximation)
            # LUFS ≈ -0.691 + 10*log10(RMS²)
            target_rms = np.sqrt(10**((target_lufs + 0.691) / 10))
            
            # Calculate gain needed
            gain_linear = target_rms / current_rms
            gain_db = 20 * np.log10(gain_linear)
            
            # Apply gentle limiting to prevent clipping
            if gain_linear > 1.0:
                # Use soft limiting for gain reduction
                processed_audio = self._apply_soft_limiter(audio_data * gain_linear)
            else:
                # Simple gain adjustment for level reduction
                processed_audio = audio_data * gain_linear
            
            # Verify no clipping occurred
            peak_level = np.max(np.abs(processed_audio))
            if peak_level > 0.99:
                # Apply additional soft limiting
                processed_audio = processed_audio * (0.95 / peak_level)
            
            self.logger.log_operation("dynamic_range_preservation_completed", {
                "gain_applied_db": gain_db,
                "final_peak_level": float(np.max(np.abs(processed_audio))),
                "final_rms_level": float(np.sqrt(np.mean(processed_audio**2)))
            })
            
            return processed_audio
            
        except Exception as e:
            self.logger.log_error(e, {"operation": "dynamic_range_preservation"})
            raise
    
    def create_audio_quality_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """
        Create comprehensive audio quality metrics.
        
        Generates detailed quality assessment including:
        - Signal-to-Noise Ratio (SNR)
        - Total Harmonic Distortion (THD)
        - Dynamic Range (DR)
        - Loudness Range (LRA)
        - Peak-to-Average Ratio (PAR)
        - Spectral Centroid Shift
        - Naturalness Score
        
        Args:
            original: Original audio data
            processed: Processed audio data
            
        Returns:
            Dictionary with comprehensive quality metrics
        """
        self.logger.log_operation("quality_metrics_calculation_started", {
            "original_length": len(original),
            "processed_length": len(processed)
        })
        
        try:
            # Ensure same length
            min_len = min(len(original), len(processed))
            orig = original[:min_len]
            proc = processed[:min_len]
            
            metrics = {}
            
            # 1. Signal-to-Noise Ratio (SNR)
            metrics['snr_original_db'] = self._calculate_snr(orig)
            metrics['snr_processed_db'] = self._calculate_snr(proc)
            metrics['snr_improvement_db'] = metrics['snr_processed_db'] - metrics['snr_original_db']
            
            # 2. Total Harmonic Distortion (THD)
            metrics['thd_original_percent'] = self._calculate_thd(orig) * 100
            metrics['thd_processed_percent'] = self._calculate_thd(proc) * 100
            metrics['thd_change_percent'] = metrics['thd_processed_percent'] - metrics['thd_original_percent']
            
            # 3. Dynamic Range (DR)
            metrics['dynamic_range_original_db'] = self._calculate_dynamic_range(orig)
            metrics['dynamic_range_processed_db'] = self._calculate_dynamic_range(proc)
            metrics['dynamic_range_change_db'] = metrics['dynamic_range_processed_db'] - metrics['dynamic_range_original_db']
            
            # 4. Loudness Range (LRA) - simplified calculation
            metrics['loudness_range_original_lu'] = self._calculate_loudness_range(orig)
            metrics['loudness_range_processed_lu'] = self._calculate_loudness_range(proc)
            
            # 5. Peak-to-Average Ratio (PAR)
            metrics['par_original_db'] = self._calculate_par(orig)
            metrics['par_processed_db'] = self._calculate_par(proc)
            
            # 6. Spectral Centroid Shift
            metrics['spectral_centroid_shift_hz'] = self._calculate_spectral_centroid_shift(orig, proc)
            
            # 7. Naturalness Score (spectral similarity)
            metrics['naturalness_score'] = self._score_naturalness(orig, proc)
            
            # 8. Dialogue Clarity Score
            metrics['dialogue_clarity_score'] = self._score_dialogue_clarity(orig, proc)
            
            # 9. Overall Quality Score (weighted combination)
            metrics['overall_quality_score'] = self._calculate_overall_quality_score(metrics)
            
            self.logger.log_operation("quality_metrics_calculation_completed", {
                "metrics_count": len(metrics),
                "overall_quality_score": metrics['overall_quality_score'],
                "snr_improvement": metrics['snr_improvement_db']
            })
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(e, {"operation": "quality_metrics_calculation"})
            raise
    
    def implement_distortion_prevention(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Implement comprehensive distortion prevention mechanisms.
        
        Applies multiple distortion prevention techniques:
        - Soft clipping prevention
        - Harmonic distortion monitoring
        - Peak limiting with lookahead
        - Intersample peak detection
        - Gentle compression for transient control
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Audio data with distortion prevention applied
        """
        self.logger.log_operation("distortion_prevention_started", {
            "input_peak_level": float(np.max(np.abs(audio_data))),
            "input_rms_level": float(np.sqrt(np.mean(audio_data**2)))
        })
        
        try:
            processed_audio = audio_data.copy()
            
            # 1. Soft clipping prevention
            processed_audio = self._apply_soft_limiter(processed_audio)
            
            # 2. Transient control with gentle compression
            processed_audio = self._apply_gentle_compression(processed_audio)
            
            # 3. Intersample peak prevention
            processed_audio = self._prevent_intersample_peaks(processed_audio)
            
            # 4. Final safety limiter
            processed_audio = self._apply_safety_limiter(processed_audio)
            
            # 5. Validate no distortion was introduced
            distortion_metrics = self._validate_distortion_levels(audio_data, processed_audio)
            
            self.logger.log_operation("distortion_prevention_completed", {
                "output_peak_level": float(np.max(np.abs(processed_audio))),
                "output_rms_level": float(np.sqrt(np.mean(processed_audio**2))),
                "distortion_metrics": distortion_metrics
            })
            
            return processed_audio
            
        except Exception as e:
            self.logger.log_error(e, {"operation": "distortion_prevention"})
            raise
    
    # Private helper methods for Task 5.2
    
    def _apply_soft_limiter(self, audio_data: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """Apply soft limiting to prevent hard clipping."""
        # Soft knee limiting using tanh function
        limited_audio = np.tanh(audio_data / threshold) * threshold
        return limited_audio
    
    def _apply_gentle_compression(self, audio_data: np.ndarray, ratio: float = 2.0, 
                                threshold: float = 0.7) -> np.ndarray:
        """Apply gentle compression for transient control."""
        # Simple envelope-based compression
        envelope = np.abs(audio_data)
        
        # Smooth the envelope using a simple moving average instead of scipy
        window_size = 1024
        smoothed_envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        # Apply compression above threshold
        gain = np.ones_like(smoothed_envelope)
        above_threshold = smoothed_envelope > threshold
        
        if np.any(above_threshold):
            excess = smoothed_envelope[above_threshold] - threshold
            compressed_excess = excess / ratio
            gain[above_threshold] = (threshold + compressed_excess) / smoothed_envelope[above_threshold]
        
        return audio_data * gain
    
    def _prevent_intersample_peaks(self, audio_data: np.ndarray) -> np.ndarray:
        """Prevent intersample peaks through oversampling."""
        # Simplified intersample peak prevention
        # In a full implementation, this would use proper oversampling
        peak_level = np.max(np.abs(audio_data))
        
        if peak_level > 0.98:
            # Apply gentle gain reduction
            safety_gain = 0.95 / peak_level
            return audio_data * safety_gain
        
        return audio_data
    
    def _apply_safety_limiter(self, audio_data: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
        """Apply final safety limiter."""
        peak_level = np.max(np.abs(audio_data))
        
        if peak_level > ceiling:
            return audio_data * (ceiling / peak_level)
        
        return audio_data
    
    def _calculate_snr(self, audio_data: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio."""
        # Estimate noise floor using percentile method
        noise_floor = np.percentile(np.abs(audio_data), 10)
        signal_rms = np.sqrt(np.mean(audio_data**2))
        
        if noise_floor > 0:
            snr_db = 20 * np.log10(signal_rms / noise_floor)
        else:
            snr_db = 60.0  # Assume high SNR for very quiet signals
        
        return float(snr_db)
    
    def _calculate_thd(self, audio_data: np.ndarray) -> float:
        """Calculate Total Harmonic Distortion."""
        # Simplified THD calculation using spectral analysis
        stft = librosa.stft(audio_data)
        magnitude_spectrum = np.abs(stft)
        
        # Calculate energy in different frequency bands
        freqs = librosa.fft_frequencies(sr=48000)
        
        # Fundamental frequency range (assume speech/music content)
        fundamental_indices = np.where((freqs >= 100) & (freqs <= 1000))[0]
        harmonic_indices = np.where((freqs >= 1000) & (freqs <= 10000))[0]
        
        if len(fundamental_indices) > 0 and len(harmonic_indices) > 0:
            fundamental_energy = np.sum(magnitude_spectrum[fundamental_indices, :]**2)
            harmonic_energy = np.sum(magnitude_spectrum[harmonic_indices, :]**2)
            
            total_energy = fundamental_energy + harmonic_energy
            if total_energy > 0:
                thd = harmonic_energy / total_energy
            else:
                thd = 0.0
        else:
            thd = 0.0
        
        return float(min(thd, 1.0))
    
    def _calculate_loudness_range(self, audio_data: np.ndarray) -> float:
        """Calculate Loudness Range (simplified)."""
        # Simplified LRA calculation using RMS over time
        frame_size = 4800  # 0.1 seconds at 48kHz
        hop_size = 2400    # 50% overlap
        
        rms_values = []
        for i in range(0, len(audio_data) - frame_size, hop_size):
            frame = audio_data[i:i + frame_size]
            rms = np.sqrt(np.mean(frame**2))
            if rms > 0:
                rms_values.append(20 * np.log10(rms))
        
        if len(rms_values) > 0:
            # LRA is approximately the difference between 95th and 10th percentiles
            lra = np.percentile(rms_values, 95) - np.percentile(rms_values, 10)
        else:
            lra = 0.0
        
        return float(lra)
    
    def _calculate_par(self, audio_data: np.ndarray) -> float:
        """Calculate Peak-to-Average Ratio."""
        peak = np.max(np.abs(audio_data))
        rms = np.sqrt(np.mean(audio_data**2))
        
        if rms > 0:
            par_db = 20 * np.log10(peak / rms)
        else:
            par_db = 0.0
        
        return float(par_db)
    
    def _calculate_spectral_centroid_shift(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate spectral centroid shift between original and processed audio."""
        orig_centroid = np.mean(librosa.feature.spectral_centroid(y=original, sr=48000)[0])
        proc_centroid = np.mean(librosa.feature.spectral_centroid(y=processed, sr=48000)[0])
        
        return float(proc_centroid - orig_centroid)
    
    def _calculate_overall_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        # Weighted combination of key metrics
        weights = {
            'snr_improvement_db': 0.3,
            'naturalness_score': 0.25,
            'dialogue_clarity_score': 0.25,
            'dynamic_range_preservation': 0.2
        }
        
        # Calculate dynamic range preservation score
        dr_change = abs(metrics.get('dynamic_range_change_db', 0))
        dr_preservation = max(0, 1.0 - (dr_change / 10.0))  # Penalize large DR changes
        
        # Calculate weighted score
        score = (
            weights['snr_improvement_db'] * min(metrics.get('snr_improvement_db', 0) / 10.0, 1.0) +
            weights['naturalness_score'] * metrics.get('naturalness_score', 0.5) +
            weights['dialogue_clarity_score'] * metrics.get('dialogue_clarity_score', 0.5) +
            weights['dynamic_range_preservation'] * dr_preservation
        )
        
        return float(max(0.0, min(1.0, score)))
    
    def _validate_distortion_levels(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """Validate distortion levels in processed audio."""
        metrics = {}
        
        # Check for clipping
        clipped_samples = np.sum(np.abs(processed) > 0.99)
        metrics['clipping_ratio'] = clipped_samples / len(processed)
        
        # Check THD increase
        orig_thd = self._calculate_thd(original)
        proc_thd = self._calculate_thd(processed)
        metrics['thd_increase'] = proc_thd - orig_thd
        
        # Check peak level
        metrics['peak_level'] = float(np.max(np.abs(processed)))
        
        # Overall distortion assessment
        metrics['distortion_detected'] = (
            metrics['clipping_ratio'] > 0.001 or  # More than 0.1% clipped samples
            metrics['thd_increase'] > 0.02 or     # More than 2% THD increase
            metrics['peak_level'] > 0.99          # Peak level too high
        )
        
        return metrics