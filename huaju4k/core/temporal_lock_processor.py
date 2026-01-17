"""
Temporal lock processor for huaju4k theater enhancement.

This module implements optical flow-based temporal locking to ensure
frame stability and reduce flickering artifacts in enhanced videos.
Includes multi-layer temporal stabilization strategies and enhanced
background model construction.
"""

import logging
import cv2
import numpy as np
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path
import time
from enum import Enum

from ..models.data_models import TemporalConfig

logger = logging.getLogger(__name__)


class TemporalStabilizationMode(Enum):
    """Temporal stabilization modes."""
    OPTICAL_FLOW = "optical_flow"
    SIMPLE_SMOOTHING = "simple_smoothing"
    HYBRID = "hybrid"


class MotionDetector:
    """Enhanced motion detection utility for temporal processing."""
    
    def __init__(self, threshold: float = 0.05):
        """
        Initialize motion detector.
        
        Args:
            threshold: Motion detection threshold
        """
        self.threshold = threshold
    
    def detect_motion_regions(self, 
                            current_frame: np.ndarray,
                            previous_frame: np.ndarray) -> np.ndarray:
        """
        Detect motion regions between frames with enhanced accuracy.
        
        Args:
            current_frame: Current frame
            previous_frame: Previous frame
            
        Returns:
            Boolean mask of motion regions
        """
        try:
            # Convert to grayscale
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate frame difference
            diff = cv2.absdiff(current_gray, previous_gray)
            diff_normalized = diff.astype(np.float32) / 255.0
            
            # Apply threshold
            motion_mask = diff_normalized > self.threshold
            
            # Enhanced morphological operations for noise reduction
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            motion_mask = cv2.morphologyEx(motion_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
            
            # Additional dilation to ensure motion regions are well-defined
            motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)
            
            return motion_mask.astype(bool)
            
        except Exception as e:
            logger.error(f"Motion detection failed: {e}")
            return np.zeros(current_frame.shape[:2], dtype=bool)
    
    def detect_motion_with_optical_flow(self,
                                      current_frame: np.ndarray,
                                      previous_frame: np.ndarray,
                                      flow_params: Dict[str, Any]) -> np.ndarray:
        """
        Detect motion using optical flow magnitude.
        
        Args:
            current_frame: Current frame
            previous_frame: Previous frame
            flow_params: Optical flow parameters
            
        Returns:
            Boolean mask of motion regions based on optical flow
        """
        try:
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray, current_gray, None, **flow_params
            )
            
            # Calculate motion magnitude
            motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            # Apply threshold
            motion_mask = motion_magnitude > self.threshold
            
            return motion_mask
            
        except Exception as e:
            logger.error(f"Optical flow motion detection failed: {e}")
            return np.zeros(current_frame.shape[:2], dtype=bool)


class TemporalLockProcessor:
    """
    Enhanced optical flow-based temporal locking processor.
    
    This class implements advanced temporal stabilization using optical flow
    with multiple stabilization modes and enhanced background modeling.
    Includes multi-layer temporal locking strategies as specified in Task 5.
    """
    
    def __init__(self, temporal_config: Optional[TemporalConfig] = None):
        """
        Initialize temporal lock processor.
        
        Args:
            temporal_config: Optional temporal processing configuration
        """
        self.config = temporal_config or TemporalConfig()
        self.background_model = None
        self.motion_detector = MotionDetector(self.config.motion_threshold)
        
        # Enhanced optical flow parameters (as specified in Task 5)
        self.optical_flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        
        # Multi-layer stabilization settings
        self.stabilization_mode = TemporalStabilizationMode.HYBRID
        self.smoothing_coefficients = {
            "high": 0.1,    # Strong smoothing
            "medium": 0.3,  # Medium smoothing  
            "low": 0.5      # Light smoothing
        }
        
        # Performance tracking
        self.processing_stats = {
            'frames_processed': 0,
            'optical_flow_time': 0.0,
            'background_lock_time': 0.0,
            'total_processing_time': 0.0
        }
        
        logger.info("Enhanced TemporalLockProcessor initialized")
    
    def set_config(self, config: TemporalConfig) -> None:
        """
        Set temporal processing configuration.
        
        Args:
            config: New temporal configuration
        """
        self.config = config
        self.motion_detector.threshold = config.motion_threshold
        logger.debug(f"Temporal config updated: background_lock={config.background_lock}")
    
    def set_stabilization_mode(self, mode: TemporalStabilizationMode) -> None:
        """
        Set temporal stabilization mode.
        
        Args:
            mode: Stabilization mode to use
        """
        self.stabilization_mode = mode
        logger.info(f"Stabilization mode set to: {mode.value}")
    
    def process_video(self, 
                     input_path: str, 
                     output_path: str,
                     progress_callback: Optional[Callable] = None) -> bool:
        """
        Process video with enhanced optical flow-based temporal locking.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            progress_callback: Optional progress callback function
            
        Returns:
            True if processing completed successfully
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting enhanced temporal lock processing: {input_path} -> {output_path}")
            
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open input video: {input_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Build enhanced background model if enabled
            if self.config.background_lock:
                if progress_callback:
                    progress_callback(0.0, "Building enhanced background model")
                
                self.background_model = self._build_enhanced_background_model(input_path)
                
                if progress_callback:
                    progress_callback(0.1, "Enhanced background model built")
            
            # Process frames with enhanced temporal locking
            success = self._process_frames_with_enhanced_temporal_lock(
                cap, out, total_frames, progress_callback
            )
            
            # Cleanup
            cap.release()
            out.release()
            
            # Update processing statistics
            self.processing_stats['total_processing_time'] = time.time() - start_time
            
            if success:
                logger.info("Enhanced temporal lock processing completed successfully")
                logger.info(f"Processing statistics: {self.get_processing_statistics()}")
            else:
                logger.error("Enhanced temporal lock processing failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Enhanced temporal lock processing failed: {e}")
            return False
    
    def _build_enhanced_background_model(self, video_path: str) -> Optional[np.ndarray]:
        """
        Build enhanced background model with improved sampling strategy.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Enhanced background model as numpy array or None if failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            # Enhanced sampling strategy - sample 50 frames uniformly
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_interval = max(1, total_frames // 50)  # Sample ~50 frames
            
            frames = []
            frame_idx = 0
            
            logger.info(f"Building enhanced background model from {total_frames} frames")
            
            while len(frames) < 50 and frame_idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Pre-process frame for better background modeling
                # Apply slight Gaussian blur to reduce noise
                frame_processed = cv2.GaussianBlur(frame, (3, 3), 0.5)
                frames.append(frame_processed.astype(np.float32))
                
                frame_idx += sample_interval
            
            cap.release()
            
            if not frames:
                logger.warning("No frames sampled for enhanced background model")
                return None
            
            # Calculate median background with improved accuracy
            frames_array = np.array(frames)
            background = np.median(frames_array, axis=0).astype(np.uint8)
            
            # Apply additional smoothing to background model
            background = cv2.bilateralFilter(background, 5, 50, 50)
            
            logger.info(f"Enhanced background model built from {len(frames)} sampled frames")
            return background
            
        except Exception as e:
            logger.error(f"Enhanced background model building failed: {e}")
            return None
    
    def _process_frames_with_enhanced_temporal_lock(self, 
                                                  cap: cv2.VideoCapture,
                                                  out: cv2.VideoWriter,
                                                  total_frames: int,
                                                  progress_callback: Optional[Callable]) -> bool:
        """
        Process video frames with enhanced temporal locking strategies.
        
        Args:
            cap: Input video capture
            out: Output video writer
            total_frames: Total number of frames
            progress_callback: Progress callback function
            
        Returns:
            True if processing successful
        """
        try:
            previous_frame = None
            frame_idx = 0
            
            # Initialize frame buffer for multi-frame analysis
            frame_buffer = []
            buffer_size = 3  # Use 3-frame buffer for better stability
            
            while True:
                ret, current_frame = cap.read()
                if not ret:
                    break
                
                start_frame_time = time.time()
                
                # Apply enhanced temporal locking
                if previous_frame is not None:
                    stabilized_frame = self._apply_enhanced_temporal_stabilization(
                        current_frame, previous_frame, frame_buffer
                    )
                else:
                    stabilized_frame = current_frame
                
                # Update frame buffer
                frame_buffer.append(current_frame.copy())
                if len(frame_buffer) > buffer_size:
                    frame_buffer.pop(0)
                
                # Write stabilized frame
                out.write(stabilized_frame)
                
                # Update processing statistics
                frame_time = time.time() - start_frame_time
                self.processing_stats['frames_processed'] += 1
                
                # Update progress
                if progress_callback and frame_idx % 10 == 0:  # Update every 10 frames
                    progress = 0.1 + 0.9 * (frame_idx / total_frames)  # 0.1 reserved for background model
                    progress_callback(progress, f"Processing frame {frame_idx}/{total_frames}")
                
                previous_frame = current_frame
                frame_idx += 1
            
            logger.info(f"Processed {frame_idx} frames with enhanced temporal locking")
            return True
            
        except Exception as e:
            logger.error(f"Enhanced frame processing failed: {e}")
            return False
    
    def _apply_enhanced_temporal_stabilization(self, 
                                             current_frame: np.ndarray,
                                             previous_frame: np.ndarray,
                                             frame_buffer: List[np.ndarray]) -> np.ndarray:
        """
        Apply enhanced temporal stabilization with multiple strategies.
        
        Args:
            current_frame: Current frame to stabilize
            previous_frame: Previous frame for reference
            frame_buffer: Buffer of recent frames for multi-frame analysis
            
        Returns:
            Enhanced temporally stabilized frame
        """
        try:
            if self.stabilization_mode == TemporalStabilizationMode.OPTICAL_FLOW:
                return self._apply_optical_flow_stabilization_enhanced(
                    current_frame, previous_frame
                )
            elif self.stabilization_mode == TemporalStabilizationMode.SIMPLE_SMOOTHING:
                return self._apply_simple_temporal_stabilization(
                    current_frame, previous_frame
                )
            elif self.stabilization_mode == TemporalStabilizationMode.HYBRID:
                # Combine optical flow and simple smoothing
                flow_stabilized = self._apply_optical_flow_stabilization_enhanced(
                    current_frame, previous_frame
                )
                
                # Apply additional smoothing if background lock is enabled
                if self.config.background_lock and self.background_model is not None:
                    return self._apply_background_locking_enhanced(
                        flow_stabilized, previous_frame, frame_buffer
                    )
                else:
                    return self._apply_simple_temporal_stabilization(
                        flow_stabilized, previous_frame
                    )
            
            return current_frame
            
        except Exception as e:
            logger.error(f"Enhanced temporal stabilization failed: {e}")
            return current_frame
    
    def _apply_optical_flow_stabilization_enhanced(self, 
                                                 current_frame: np.ndarray,
                                                 previous_frame: np.ndarray) -> np.ndarray:
        """
        Apply enhanced optical flow-based stabilization with background remapping.
        
        Args:
            current_frame: Current frame
            previous_frame: Previous frame
            
        Returns:
            Enhanced stabilized frame
        """
        start_time = time.time()
        
        try:
            # Convert to grayscale for optical flow
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray, 
                current_gray, 
                None, 
                **self.optical_flow_params
            )
            
            # Enhanced motion detection using optical flow
            motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_mask = motion_magnitude > self.config.motion_threshold
            
            # Background region remapping as specified in Task 5
            if self.background_model is not None:
                h, w = flow.shape[:2]
                
                # Create remapping coordinates
                y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
                new_x = x_coords + flow[..., 0]
                new_y = y_coords + flow[..., 1]
                
                # Remap background using optical flow
                stabilized_background = cv2.remap(
                    self.background_model,
                    new_x, new_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )
                
                # Combine motion regions and stabilized background
                result = np.where(
                    motion_mask[..., np.newaxis],
                    current_frame,
                    stabilized_background
                )
                
                # Update processing statistics
                self.processing_stats['optical_flow_time'] += time.time() - start_time
                
                return result.astype(np.uint8)
            else:
                # Fallback to basic flow stabilization
                return self._stabilize_with_flow_enhanced(
                    current_frame, previous_frame, flow, motion_mask
                )
            
        except Exception as e:
            logger.error(f"Enhanced optical flow stabilization failed: {e}")
            return current_frame
    
    def _apply_simple_temporal_stabilization(self, 
                                           current_frame: np.ndarray,
                                           previous_frame: np.ndarray) -> np.ndarray:
        """
        Apply simple temporal stabilization using exponential weighted averaging.
        
        Args:
            current_frame: Current frame
            previous_frame: Previous frame
            
        Returns:
            Smoothed frame using exponential weighted averaging
        """
        try:
            # Detect motion regions
            motion_mask = self.motion_detector.detect_motion_regions(
                current_frame, previous_frame
            )
            
            # Get smoothing coefficient based on strength setting
            strength = getattr(self.config, 'strength', 'medium')
            alpha = self.smoothing_coefficients.get(strength, 0.3)
            
            # Apply exponential weighted averaging to background regions
            smoothed_frame = cv2.addWeighted(
                current_frame.astype(np.float32), 1 - alpha,
                previous_frame.astype(np.float32), alpha,
                0
            )
            
            # Combine motion regions and smoothed background
            result = np.where(
                motion_mask[..., np.newaxis],
                current_frame,
                smoothed_frame
            )
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Simple temporal stabilization failed: {e}")
            return current_frame
    
    def _apply_background_locking_enhanced(self, 
                                         frame: np.ndarray,
                                         previous_frame: np.ndarray,
                                         frame_buffer: List[np.ndarray]) -> np.ndarray:
        """
        Apply enhanced background locking with multi-frame analysis.
        
        Args:
            frame: Current frame
            previous_frame: Previous frame
            frame_buffer: Buffer of recent frames
            
        Returns:
            Background-locked frame with enhanced stability
        """
        start_time = time.time()
        
        try:
            if self.background_model is None:
                return frame
            
            # Enhanced motion detection using frame buffer
            motion_mask = self._detect_motion_multi_frame(frame, frame_buffer)
            
            # Expand motion mask to include nearby regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Larger kernel
            motion_mask_expanded = cv2.morphologyEx(
                motion_mask.astype(np.uint8), cv2.MORPH_DILATE, kernel, iterations=2
            ).astype(bool)
            
            # Create enhanced background-locked frame
            locked_frame = frame.copy()
            
            # Replace static regions with background model
            static_mask = ~motion_mask_expanded
            for c in range(3):  # RGB channels
                locked_frame[:, :, c] = np.where(
                    static_mask,
                    self.background_model[:, :, c],
                    frame[:, :, c]
                )
            
            # Apply enhanced boundary smoothing
            locked_frame = self._apply_boundary_smoothing_enhanced(
                locked_frame, frame, motion_mask_expanded
            )
            
            # Update processing statistics
            self.processing_stats['background_lock_time'] += time.time() - start_time
            
            return locked_frame
            
        except Exception as e:
            logger.error(f"Enhanced background locking failed: {e}")
            return frame
    
    def _stabilize_with_flow_enhanced(self, 
                                    current_frame: np.ndarray,
                                    previous_frame: np.ndarray,
                                    flow: np.ndarray,
                                    motion_mask: np.ndarray) -> np.ndarray:
        """
        Apply enhanced flow-based stabilization to frame.
        
        Args:
            current_frame: Current frame
            previous_frame: Previous frame
            flow: Optical flow field
            motion_mask: Motion detection mask
            
        Returns:
            Enhanced stabilized frame
        """
        try:
            # Calculate global motion compensation with improved accuracy
            static_regions = ~motion_mask
            
            if np.sum(static_regions) > 0:
                # Use robust statistics for better flow estimation
                flow_x_values = flow[static_regions, 0]
                flow_y_values = flow[static_regions, 1]
                
                # Use median for robustness against outliers
                flow_x_median = np.median(flow_x_values)
                flow_y_median = np.median(flow_y_values)
                
                # Additional filtering: remove extreme values
                flow_x_std = np.std(flow_x_values)
                flow_y_std = np.std(flow_y_values)
                
                # Clamp flow values to reasonable range
                flow_x_median = np.clip(flow_x_median, -flow_x_std*2, flow_x_std*2)
                flow_y_median = np.clip(flow_y_median, -flow_y_std*2, flow_y_std*2)
                
                # Create transformation matrix for global compensation
                M = np.float32([[1, 0, -flow_x_median], [0, 1, -flow_y_median]])
                
                # Apply global stabilization
                height, width = current_frame.shape[:2]
                stabilized = cv2.warpAffine(current_frame, M, (width, height))
                
                # Enhanced blending with adaptive alpha
                motion_ratio = np.sum(motion_mask) / motion_mask.size
                alpha = 0.8 - 0.3 * motion_ratio  # More stabilization for static scenes
                alpha = np.clip(alpha, 0.3, 0.8)
                
                result = cv2.addWeighted(stabilized, alpha, current_frame, 1-alpha, 0)
                
                return result
            else:
                # No static regions found, return original
                return current_frame
            
        except Exception as e:
            logger.error(f"Enhanced flow-based stabilization failed: {e}")
            return current_frame
    
    def _detect_motion_multi_frame(self, 
                                 current_frame: np.ndarray,
                                 frame_buffer: List[np.ndarray]) -> np.ndarray:
        """
        Detect motion using multiple frames for improved accuracy.
        
        Args:
            current_frame: Current frame
            frame_buffer: Buffer of recent frames
            
        Returns:
            Enhanced motion mask using multi-frame analysis
        """
        try:
            if len(frame_buffer) < 2:
                # Fallback to single frame motion detection
                if len(frame_buffer) == 1:
                    return self.motion_detector.detect_motion_regions(
                        current_frame, frame_buffer[0]
                    )
                else:
                    return np.zeros(current_frame.shape[:2], dtype=bool)
            
            # Multi-frame motion detection
            motion_masks = []
            
            # Compare with multiple previous frames
            for prev_frame in frame_buffer[-2:]:  # Use last 2 frames
                motion_mask = self.motion_detector.detect_motion_regions(
                    current_frame, prev_frame
                )
                motion_masks.append(motion_mask)
            
            # Combine motion masks using logical OR
            combined_mask = np.logical_or.reduce(motion_masks)
            
            # Apply temporal consistency filter
            # Dilate to ensure motion regions are well connected
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(
                combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
            ).astype(bool)
            
            return combined_mask
            
        except Exception as e:
            logger.error(f"Multi-frame motion detection failed: {e}")
            return np.zeros(current_frame.shape[:2], dtype=bool)
    
    def _apply_boundary_smoothing_enhanced(self, 
                                         locked_frame: np.ndarray,
                                         original_frame: np.ndarray,
                                         motion_mask: np.ndarray) -> np.ndarray:
        """
        Apply enhanced smoothing at motion/static boundaries.
        
        Args:
            locked_frame: Background-locked frame
            original_frame: Original frame
            motion_mask: Motion region mask
            
        Returns:
            Boundary-smoothed frame with enhanced transitions
        """
        try:
            # Find boundary regions with larger kernel for smoother transitions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            motion_dilated = cv2.morphologyEx(
                motion_mask.astype(np.uint8), cv2.MORPH_DILATE, kernel, iterations=2
            )
            motion_eroded = cv2.morphologyEx(
                motion_mask.astype(np.uint8), cv2.MORPH_ERODE, kernel, iterations=2
            )
            
            boundary_mask = (motion_dilated - motion_eroded) > 0
            
            # Apply enhanced Gaussian smoothing to boundary regions
            if np.sum(boundary_mask) > 0:
                # Use bilateral filter for edge-preserving smoothing
                smoothed = cv2.bilateralFilter(locked_frame, 9, 75, 75)
                
                # Create smooth transition using distance transform
                boundary_distance = cv2.distanceTransform(
                    (~boundary_mask).astype(np.uint8), cv2.DIST_L2, 5
                )
                boundary_weight = np.clip(boundary_distance / 10.0, 0, 1)
                
                # Apply weighted blending for smooth transitions
                for c in range(3):
                    locked_frame[:, :, c] = np.where(
                        boundary_mask,
                        (smoothed[:, :, c] * (1 - boundary_weight) + 
                         locked_frame[:, :, c] * boundary_weight).astype(np.uint8),
                        locked_frame[:, :, c]
                    )
            
            return locked_frame
            
        except Exception as e:
            logger.error(f"Enhanced boundary smoothing failed: {e}")
            return locked_frame
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get detailed temporal processing statistics.
        
        Returns:
            Dictionary with comprehensive processing statistics
        """
        stats = {
            'config': {
                'background_lock': self.config.background_lock,
                'motion_threshold': self.config.motion_threshold,
                'stabilization_mode': self.stabilization_mode.value,
                'optical_flow_params': self.optical_flow_params
            },
            'background_model_available': self.background_model is not None,
            'background_model_shape': self.background_model.shape if self.background_model is not None else None,
            'performance': self.processing_stats.copy()
        }
        
        # Calculate performance metrics
        if self.processing_stats['frames_processed'] > 0:
            stats['performance']['avg_time_per_frame'] = (
                self.processing_stats['total_processing_time'] / 
                self.processing_stats['frames_processed']
            )
            stats['performance']['fps'] = (
                self.processing_stats['frames_processed'] / 
                self.processing_stats['total_processing_time']
            ) if self.processing_stats['total_processing_time'] > 0 else 0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            'frames_processed': 0,
            'optical_flow_time': 0.0,
            'background_lock_time': 0.0,
            'total_processing_time': 0.0
        }
    
    def validate_optical_flow_parameters(self) -> Dict[str, bool]:
        """
        Validate optical flow parameters against Task 5 specifications.
        
        Returns:
            Dictionary with validation results
        """
        expected_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        
        validation_results = {}
        for param, expected_value in expected_params.items():
            actual_value = self.optical_flow_params.get(param)
            validation_results[f'{param}_correct'] = actual_value == expected_value
        
        validation_results['all_parameters_correct'] = all(validation_results.values())
        
        return validation_results