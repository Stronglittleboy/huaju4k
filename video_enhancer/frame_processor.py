"""
Frame Processing Module for Video Enhancement

This module handles frame extraction, preprocessing, and reassembly for video enhancement.
"""

import os
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import numpy as np

from .config import ConfigManager
from .models import ProcessingConfig


class FrameProcessor:
    """Handles frame extraction, preprocessing, and video reassembly."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the frame processor."""
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
    def extract_frames(self, video_path: str, output_dir: str, 
                      format: str = "png") -> Tuple[bool, int]:
        """
        Extract frames from video using FFmpeg.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted frames
            format: Output format for frames (png, jpg)
            
        Returns:
            Tuple of (success, frame_count)
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Construct FFmpeg command for frame extraction
            output_pattern = os.path.join(output_dir, f"frame_%08d.{format}")
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', 'scale=in_color_matrix=auto:out_color_matrix=bt709',  # Color space handling
                '-pix_fmt', 'rgb24',  # Ensure consistent pixel format
                '-y',  # Overwrite existing files
                output_pattern
            ]
            
            self.logger.info(f"Extracting frames from {video_path}")
            self.logger.info(f"Command: {' '.join(cmd)}")
            
            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Count extracted frames
            frame_count = len([f for f in os.listdir(output_dir) 
                             if f.startswith('frame_') and f.endswith(f'.{format}')])
            
            self.logger.info(f"Successfully extracted {frame_count} frames")
            return True, frame_count
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg frame extraction failed: {e.stderr}")
            return False, 0
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {str(e)}")
            return False, 0
    
    def extract_audio(self, video_path: str, output_path: str) -> bool:
        """
        Extract audio track from video.
        
        Args:
            video_path: Path to input video file
            output_path: Path for extracted audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Construct FFmpeg command for audio extraction
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'copy',  # Copy audio without re-encoding
                '-y',  # Overwrite existing files
                output_path
            ]
            
            self.logger.info(f"Extracting audio from {video_path}")
            
            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            self.logger.info(f"Successfully extracted audio to {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Audio extraction failed: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {str(e)}")
            return False
    
    def apply_preprocessing(self, frame_dir: str, processing_config: ProcessingConfig) -> bool:
        """
        Apply preprocessing filters to extracted frames.
        
        Args:
            frame_dir: Directory containing extracted frames
            processing_config: Processing configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            frame_files = sorted([f for f in os.listdir(frame_dir) 
                                if f.startswith('frame_') and f.endswith('.png')])
            
            if not frame_files:
                self.logger.error("No frames found for preprocessing")
                return False
            
            self.logger.info(f"Applying preprocessing to {len(frame_files)} frames")
            self.logger.info(f"Denoise level: {processing_config.denoise_level}")
            
            processed_count = 0
            for frame_file in frame_files:
                frame_path = os.path.join(frame_dir, frame_file)
                
                # Load frame
                frame = cv2.imread(frame_path)
                if frame is None:
                    self.logger.warning(f"Could not load frame: {frame_file}")
                    continue
                
                # Apply theater-specific preprocessing
                frame = self._apply_theater_preprocessing(frame, processing_config)
                
                # Save processed frame
                cv2.imwrite(frame_path, frame)
                processed_count += 1
                
                # Log progress every 100 frames
                if processed_count % 100 == 0:
                    self.logger.info(f"Processed {processed_count}/{len(frame_files)} frames")
            
            self.logger.info(f"Preprocessing completed for {processed_count} frames")
            return True
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            return False
    
    def _apply_denoising(self, frame: np.ndarray, denoise_level: int) -> np.ndarray:
        """Apply denoising to a frame."""
        if denoise_level == 1:
            # Light denoising
            return cv2.bilateralFilter(frame, 5, 50, 50)
        elif denoise_level == 2:
            # Moderate denoising
            return cv2.bilateralFilter(frame, 7, 75, 75)
        elif denoise_level >= 3:
            # Strong denoising
            denoised = cv2.bilateralFilter(frame, 9, 100, 100)
            return cv2.fastNlMeansDenoisingColored(denoised, None, 10, 10, 7, 21)
        else:
            return frame
    
    def _apply_theater_preprocessing(self, frame: np.ndarray, processing_config: ProcessingConfig) -> np.ndarray:
        """
        Apply theater-specific preprocessing optimized for stage lighting.
        
        Args:
            frame: Input frame
            processing_config: Processing configuration
            
        Returns:
            Preprocessed frame
        """
        # Apply denoising appropriate for theater lighting
        if processing_config.denoise_level > 0:
            frame = self._apply_denoising(frame, processing_config.denoise_level)
        
        # Apply theater-specific color correction
        frame = self._apply_theater_color_correction(frame)
        
        # Apply subtle sharpening for actor detail enhancement
        frame = self._apply_subtle_sharpening(frame)
        
        return frame
    
    def _apply_theater_color_correction(self, frame: np.ndarray) -> np.ndarray:
        """Apply color correction optimized for theater stage lighting."""
        # Convert to LAB color space for better color correction
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for contrast enhancement
        # Use smaller tile size for theater content to preserve lighting atmosphere
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6, 6))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge([l, a, b])
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply gentle gamma correction for theater lighting
        gamma = 1.1  # Slight brightening for dark theater scenes
        corrected = np.power(corrected / 255.0, 1.0 / gamma) * 255.0
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return corrected
    
    def _apply_subtle_sharpening(self, frame: np.ndarray) -> np.ndarray:
        """Apply subtle sharpening to enhance actor details without artifacts."""
        # Create unsharp mask
        gaussian = cv2.GaussianBlur(frame, (0, 0), 1.0)
        unsharp_mask = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
        
        # Blend with original to avoid over-sharpening
        sharpened = cv2.addWeighted(frame, 0.7, unsharp_mask, 0.3, 0)
        
        return sharpened
    
    def reassemble_video(self, frame_dir: str, audio_path: str, output_path: str,
                        frame_rate: float, resolution: Tuple[int, int]) -> bool:
        """
        Reassemble enhanced frames into final video with audio.
        
        Args:
            frame_dir: Directory containing enhanced frames
            audio_path: Path to extracted audio file
            output_path: Path for final output video
            frame_rate: Frame rate for output video
            resolution: Resolution tuple (width, height)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if enhanced frames exist
            frame_files = sorted([f for f in os.listdir(frame_dir) 
                                if f.startswith('frame_') and f.endswith('.png')])
            
            if not frame_files:
                self.logger.error("No enhanced frames found for reassembly")
                return False
            
            # Construct frame input pattern
            frame_pattern = os.path.join(frame_dir, "frame_%08d.png")
            
            # Construct FFmpeg command for video reassembly
            cmd = [
                'ffmpeg',
                '-framerate', str(frame_rate),
                '-i', frame_pattern,
                '-i', audio_path,
                '-c:v', 'libx264',  # H.264 video codec
                '-preset', 'slow',  # High quality encoding
                '-crf', '18',  # High quality (lower CRF = higher quality)
                '-pix_fmt', 'yuv420p',  # Compatible pixel format
                '-c:a', 'aac',  # AAC audio codec
                '-b:a', '192k',  # Audio bitrate
                '-movflags', '+faststart',  # Optimize for streaming
                '-y',  # Overwrite existing files
                output_path
            ]
            
            self.logger.info(f"Reassembling video with {len(frame_files)} frames")
            self.logger.info(f"Output resolution: {resolution[0]}x{resolution[1]}")
            self.logger.info(f"Frame rate: {frame_rate} fps")
            
            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Verify output file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                self.logger.info(f"Successfully created output video: {output_path}")
                self.logger.info(f"Output file size: {file_size / (1024*1024):.1f} MB")
                return True
            else:
                self.logger.error("Output video file was not created")
                return False
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Video reassembly failed: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Video reassembly failed: {str(e)}")
            return False
    
    def cleanup_intermediate_files(self, directories: List[str]) -> None:
        """
        Clean up intermediate processing files.
        
        Args:
            directories: List of directories to clean up
        """
        for directory in directories:
            if os.path.exists(directory):
                try:
                    shutil.rmtree(directory)
                    self.logger.info(f"Cleaned up directory: {directory}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {directory}: {str(e)}")
    
    def validate_frame_sequence(self, frame_dir: str, expected_count: int) -> bool:
        """
        Validate that frame sequence is complete and consistent.
        
        Args:
            frame_dir: Directory containing frames
            expected_count: Expected number of frames
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            frame_files = sorted([f for f in os.listdir(frame_dir) 
                                if f.startswith('frame_') and f.endswith('.png')])
            
            if len(frame_files) != expected_count:
                self.logger.error(f"Frame count mismatch: expected {expected_count}, "
                                f"found {len(frame_files)}")
                return False
            
            # Check for missing frames in sequence
            for i, frame_file in enumerate(frame_files, 1):
                expected_name = f"frame_{i:08d}.png"
                if frame_file != expected_name:
                    self.logger.error(f"Missing or misnamed frame: expected {expected_name}, "
                                    f"found {frame_file}")
                    return False
            
            # Validate frame integrity (sample check)
            sample_indices = [0, len(frame_files)//2, len(frame_files)-1]
            for idx in sample_indices:
                frame_path = os.path.join(frame_dir, frame_files[idx])
                frame = cv2.imread(frame_path)
                if frame is None:
                    self.logger.error(f"Corrupted frame detected: {frame_files[idx]}")
                    return False
            
            self.logger.info(f"Frame sequence validation passed: {len(frame_files)} frames")
            return True
            
        except Exception as e:
            self.logger.error(f"Frame validation failed: {str(e)}")
            return False