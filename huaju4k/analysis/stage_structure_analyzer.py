"""
Stage Structure Analyzer for Theater Enhancement

This module analyzes video structure to extract objective numerical features
for theater-grade enhancement strategy generation.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json

from ..models.data_models import StructureFeatures

logger = logging.getLogger(__name__)


class StageStructureAnalyzer:
    """
    Analyzes stage structure features from video files.
    
    This analyzer extracts objective numerical features without making
    subjective judgments about video content.
    """
    
    def __init__(self, sample_frames: int = 30):
        """
        Initialize the analyzer.
        
        Args:
            sample_frames: Number of frames to sample for analysis (optimized to 30 for speed)
        """
        self.sample_frames = sample_frames
        
    def analyze_structure(self, video_path: str) -> StructureFeatures:
        """
        Main analysis method that returns stage structure features.
        
        Args:
            video_path: Path to the video file to analyze
            
        Returns:
            StructureFeatures: Structured features data
            
        Raises:
            RuntimeError: If video cannot be opened or analyzed
        """
        logger.info(f"Starting stage structure analysis for: {video_path}")
        
        # Validate input
        if not Path(video_path).exists():
            raise RuntimeError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        try:
            # Get basic video info
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video info: {width}x{height}, {fps}fps, {duration:.1f}s, {total_frames} frames")
            
            # Sample frames for analysis
            frames = self._sample_frames(cap, total_frames)
            
            if not frames:
                raise RuntimeError("No frames could be sampled from video")
            
            logger.info(f"Sampled {len(frames)} frames for analysis")
            
            # Analyze lighting structure
            lighting_analysis = self._analyze_lighting_structure(frames)
            
            # Analyze edge density
            edge_density = self._analyze_edge_density(frames)
            
            # Analyze frame changes
            motion_analysis = self._analyze_frame_changes(frames)
            
            # Analyze noise level
            noise_score = self._analyze_noise_level(frames)
            
            # Create structure features
            features = StructureFeatures(
                # Basic video info
                resolution=(width, height),
                fps=fps,
                duration=duration,
                total_frames=total_frames,
                
                # Stage structure features
                is_static_camera=motion_analysis['frame_diff_mean'] < 0.02,
                highlight_ratio=lighting_analysis['highlight_ratio'],
                dark_ratio=lighting_analysis['dark_ratio'],
                midtone_ratio=lighting_analysis['midtone_ratio'],
                edge_density=edge_density,
                frame_diff_mean=motion_analysis['frame_diff_mean'],
                noise_score=noise_score,
                
                # Analysis metadata
                sample_frames=len(frames),
                analysis_timestamp=datetime.now()
            )
            
            logger.info(f"Analysis completed: static_camera={features.is_static_camera}, "
                       f"highlight_ratio={features.highlight_ratio:.3f}, "
                       f"edge_density={features.edge_density:.3f}")
            
            return features
            
        finally:
            cap.release()
    
    def _sample_frames(self, cap: cv2.VideoCapture, total_frames: int) -> List[np.ndarray]:
        """
        Sample frames uniformly from the video.
        
        Args:
            cap: OpenCV video capture object
            total_frames: Total number of frames in video
            
        Returns:
            List of sampled frames
        """
        frames = []
        
        if total_frames <= self.sample_frames:
            # If video has fewer frames than sample size, use all frames
            frame_indices = list(range(total_frames))
        else:
            # Sample uniformly across the video
            step = total_frames // self.sample_frames
            frame_indices = [i * step for i in range(self.sample_frames)]
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frames.append(frame)
            else:
                logger.warning(f"Failed to read frame at index {frame_idx}")
        
        return frames
    
    def _analyze_lighting_structure(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze lighting structure: highlight ratio, dark ratio, midtone ratio.
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary with lighting analysis results
        """
        highlight_ratios = []
        dark_ratios = []
        midtone_ratios = []
        
        for frame in frames:
            # Convert to grayscale for luminance analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            normalized = gray.astype(np.float32) / 255.0
            
            # Calculate ratios using thresholds
            highlight_mask = normalized > 0.85  # High threshold for highlights
            dark_mask = normalized < 0.15       # Low threshold for shadows
            midtone_mask = (normalized >= 0.15) & (normalized <= 0.85)
            
            total_pixels = normalized.size
            
            highlight_ratio = np.sum(highlight_mask) / total_pixels
            dark_ratio = np.sum(dark_mask) / total_pixels
            midtone_ratio = np.sum(midtone_mask) / total_pixels
            
            highlight_ratios.append(highlight_ratio)
            dark_ratios.append(dark_ratio)
            midtone_ratios.append(midtone_ratio)
        
        return {
            'highlight_ratio': float(np.mean(highlight_ratios)),
            'dark_ratio': float(np.mean(dark_ratios)),
            'midtone_ratio': float(np.mean(midtone_ratios))
        }
    
    def _analyze_edge_density(self, frames: List[np.ndarray]) -> float:
        """
        Analyze edge density using Canny edge detector.
        
        Args:
            frames: List of video frames
            
        Returns:
            Average edge density (0-1)
        """
        edge_densities = []
        
        for frame in frames:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.size
            edge_density = edge_pixels / total_pixels
            
            edge_densities.append(edge_density)
        
        return float(np.mean(edge_densities))
    
    def _analyze_frame_changes(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze frame-to-frame changes to detect static camera and motion.
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary with motion analysis results
        """
        if len(frames) < 2:
            return {'frame_diff_mean': 0.0}
        
        frame_diffs = []
        
        for i in range(1, len(frames)):
            # Convert frames to grayscale
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Normalize and calculate mean difference
            diff_normalized = diff.astype(np.float32) / 255.0
            mean_diff = np.mean(diff_normalized)
            
            frame_diffs.append(mean_diff)
        
        return {
            'frame_diff_mean': float(np.mean(frame_diffs))
        }
    
    def _analyze_noise_level(self, frames: List[np.ndarray]) -> float:
        """
        Analyze noise level using Laplacian variance method.
        
        Args:
            frames: List of video frames
            
        Returns:
            Average noise score (0-1, higher means more noise)
        """
        noise_scores = []
        
        for frame in frames:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            # Normalize variance to 0-1 range (empirically determined scaling)
            # Higher variance indicates more noise/detail
            normalized_score = min(1.0, variance / 1000.0)
            
            noise_scores.append(normalized_score)
        
        return float(np.mean(noise_scores))