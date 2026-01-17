"""
Video Analysis Module for Theater Drama Enhancement

This module analyzes video files to extract metadata, assess quality characteristics,
and evaluate theater-specific content for optimal enhancement processing.
"""

import os
import subprocess
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

from .models import (
    VideoMetadata, AudioTrack, TheaterAnalysis, 
    LightingType, SceneComplexity, StageDepth, ProcessingPriority
)


class VideoAnalyzer:
    """Analyzes video files for enhancement processing."""
    
    def __init__(self):
        """Initialize the video analyzer."""
        self.logger = logging.getLogger(__name__)
        
    def analyze_video_file(self, video_path: str) -> VideoMetadata:
        """
        Perform complete analysis of a video file.
        
        Args:
            video_path: Path to the video file to analyze
            
        Returns:
            VideoMetadata with complete analysis results
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If analysis fails
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        self.logger.info(f"Starting analysis of video file: {video_path}")
        
        try:
            # Extract basic metadata using FFprobe
            metadata = self._extract_metadata_ffprobe(video_path)
            
            # Analyze theater-specific characteristics
            theater_analysis = self._analyze_theater_characteristics(video_path)
            
            # Combine results
            video_metadata = VideoMetadata(
                resolution=metadata['resolution'],
                frame_rate=metadata['frame_rate'],
                duration=metadata['duration'],
                codec=metadata['codec'],
                bitrate=metadata['bitrate'],
                audio_tracks=metadata['audio_tracks'],
                theater_characteristics=theater_analysis
            )
            
            self.logger.info("Video analysis completed successfully")
            return video_metadata
            
        except Exception as e:
            self.logger.error(f"Video analysis failed: {str(e)}")
            raise RuntimeError(f"Failed to analyze video: {str(e)}")
    
    def _extract_metadata_ffprobe(self, video_path: str) -> Dict[str, Any]:
        """
        Extract video metadata using FFprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing metadata
        """
        try:
            # Run ffprobe to get detailed video information
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)
            
            # Extract video stream information
            video_stream = None
            audio_streams = []
            
            for stream in probe_data['streams']:
                if stream['codec_type'] == 'video' and video_stream is None:
                    video_stream = stream
                elif stream['codec_type'] == 'audio':
                    audio_streams.append(stream)
            
            if not video_stream:
                raise RuntimeError("No video stream found in file")
            
            # Parse video metadata
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            # Parse frame rate
            frame_rate_str = video_stream.get('r_frame_rate', '25/1')
            if '/' in frame_rate_str:
                num, den = map(int, frame_rate_str.split('/'))
                frame_rate = num / den if den != 0 else 25.0
            else:
                frame_rate = float(frame_rate_str)
            
            # Parse duration
            duration = float(video_stream.get('duration', 
                           probe_data.get('format', {}).get('duration', 0)))
            
            # Parse bitrate
            bitrate = int(video_stream.get('bit_rate', 
                         probe_data.get('format', {}).get('bit_rate', 0)))
            
            # Parse audio tracks
            audio_tracks = []
            for audio_stream in audio_streams:
                audio_track = AudioTrack(
                    codec=audio_stream.get('codec_name', 'unknown'),
                    bitrate=int(audio_stream.get('bit_rate', 0)),
                    channels=int(audio_stream.get('channels', 2)),
                    sample_rate=int(audio_stream.get('sample_rate', 44100)),
                    language=audio_stream.get('tags', {}).get('language')
                )
                audio_tracks.append(audio_track)
            
            return {
                'resolution': (width, height),
                'frame_rate': frame_rate,
                'duration': duration,
                'codec': video_stream.get('codec_name', 'unknown'),
                'bitrate': bitrate,
                'audio_tracks': audio_tracks
            }
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFprobe failed: {e.stderr}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise RuntimeError(f"Failed to parse FFprobe output: {str(e)}")
    
    def _analyze_theater_characteristics(self, video_path: str) -> TheaterAnalysis:
        """
        Analyze theater-specific characteristics of the video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            TheaterAnalysis with theater-specific insights
        """
        try:
            # Open video for frame analysis
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open video file for analysis")
            
            # Sample frames for analysis (every 30 seconds)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_interval = int(frame_rate * 30)  # Every 30 seconds
            
            frames_analyzed = 0
            lighting_scores = {'stage': 0, 'natural': 0, 'mixed': 0}
            complexity_scores = {'simple': 0, 'moderate': 0, 'complex': 0}
            depth_scores = {'shallow': 0, 'medium': 0, 'deep': 0}
            motion_levels = []
            brightness_levels = []
            
            for frame_num in range(0, total_frames, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                frames_analyzed += 1
                
                # Analyze lighting characteristics
                lighting_type = self._analyze_frame_lighting(frame)
                lighting_scores[lighting_type] += 1
                
                # Analyze scene complexity
                complexity = self._analyze_frame_complexity(frame)
                complexity_scores[complexity] += 1
                
                # Analyze stage depth
                depth = self._analyze_stage_depth(frame)
                depth_scores[depth] += 1
                
                # Track brightness and motion for overall assessment
                brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                brightness_levels.append(brightness)
                
                # Limit analysis to prevent excessive processing
                if frames_analyzed >= 20:
                    break
            
            cap.release()
            
            if frames_analyzed == 0:
                raise RuntimeError("No frames could be analyzed")
            
            # Determine dominant characteristics
            lighting_type = LightingType(max(lighting_scores, key=lighting_scores.get))
            scene_complexity = SceneComplexity(max(complexity_scores, key=complexity_scores.get))
            stage_depth = StageDepth(max(depth_scores, key=depth_scores.get))
            
            # Estimate actor count (simplified heuristic)
            actor_count = self._estimate_actor_count(complexity_scores, frames_analyzed)
            
            # Determine recommended model and priority
            recommended_model = self._get_recommended_model(lighting_type, scene_complexity)
            processing_priority = self._get_processing_priority(scene_complexity, stage_depth)
            
            return TheaterAnalysis(
                lighting_type=lighting_type,
                scene_complexity=scene_complexity,
                actor_count=actor_count,
                stage_depth=stage_depth,
                recommended_model=recommended_model,
                processing_priority=processing_priority
            )
            
        except Exception as e:
            self.logger.error(f"Theater analysis failed: {str(e)}")
            # Return default analysis if detailed analysis fails
            return TheaterAnalysis(
                lighting_type=LightingType.MIXED,
                scene_complexity=SceneComplexity.MODERATE,
                actor_count=3,
                stage_depth=StageDepth.MEDIUM,
                recommended_model="realesrgan-x4plus",
                processing_priority=ProcessingPriority.BALANCED
            )
    
    def _analyze_frame_lighting(self, frame: np.ndarray) -> str:
        """Analyze lighting characteristics of a frame."""
        # Convert to HSV for better lighting analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Analyze brightness distribution
        brightness = hsv[:, :, 2]
        mean_brightness = np.mean(brightness)
        brightness_std = np.std(brightness)
        
        # Analyze color temperature (simplified)
        b, g, r = cv2.split(frame)
        color_temp_ratio = np.mean(r) / (np.mean(b) + 1)  # Avoid division by zero
        
        # Heuristic classification
        if brightness_std > 60 and color_temp_ratio > 1.2:
            return "stage"  # High contrast, warm lighting typical of stage
        elif mean_brightness > 120 and brightness_std < 40:
            return "natural"  # Even, bright lighting
        else:
            return "mixed"  # Mixed or complex lighting
    
    def _analyze_frame_complexity(self, frame: np.ndarray) -> str:
        """Analyze visual complexity of a frame."""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate texture complexity using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Heuristic classification
        if edge_density > 0.15 or laplacian_var > 1000:
            return "complex"
        elif edge_density > 0.08 or laplacian_var > 500:
            return "moderate"
        else:
            return "simple"
    
    def _analyze_stage_depth(self, frame: np.ndarray) -> str:
        """Analyze perceived depth of the stage setup."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Analyze depth cues using gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Analyze vertical distribution of features
        height, width = gray.shape
        top_third = gradient_magnitude[:height//3, :]
        middle_third = gradient_magnitude[height//3:2*height//3, :]
        bottom_third = gradient_magnitude[2*height//3:, :]
        
        top_activity = np.mean(top_third)
        middle_activity = np.mean(middle_third)
        bottom_activity = np.mean(bottom_third)
        
        # Heuristic classification based on depth perception
        depth_ratio = (top_activity + middle_activity) / (bottom_activity + 1)
        
        if depth_ratio > 1.5:
            return "deep"  # Activity distributed across depth planes
        elif depth_ratio > 0.8:
            return "medium"  # Moderate depth perception
        else:
            return "shallow"  # Mostly foreground activity
    
    def _estimate_actor_count(self, complexity_scores: Dict[str, int], frames_analyzed: int) -> int:
        """Estimate number of actors based on scene complexity."""
        # Simple heuristic based on complexity distribution
        complex_ratio = complexity_scores['complex'] / frames_analyzed
        moderate_ratio = complexity_scores['moderate'] / frames_analyzed
        
        if complex_ratio > 0.6:
            return 5  # Many actors or complex staging
        elif complex_ratio > 0.3 or moderate_ratio > 0.7:
            return 3  # Moderate number of actors
        else:
            return 2  # Few actors or simple staging
    
    def _get_recommended_model(self, lighting_type: LightingType, 
                             scene_complexity: SceneComplexity) -> str:
        """Get recommended AI model based on content characteristics."""
        # Model recommendations based on content analysis
        if lighting_type == LightingType.STAGE and scene_complexity == SceneComplexity.COMPLEX:
            return "realesrgan-x4plus"  # Best for complex stage lighting
        elif lighting_type == LightingType.NATURAL:
            return "real-cugan"  # Good for natural lighting
        else:
            return "realesrgan-x4plus"  # Default robust choice
    
    def _get_processing_priority(self, scene_complexity: SceneComplexity, 
                               stage_depth: StageDepth) -> ProcessingPriority:
        """Determine processing priority based on content characteristics."""
        if scene_complexity == SceneComplexity.COMPLEX or stage_depth == StageDepth.DEEP:
            return ProcessingPriority.QUALITY  # Complex content needs quality focus
        elif scene_complexity == SceneComplexity.SIMPLE:
            return ProcessingPriority.SPEED  # Simple content can prioritize speed
        else:
            return ProcessingPriority.BALANCED  # Balanced approach for moderate content
    
    def print_analysis_report(self, metadata: VideoMetadata) -> None:
        """Print a comprehensive analysis report."""
        print("\n" + "="*60)
        print("VIDEO ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nBASIC SPECIFICATIONS:")
        print(f"  Resolution: {metadata.width}x{metadata.height}")
        print(f"  Frame Rate: {metadata.frame_rate:.2f} fps")
        print(f"  Duration: {metadata.duration:.2f} seconds ({metadata.duration/60:.1f} minutes)")
        print(f"  Video Codec: {metadata.codec}")
        print(f"  Bitrate: {metadata.bitrate:,} bps ({metadata.bitrate/1000000:.1f} Mbps)")
        print(f"  4K Status: {'Already 4K' if metadata.is_4k else 'Needs upscaling'}")
        
        print(f"\nAUDIO TRACKS ({len(metadata.audio_tracks)}):")
        for i, audio in enumerate(metadata.audio_tracks):
            print(f"  Track {i+1}: {audio.codec}, {audio.channels} channels, "
                  f"{audio.sample_rate} Hz, {audio.bitrate:,} bps")
            if audio.language:
                print(f"    Language: {audio.language}")
        
        if metadata.theater_characteristics:
            tc = metadata.theater_characteristics
            print(f"\nTHEATER CHARACTERISTICS:")
            print(f"  Lighting Type: {tc.lighting_type.value.title()}")
            print(f"  Scene Complexity: {tc.scene_complexity.value.title()}")
            print(f"  Estimated Actor Count: {tc.actor_count}")
            print(f"  Stage Depth: {tc.stage_depth.value.title()}")
            print(f"  Recommended Model: {tc.recommended_model}")
            print(f"  Processing Priority: {tc.processing_priority.value.title()}")
        
        print("\n" + "="*60)