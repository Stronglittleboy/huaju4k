"""
Video Processor Implementation

Integrates existing video processing functionality into the modular architecture.
"""

import os
import subprocess
import shutil
import logging
import json
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import cv2
import numpy as np

from .interfaces import IVideoProcessor
from .models import VideoConfig
from ..infrastructure.interfaces import ILogger, IProgressTracker


class VideoProcessor(IVideoProcessor):
    """Video processing implementation integrating existing functionality."""
    
    def __init__(self, config: VideoConfig, logger: ILogger, progress_tracker: IProgressTracker):
        """Initialize video processor.
        
        Args:
            config: Video processing configuration
            logger: Logging service
            progress_tracker: Progress tracking service
        """
        self.config = config
        self.logger = logger
        self.progress_tracker = progress_tracker
    
    def extract_frames(self, video_path: str, output_dir: str) -> List[str]:
        """Extract video frames for processing.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted frames
            
        Returns:
            List of paths to extracted frame files
        """
        try:
            self.logger.log_operation("frame_extraction_started", {
                "video_path": video_path,
                "output_dir": output_dir
            })
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Start progress tracking
            task_id = "frame_extraction"
            self.progress_tracker.start_task(task_id, 100, "Extracting video frames")
            
            # Construct FFmpeg command for frame extraction
            output_pattern = os.path.join(output_dir, "frame_%08d.png")
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', 'scale=in_color_matrix=auto:out_color_matrix=bt709',
                '-pix_fmt', 'rgb24',
                '-y',  # Overwrite existing files
                output_pattern
            ]
            
            self.logger.log_operation("ffmpeg_command", {"command": " ".join(cmd)})
            
            # Update progress
            self.progress_tracker.update_progress(task_id, 25, "Running FFmpeg extraction")
            
            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Update progress
            self.progress_tracker.update_progress(task_id, 75, "Counting extracted frames")
            
            # Get list of extracted frame files
            frame_files = []
            if os.path.exists(output_dir):
                frame_files = sorted([
                    os.path.join(output_dir, f) 
                    for f in os.listdir(output_dir) 
                    if f.startswith('frame_') and f.endswith('.png')
                ])
            
            # Complete progress tracking
            self.progress_tracker.update_progress(task_id, 100, f"Extracted {len(frame_files)} frames")
            self.progress_tracker.complete_task(task_id, True, f"Successfully extracted {len(frame_files)} frames")
            
            self.logger.log_operation("frame_extraction_completed", {
                "frame_count": len(frame_files),
                "success": True
            })
            
            return frame_files
            
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg frame extraction failed: {e.stderr}"
            self.logger.log_error(Exception(error_msg), {"video_path": video_path})
            self.progress_tracker.complete_task(task_id, False, error_msg)
            return []
        except Exception as e:
            error_msg = f"Frame extraction failed: {str(e)}"
            self.logger.log_error(e, {"video_path": video_path})
            if 'task_id' in locals():
                self.progress_tracker.complete_task(task_id, False, error_msg)
            return []
    
    def enhance_frames(self, frame_paths: List[str]) -> List[str]:
        """Apply AI enhancement to frames.
        
        Args:
            frame_paths: List of paths to frame files
            
        Returns:
            List of paths to enhanced frame files
        """
        try:
            if not frame_paths:
                return []
            
            self.logger.log_operation("frame_enhancement_started", {
                "frame_count": len(frame_paths),
                "ai_model": self.config.ai_model
            })
            
            # Start progress tracking
            task_id = "frame_enhancement"
            self.progress_tracker.start_task(task_id, len(frame_paths), "Enhancing frames with AI")
            
            # Determine output directory
            first_frame_dir = os.path.dirname(frame_paths[0])
            enhanced_dir = os.path.join(os.path.dirname(first_frame_dir), "enhanced")
            os.makedirs(enhanced_dir, exist_ok=True)
            
            enhanced_frames = []
            
            # Process frames based on available AI models
            if self._is_real_esrgan_available():
                enhanced_frames = self._enhance_with_real_esrgan(frame_paths, enhanced_dir, task_id)
            else:
                # Fallback to OpenCV enhancement
                enhanced_frames = self._enhance_with_opencv(frame_paths, enhanced_dir, task_id)
            
            self.progress_tracker.complete_task(task_id, True, f"Enhanced {len(enhanced_frames)} frames")
            
            self.logger.log_operation("frame_enhancement_completed", {
                "enhanced_count": len(enhanced_frames),
                "success": True
            })
            
            return enhanced_frames
            
        except Exception as e:
            error_msg = f"Frame enhancement failed: {str(e)}"
            self.logger.log_error(e, {"frame_count": len(frame_paths) if frame_paths else 0})
            if 'task_id' in locals():
                self.progress_tracker.complete_task(task_id, False, error_msg)
            return []
    
    def reassemble_video(self, enhanced_frames: List[str], output_path: str) -> str:
        """Reassemble enhanced frames into video.
        
        Args:
            enhanced_frames: List of paths to enhanced frame files
            output_path: Path for output video file
            
        Returns:
            Path to final enhanced video file
        """
        try:
            if not enhanced_frames:
                raise ValueError("No enhanced frames provided for reassembly")
            
            self.logger.log_operation("video_reassembly_started", {
                "frame_count": len(enhanced_frames),
                "output_path": output_path
            })
            
            # Start progress tracking
            task_id = "video_reassembly"
            self.progress_tracker.start_task(task_id, 100, "Reassembling enhanced video")
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Update progress
            self.progress_tracker.update_progress(task_id, 25, "Preparing frame sequence")
            
            # Create frame list file for FFmpeg
            frame_dir = os.path.dirname(enhanced_frames[0])
            frame_list_file = os.path.join(frame_dir, "frame_list.txt")
            
            with open(frame_list_file, 'w') as f:
                for frame_path in sorted(enhanced_frames):
                    f.write(f"file '{frame_path}'\n")
            
            # Update progress
            self.progress_tracker.update_progress(task_id, 50, "Running FFmpeg reassembly")
            
            # Construct FFmpeg command for video reassembly
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', frame_list_file,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '18',  # High quality
                '-preset', 'medium',
                '-y',  # Overwrite existing files
                output_path
            ]
            
            self.logger.log_operation("ffmpeg_reassembly_command", {"command": " ".join(cmd)})
            
            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Update progress
            self.progress_tracker.update_progress(task_id, 90, "Verifying output video")
            
            # Verify output file exists and has reasonable size
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                # Clean up temporary files
                try:
                    os.remove(frame_list_file)
                except:
                    pass
                
                self.progress_tracker.update_progress(task_id, 100, "Video reassembly completed")
                self.progress_tracker.complete_task(task_id, True, f"Successfully created {output_path}")
                
                self.logger.log_operation("video_reassembly_completed", {
                    "output_path": output_path,
                    "file_size": os.path.getsize(output_path),
                    "success": True
                })
                
                return output_path
            else:
                raise Exception("Output video file was not created or is empty")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg video reassembly failed: {e.stderr}"
            self.logger.log_error(Exception(error_msg), {"output_path": output_path})
            self.progress_tracker.complete_task(task_id, False, error_msg)
            return ""
        except Exception as e:
            error_msg = f"Video reassembly failed: {str(e)}"
            self.logger.log_error(e, {"output_path": output_path})
            if 'task_id' in locals():
                self.progress_tracker.complete_task(task_id, False, error_msg)
            return ""
    
    def extract_audio(self, video_path: str, output_path: str) -> str:
        """Extract audio track from video file.
        
        Args:
            video_path: Path to input video file
            output_path: Path for extracted audio file
            
        Returns:
            Path to extracted audio file, empty string if failed
        """
        try:
            self.logger.log_operation("audio_extraction_started", {
                "video_path": video_path,
                "output_path": output_path
            })
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Start progress tracking
            task_id = "audio_extraction"
            self.progress_tracker.start_task(task_id, 100, "Extracting audio track")
            
            # Update progress
            self.progress_tracker.update_progress(task_id, 25, "Running FFmpeg audio extraction")
            
            # Construct FFmpeg command for audio extraction
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'copy',  # Copy audio codec
                '-y',  # Overwrite existing files
                output_path
            ]
            
            self.logger.log_operation("ffmpeg_audio_extraction_command", {"command": " ".join(cmd)})
            
            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Update progress
            self.progress_tracker.update_progress(task_id, 75, "Verifying extracted audio")
            
            # Verify output file exists and has reasonable size
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                self.progress_tracker.update_progress(task_id, 100, "Audio extraction completed")
                self.progress_tracker.complete_task(task_id, True, f"Successfully extracted audio to {output_path}")
                
                self.logger.log_operation("audio_extraction_completed", {
                    "output_path": output_path,
                    "file_size": os.path.getsize(output_path),
                    "success": True
                })
                
                return output_path
            else:
                raise Exception("Audio file was not created or is empty")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg audio extraction failed: {e.stderr}"
            self.logger.log_error(Exception(error_msg), {"video_path": video_path})
            self.progress_tracker.complete_task(task_id, False, error_msg)
            return ""
        except Exception as e:
            error_msg = f"Audio extraction failed: {str(e)}"
            self.logger.log_error(e, {"video_path": video_path})
            if 'task_id' in locals():
                self.progress_tracker.complete_task(task_id, False, error_msg)
            return ""
    
    def merge_audio_video(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Merge audio and video files with synchronization preservation.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_path: Path for merged output file
            
        Returns:
            Path to merged video file, empty string if failed
        """
        try:
            self.logger.log_operation("audio_video_merge_started", {
                "video_path": video_path,
                "audio_path": audio_path,
                "output_path": output_path
            })
            
            # Start progress tracking
            task_id = "audio_video_merge"
            self.progress_tracker.start_task(task_id, 100, "Merging audio and video")
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Update progress
            self.progress_tracker.update_progress(task_id, 25, "Analyzing audio-video sync")
            
            # Get video and audio information for sync validation
            video_info = self._get_media_info(video_path)
            audio_info = self._get_media_info(audio_path)
            
            # Update progress
            self.progress_tracker.update_progress(task_id, 50, "Running FFmpeg merge")
            
            # Construct FFmpeg command for merging with sync preservation
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',  # Copy video stream
                '-c:a', 'aac',   # Re-encode audio to AAC
                '-b:a', '192k',  # Audio bitrate
                '-ar', '48000',  # Audio sample rate
                '-ac', '2',      # Stereo audio
                '-map', '0:v:0', # Map first video stream
                '-map', '1:a:0', # Map first audio stream
                '-shortest',     # End when shortest stream ends
                '-avoid_negative_ts', 'make_zero',  # Handle timestamp issues
                '-y',            # Overwrite existing files
                output_path
            ]
            
            self.logger.log_operation("ffmpeg_merge_command", {"command": " ".join(cmd)})
            
            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Update progress
            self.progress_tracker.update_progress(task_id, 90, "Validating merged output")
            
            # Validate the merged output
            if self._validate_merged_output(output_path, video_info, audio_info):
                self.progress_tracker.update_progress(task_id, 100, "Audio-video merge completed")
                self.progress_tracker.complete_task(task_id, True, f"Successfully merged to {output_path}")
                
                self.logger.log_operation("audio_video_merge_completed", {
                    "output_path": output_path,
                    "file_size": os.path.getsize(output_path),
                    "success": True
                })
                
                return output_path
            else:
                raise Exception("Merged output validation failed")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg audio-video merge failed: {e.stderr}"
            self.logger.log_error(Exception(error_msg), {"output_path": output_path})
            self.progress_tracker.complete_task(task_id, False, error_msg)
            return ""
        except Exception as e:
            error_msg = f"Audio-video merge failed: {str(e)}"
            self.logger.log_error(e, {"output_path": output_path})
            if 'task_id' in locals():
                self.progress_tracker.complete_task(task_id, False, error_msg)
            return ""
    
    def validate_processing_stage(self, stage_name: str, input_files: List[str], output_files: List[str]) -> Dict[str, Any]:
        """Validate intermediate processing stage results.
        
        Args:
            stage_name: Name of the processing stage
            input_files: List of input file paths
            output_files: List of output file paths
            
        Returns:
            Dictionary with validation results
        """
        try:
            self.logger.log_operation("stage_validation_started", {
                "stage_name": stage_name,
                "input_count": len(input_files),
                "output_count": len(output_files)
            })
            
            validation_results = {
                "stage_name": stage_name,
                "success": True,
                "input_count": len(input_files),
                "output_count": len(output_files),
                "missing_files": [],
                "corrupted_files": [],
                "size_issues": [],
                "validation_errors": []
            }
            
            # Check if all expected output files exist
            for output_file in output_files:
                if not os.path.exists(output_file):
                    validation_results["missing_files"].append(output_file)
                    validation_results["success"] = False
                elif os.path.getsize(output_file) == 0:
                    validation_results["size_issues"].append(output_file)
                    validation_results["success"] = False
            
            # For image files, validate they can be loaded
            if stage_name in ["frame_extraction", "frame_enhancement"]:
                for output_file in output_files:
                    if os.path.exists(output_file):
                        try:
                            img = cv2.imread(output_file)
                            if img is None:
                                validation_results["corrupted_files"].append(output_file)
                                validation_results["success"] = False
                        except Exception as e:
                            validation_results["corrupted_files"].append(output_file)
                            validation_results["success"] = False
            
            # For video files, validate basic properties
            elif stage_name in ["video_reassembly", "audio_video_merge"]:
                for output_file in output_files:
                    if os.path.exists(output_file):
                        try:
                            media_info = self._get_media_info(output_file)
                            if not media_info or media_info.get("duration", 0) <= 0:
                                validation_results["corrupted_files"].append(output_file)
                                validation_results["success"] = False
                        except Exception as e:
                            validation_results["corrupted_files"].append(output_file)
                            validation_results["success"] = False
            
            self.logger.log_operation("stage_validation_completed", validation_results)
            
            return validation_results
            
        except Exception as e:
            error_msg = f"Stage validation failed: {str(e)}"
            self.logger.log_error(e, {"stage_name": stage_name})
            return {
                "stage_name": stage_name,
                "success": False,
                "validation_errors": [error_msg]
            }
    
    def validate_final_output(self, output_path: str, original_video_path: str) -> Dict[str, Any]:
        """Validate final enhanced video output quality and properties.
        
        Args:
            output_path: Path to final enhanced video
            original_video_path: Path to original video for comparison
            
        Returns:
            Dictionary with validation results
        """
        try:
            self.logger.log_operation("final_validation_started", {
                "output_path": output_path,
                "original_path": original_video_path
            })
            
            validation_results = {
                "success": True,
                "file_exists": False,
                "file_size": 0,
                "duration_match": False,
                "resolution_enhanced": False,
                "audio_present": False,
                "sync_preserved": False,
                "quality_metrics": {},
                "validation_errors": []
            }
            
            # Check if output file exists
            if not os.path.exists(output_path):
                validation_results["validation_errors"].append("Output file does not exist")
                validation_results["success"] = False
                return validation_results
            
            validation_results["file_exists"] = True
            validation_results["file_size"] = os.path.getsize(output_path)
            
            # Get media information
            output_info = self._get_media_info(output_path)
            original_info = self._get_media_info(original_video_path)
            
            if not output_info or not original_info:
                validation_results["validation_errors"].append("Failed to analyze media files")
                validation_results["success"] = False
                return validation_results
            
            # Check duration match (within 1% tolerance)
            original_duration = original_info.get("duration", 0)
            output_duration = output_info.get("duration", 0)
            
            if original_duration > 0 and output_duration > 0:
                duration_diff = abs(original_duration - output_duration) / original_duration
                validation_results["duration_match"] = duration_diff < 0.01
                if not validation_results["duration_match"]:
                    validation_results["validation_errors"].append(
                        f"Duration mismatch: original={original_duration:.2f}s, output={output_duration:.2f}s"
                    )
            
            # Check resolution enhancement
            original_width = original_info.get("width", 0)
            original_height = original_info.get("height", 0)
            output_width = output_info.get("width", 0)
            output_height = output_info.get("height", 0)
            
            if output_width > original_width or output_height > original_height:
                validation_results["resolution_enhanced"] = True
            
            # Check audio presence
            validation_results["audio_present"] = output_info.get("has_audio", False)
            
            # Basic sync check (audio and video streams have similar duration)
            if validation_results["audio_present"]:
                video_duration = output_info.get("video_duration", 0)
                audio_duration = output_info.get("audio_duration", 0)
                if video_duration > 0 and audio_duration > 0:
                    sync_diff = abs(video_duration - audio_duration) / max(video_duration, audio_duration)
                    validation_results["sync_preserved"] = sync_diff < 0.02  # 2% tolerance
            
            # Store quality metrics
            validation_results["quality_metrics"] = {
                "original_resolution": f"{original_width}x{original_height}",
                "output_resolution": f"{output_width}x{output_height}",
                "original_duration": original_duration,
                "output_duration": output_duration,
                "file_size_mb": validation_results["file_size"] / (1024 * 1024)
            }
            
            # Overall success check
            if not validation_results["duration_match"] or not validation_results["audio_present"]:
                validation_results["success"] = False
            
            self.logger.log_operation("final_validation_completed", validation_results)
            
            return validation_results
            
        except Exception as e:
            error_msg = f"Final validation failed: {str(e)}"
            self.logger.log_error(e, {"output_path": output_path})
            return {
                "success": False,
                "validation_errors": [error_msg]
            }
    
    def _get_media_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get media file information using FFprobe.
        
        Args:
            file_path: Path to media file
            
        Returns:
            Dictionary with media information or None if failed
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)
            
            # Extract relevant information
            info = {
                "duration": float(probe_data.get("format", {}).get("duration", 0)),
                "size": int(probe_data.get("format", {}).get("size", 0)),
                "has_video": False,
                "has_audio": False,
                "width": 0,
                "height": 0,
                "video_duration": 0,
                "audio_duration": 0
            }
            
            # Process streams
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "video":
                    info["has_video"] = True
                    info["width"] = int(stream.get("width", 0))
                    info["height"] = int(stream.get("height", 0))
                    info["video_duration"] = float(stream.get("duration", 0))
                elif stream.get("codec_type") == "audio":
                    info["has_audio"] = True
                    info["audio_duration"] = float(stream.get("duration", 0))
            
            return info
            
        except Exception as e:
            self.logger.log_error(e, {"file_path": file_path})
            return None
    
    def _validate_merged_output(self, output_path: str, video_info: Dict[str, Any], audio_info: Dict[str, Any]) -> bool:
        """Validate merged audio-video output.
        
        Args:
            output_path: Path to merged output file
            video_info: Original video information
            audio_info: Original audio information
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            output_info = self._get_media_info(output_path)
            if not output_info:
                return False
            
            # Check that output has both video and audio
            if not output_info["has_video"] or not output_info["has_audio"]:
                return False
            
            # Check duration is reasonable (within 5% of original video)
            original_duration = video_info.get("duration", 0)
            output_duration = output_info.get("duration", 0)
            
            if original_duration > 0 and output_duration > 0:
                duration_diff = abs(original_duration - output_duration) / original_duration
                if duration_diff > 0.05:  # 5% tolerance
                    return False
            
            # Check file size is reasonable (not empty, not too small)
            if output_info["size"] < 1024:  # Less than 1KB
                return False
            
            return True
            
        except Exception as e:
            self.logger.log_error(e, {"output_path": output_path})
            return False
    
    def _is_real_esrgan_available(self) -> bool:
        """Check if Real-ESRGAN is available."""
        try:
            result = subprocess.run(['realesrgan-ncnn-vulkan', '--help'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _enhance_with_real_esrgan(self, frame_paths: List[str], output_dir: str, task_id: str) -> List[str]:
        """Enhance frames using Real-ESRGAN."""
        enhanced_frames = []
        
        for i, frame_path in enumerate(frame_paths):
            try:
                # Update progress
                self.progress_tracker.update_progress(
                    task_id, i, f"Processing frame {i+1}/{len(frame_paths)} with Real-ESRGAN"
                )
                
                # Generate output path
                frame_name = os.path.basename(frame_path)
                output_path = os.path.join(output_dir, frame_name)
                
                # Run Real-ESRGAN
                cmd = [
                    'realesrgan-ncnn-vulkan',
                    '-i', frame_path,
                    '-o', output_path,
                    '-n', self.config.ai_model,
                    '-s', str(self.config.tile_size)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                if os.path.exists(output_path):
                    enhanced_frames.append(output_path)
                
            except Exception as e:
                self.logger.log_error(e, {"frame_path": frame_path})
                # Continue with next frame
                continue
        
        return enhanced_frames
    
    def _enhance_with_opencv(self, frame_paths: List[str], output_dir: str, task_id: str) -> List[str]:
        """Enhance frames using OpenCV as fallback."""
        enhanced_frames = []
        
        for i, frame_path in enumerate(frame_paths):
            try:
                # Update progress
                self.progress_tracker.update_progress(
                    task_id, i, f"Processing frame {i+1}/{len(frame_paths)} with OpenCV"
                )
                
                # Load image
                img = cv2.imread(frame_path)
                if img is None:
                    continue
                
                # Apply basic enhancement (upscaling + sharpening)
                height, width = img.shape[:2]
                new_width = int(width * 2)  # 2x upscaling
                new_height = int(height * 2)
                
                # Upscale using INTER_CUBIC
                upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                # Apply sharpening
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                enhanced = cv2.filter2D(upscaled, -1, kernel)
                
                # Save enhanced frame
                frame_name = os.path.basename(frame_path)
                output_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(output_path, enhanced)
                
                enhanced_frames.append(output_path)
                
            except Exception as e:
                self.logger.log_error(e, {"frame_path": frame_path})
                continue
        
        return enhanced_frames