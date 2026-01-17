"""
FFmpeg Media Controller - Professional media pipeline orchestration.

This module replaces OpenCV video writing with FFmpeg-based media control,
solving audio, codec compatibility, and file size issues at the architecture level.
"""

import subprocess
import logging
import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Generator
from queue import Queue, Empty
import numpy as np

from ..models.data_models import VideoInfo, AudioConfig
from ..utils.system_utils import get_system_info


class FFmpegMediaController:
    """
    FFmpeg-based media pipeline controller.
    
    Responsibilities:
    - Launch and manage FFmpeg processes
    - Handle stdin/stdout pipes for raw frame streaming
    - Control audio copy/enhancement modes
    - Manage final encoding formats (H.264/H.265 + AAC)
    - Orchestrate video+audio muxing
    
    NOT responsible for:
    - AI enhancement (handled by core processors)
    - Frame-level processing (handled by streaming processor)
    - Business logic (handled by strategy layer)
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.temp_dir = Path(temp_dir) if temp_dir else Path("./temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Process management
        self.decoder_process: Optional[subprocess.Popen] = None
        self.encoder_process: Optional[subprocess.Popen] = None
        self.audio_process: Optional[subprocess.Popen] = None
        
        # Pipeline state
        self.input_video_info: Optional[VideoInfo] = None
        self.output_resolution: Optional[Tuple[int, int]] = None
        self.frame_rate: Optional[float] = None
        
        # Threading for pipe management
        self._stop_event = threading.Event()
        self._error_queue = Queue()
    
    def analyze_input_video(self, input_path: str) -> VideoInfo:
        """
        Analyze input video using FFprobe.
        
        Args:
            input_path: Path to input video file
            
        Returns:
            VideoInfo object with complete video characteristics
        """
        try:
            # Use FFprobe to get detailed video information
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", input_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            probe_data = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            audio_stream = None
            
            for stream in probe_data["streams"]:
                if stream["codec_type"] == "video" and video_stream is None:
                    video_stream = stream
                elif stream["codec_type"] == "audio" and audio_stream is None:
                    audio_stream = stream
            
            if not video_stream:
                raise ValueError("No video stream found in input file")
            
            # Parse video information
            width = int(video_stream["width"])
            height = int(video_stream["height"])
            
            # Calculate frame rate
            fps_str = video_stream.get("r_frame_rate", "25/1")
            if "/" in fps_str:
                num, den = map(int, fps_str.split("/"))
                framerate = num / den if den != 0 else 25.0
            else:
                framerate = float(fps_str)
            
            # Calculate duration
            duration = float(probe_data["format"].get("duration", 0))
            
            # Audio information
            has_audio = audio_stream is not None
            audio_channels = int(audio_stream.get("channels", 0)) if has_audio else 0
            audio_sample_rate = int(audio_stream.get("sample_rate", 0)) if has_audio else 0
            
            # File information
            file_size = int(probe_data["format"].get("size", 0))
            format_name = probe_data["format"].get("format_name", "unknown")
            codec = video_stream.get("codec_name", "unknown")
            bitrate = int(probe_data["format"].get("bit_rate", 0))
            
            video_info = VideoInfo(
                resolution=(width, height),
                duration=duration,
                framerate=framerate,
                codec=codec,
                bitrate=bitrate,
                has_audio=has_audio,
                audio_channels=audio_channels,
                audio_sample_rate=audio_sample_rate,
                file_size=file_size,
                format=format_name
            )
            
            self.input_video_info = video_info
            self.frame_rate = framerate
            
            self.logger.info(f"Analyzed input video: {width}x{height} @ {framerate}fps, "
                           f"duration: {duration:.1f}s, audio: {has_audio}")
            
            return video_info
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFprobe failed: {e}")
            raise RuntimeError(f"Failed to analyze input video: {e}")
        except Exception as e:
            self.logger.error(f"Video analysis error: {e}")
            raise
    
    def start_decoder_pipeline(self, input_path: str, target_resolution: Tuple[int, int]) -> subprocess.Popen:
        """
        Start FFmpeg decoder process for raw frame extraction.
        
        Args:
            input_path: Input video file path
            target_resolution: Target resolution for processing (width, height)
            
        Returns:
            FFmpeg decoder process with stdout pipe
        """
        self.output_resolution = target_resolution
        width, height = target_resolution
        
        # FFmpeg command for raw frame extraction
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-map", "0:v:0",  # Map only video stream
            "-vf", f"scale={width}:{height}:flags=lanczos",  # Scale to target resolution
            "-f", "rawvideo",  # Raw video format
            "-pix_fmt", "rgb24",  # RGB24 pixel format for Python processing
            "-vsync", "0",  # Disable frame rate conversion
            "-y",  # Overwrite output
            "pipe:1"  # Output to stdout
        ]
        
        try:
            self.decoder_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0  # Unbuffered for real-time processing
            )
            
            self.logger.info(f"Started decoder pipeline: {width}x{height} RGB24 frames")
            return self.decoder_process
            
        except Exception as e:
            self.logger.error(f"Failed to start decoder pipeline: {e}")
            raise RuntimeError(f"Decoder pipeline failed: {e}")
    
    def start_encoder_pipeline(self, output_path: str, input_path: str, 
                             audio_mode: str = "copy", 
                             video_codec: str = "libx264",
                             audio_config: Optional[AudioConfig] = None) -> subprocess.Popen:
        """
        Start FFmpeg encoder process for final video creation.
        
        Args:
            output_path: Output video file path
            input_path: Original input path (for audio extraction)
            audio_mode: "copy" or "theater_enhanced"
            video_codec: Video codec (libx264, libx265)
            audio_config: Audio configuration for enhancement
            
        Returns:
            FFmpeg encoder process with stdin pipe
        """
        if not self.output_resolution or not self.frame_rate:
            raise RuntimeError("Must call start_decoder_pipeline first")
        
        width, height = self.output_resolution
        
        # Base encoder command
        cmd = [
            "ffmpeg",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(self.frame_rate),
            "-i", "pipe:0",  # Enhanced video frames from stdin
        ]
        
        # Add audio input
        if self.input_video_info and self.input_video_info.has_audio:
            cmd.extend(["-i", input_path])  # Original video for audio
        
        # Video encoding settings
        cmd.extend([
            "-map", "0:v:0",  # Map enhanced video from pipe
            "-c:v", video_codec,
            "-pix_fmt", "yuv420p",  # Standard pixel format for compatibility
            "-preset", "medium",  # Encoding speed vs quality balance
            "-crf", "18",  # High quality (lower = better quality)
        ])
        
        # Audio handling
        if self.input_video_info and self.input_video_info.has_audio:
            if audio_mode == "copy":
                cmd.extend([
                    "-map", "1:a:0",  # Map original audio
                    "-c:a", "copy"  # Copy audio without re-encoding
                ])
            elif audio_mode == "theater_enhanced":
                # Theater audio enhancement filters
                audio_filters = self._build_theater_audio_filters(audio_config)
                cmd.extend([
                    "-map", "1:a:0",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-af", audio_filters
                ])
        
        # Output settings
        cmd.extend([
            "-movflags", "+faststart",  # Optimize for streaming
            "-y",  # Overwrite output
            output_path
        ])
        
        try:
            self.encoder_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            self.logger.info(f"Started encoder pipeline: {video_codec} + {audio_mode} audio")
            return self.encoder_process
            
        except Exception as e:
            self.logger.error(f"Failed to start encoder pipeline: {e}")
            raise RuntimeError(f"Encoder pipeline failed: {e}")
    
    def _build_theater_audio_filters(self, audio_config: Optional[AudioConfig]) -> str:
        """
        Build FFmpeg audio filter chain for theater enhancement.
        
        Args:
            audio_config: Audio configuration settings
            
        Returns:
            FFmpeg audio filter string
        """
        if not audio_config:
            # Default theater enhancement
            return "loudnorm=I=-16:TP=-1.5:LRA=11,highpass=f=80,lowpass=f=15000"
        
        filters = []
        
        # Noise reduction (using afftdn filter)
        if hasattr(audio_config, 'theater_presets'):
            preset = audio_config.theater_presets.get('medium', {})
            noise_reduction = preset.get('noise_reduction', 0.5)
            if noise_reduction > 0:
                filters.append(f"afftdn=nr={noise_reduction * 20}")
        
        # Dialogue enhancement (EQ boost in speech frequencies)
        dialogue_boost = getattr(audio_config, 'dialogue_enhancement', 0.7)
        if dialogue_boost > 0:
            # Boost 1-4kHz range for dialogue clarity
            boost_db = dialogue_boost * 6  # Max 6dB boost
            filters.append(f"equalizer=f=2000:width_type=o:width=2:g={boost_db}")
        
        # Dynamic range compression for theater
        filters.append("acompressor=threshold=-18dB:ratio=3:attack=5:release=50")
        
        # Loudness normalization
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
        
        return ",".join(filters) if filters else "loudnorm=I=-16:TP=-1.5:LRA=11"
    
    def read_raw_frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields raw frames from decoder pipeline.
        
        Yields:
            numpy arrays representing RGB24 frames
        """
        if not self.decoder_process or not self.output_resolution:
            raise RuntimeError("Decoder pipeline not started")
        
        width, height = self.output_resolution
        frame_size = width * height * 3  # RGB24 = 3 bytes per pixel
        
        try:
            while True:
                # Read one frame worth of data
                frame_data = self.decoder_process.stdout.read(frame_size)
                
                if len(frame_data) != frame_size:
                    # End of stream or error
                    break
                
                # Convert to numpy array
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((height, width, 3))
                
                yield frame
                
        except Exception as e:
            self.logger.error(f"Error reading raw frames: {e}")
            raise
        finally:
            self._cleanup_decoder()
    
    def write_raw_frame(self, frame: np.ndarray) -> bool:
        """
        Write enhanced frame to encoder pipeline.
        
        Args:
            frame: Enhanced frame as numpy array (RGB24)
            
        Returns:
            True if successful, False if pipeline closed
        """
        if not self.encoder_process:
            raise RuntimeError("Encoder pipeline not started")
        
        try:
            # Ensure frame is in correct format
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Write frame data to encoder stdin
            frame_bytes = frame.tobytes()
            self.encoder_process.stdin.write(frame_bytes)
            self.encoder_process.stdin.flush()
            
            return True
            
        except BrokenPipeError:
            self.logger.warning("Encoder pipeline closed")
            return False
        except Exception as e:
            self.logger.error(f"Error writing frame: {e}")
            return False
    
    def finalize_encoding(self) -> bool:
        """
        Finalize encoding process and wait for completion.
        
        Returns:
            True if encoding completed successfully
        """
        success = True
        
        try:
            # Close encoder stdin to signal end of input
            if self.encoder_process and self.encoder_process.stdin:
                self.encoder_process.stdin.close()
            
            # Wait for encoder to finish
            if self.encoder_process:
                return_code = self.encoder_process.wait(timeout=300)  # 5 minute timeout
                if return_code != 0:
                    stderr_output = self.encoder_process.stderr.read().decode('utf-8')
                    self.logger.error(f"Encoder failed with code {return_code}: {stderr_output}")
                    success = False
                else:
                    self.logger.info("Encoding completed successfully")
            
        except subprocess.TimeoutExpired:
            self.logger.error("Encoder timeout - killing process")
            if self.encoder_process:
                self.encoder_process.kill()
            success = False
        except Exception as e:
            self.logger.error(f"Error finalizing encoding: {e}")
            success = False
        finally:
            self._cleanup_encoder()
        
        return success
    
    def _cleanup_decoder(self):
        """Clean up decoder process."""
        if self.decoder_process:
            try:
                if self.decoder_process.poll() is None:
                    self.decoder_process.terminate()
                    self.decoder_process.wait(timeout=10)
            except:
                pass
            finally:
                self.decoder_process = None
    
    def _cleanup_encoder(self):
        """Clean up encoder process."""
        if self.encoder_process:
            try:
                if self.encoder_process.poll() is None:
                    self.encoder_process.terminate()
                    self.encoder_process.wait(timeout=10)
            except:
                pass
            finally:
                self.encoder_process = None
    
    def cleanup(self):
        """Clean up all processes and resources."""
        self._stop_event.set()
        self._cleanup_decoder()
        self._cleanup_encoder()
        
        # Clean up temporary files
        try:
            for temp_file in self.temp_dir.glob("ffmpeg_*"):
                temp_file.unlink()
        except Exception as e:
            self.logger.warning(f"Failed to clean temp files: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class MediaPipelineError(Exception):
    """Exception raised for media pipeline errors."""
    pass


def check_ffmpeg_availability() -> Dict[str, bool]:
    """
    Check FFmpeg and FFprobe availability.
    
    Returns:
        Dictionary with availability status
    """
    availability = {
        "ffmpeg": False,
        "ffprobe": False,
        "codecs": {}
    }
    
    try:
        # Check FFmpeg
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True, timeout=10)
        availability["ffmpeg"] = result.returncode == 0
        
        # Check FFprobe
        result = subprocess.run(["ffprobe", "-version"], 
                              capture_output=True, text=True, timeout=10)
        availability["ffprobe"] = result.returncode == 0
        
        # Check codec availability
        if availability["ffmpeg"]:
            result = subprocess.run(["ffmpeg", "-codecs"], 
                                  capture_output=True, text=True, timeout=10)
            codec_output = result.stdout
            
            availability["codecs"] = {
                "libx264": "libx264" in codec_output,
                "libx265": "libx265" in codec_output,
                "aac": " aac " in codec_output,
                "mp3": " mp3 " in codec_output
            }
    
    except Exception:
        pass
    
    return availability