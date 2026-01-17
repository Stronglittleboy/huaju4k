#!/usr/bin/env python3
"""
Theater Video 4K Enhancement - Automated Processing Script
Complete implementation for processing theater drama videos to 4K resolution

Usage:
    python automated_theater_video_enhancer.py input_video.mp4 [config.json]
"""

import os
import sys
import json
import subprocess
import logging
import shutil
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from PIL import Image

class TheaterVideoEnhancer:
    def __init__(self, config_file="processing_config.json"):
        self.config = self.load_config(config_file) if Path(config_file).exists() else self.default_config()
        self.setup_logging()
        self.workspace = Path("workspace")
        self.workspace.mkdir(exist_ok=True)
        
    def default_config(self):
        """Default processing configuration"""
        return {
            "project_name": "theater_video_enhancement",
            "processing_config": {
                "recommended_tool": "real-esrgan",
                "ai_model": "RealESRGAN_x4plus",
                "backup_model": "RealESRNet_x4plus",
                "scale_factor": 2,
                "tile_size": 640,
                "batch_size": 1,
                "denoise_level": 2,
                "gpu_acceleration": True,
                "memory_optimization": True,
                "processing_environment": "wsl"
            },
            "quality_settings": {
                "frame_extraction_format": "png",
                "intermediate_cleanup": True,
                "quality_check_enabled": True,
                "temporal_consistency": True
            }
        }
    
    def load_config(self, config_file):
        """Load processing configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self.default_config()
    
    def setup_logging(self):
        """Configure logging for the enhancement process"""
        log_file = f"enhancement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_command(self, command, check=True):
        """Execute shell command with logging"""
        self.logger.info(f"Executing: {command}")
        try:
            result = subprocess.run(command, shell=True, check=check, 
                                  capture_output=True, text=True)
            if result.stdout:
                self.logger.info(f"Output: {result.stdout.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e}")
            if e.stderr:
                self.logger.error(f"Error: {e.stderr}")
            raise
    
    def analyze_system(self):
        """Analyze system capabilities and requirements"""
        self.logger.info("Analyzing system specifications...")
        
        # Check GPU
        try:
            result = self.run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits")
            gpu_info = result.stdout.strip().split(', ')
            self.logger.info(f"GPU: {gpu_info[0]}, VRAM: {gpu_info[1]} MB")
        except:
            self.logger.warning("NVIDIA GPU not detected or nvidia-smi not available")
        
        # Check available space
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (1024**3)
        self.logger.info(f"Available storage: {free_gb} GB")
        
        if free_gb < 10:
            raise RuntimeError("Insufficient storage space (minimum 10GB required)")
    
    def analyze_video(self, input_video):
        """Analyze input video characteristics"""
        self.logger.info(f"Analyzing video: {input_video}")
        
        if not Path(input_video).exists():
            raise FileNotFoundError(f"Input video not found: {input_video}")
        
        # Get video information using ffprobe
        cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams "{input_video}"'
        result = self.run_command(cmd)
        video_info = json.loads(result.stdout)
        
        # Extract video stream info
        video_stream = next(s for s in video_info['streams'] if s['codec_type'] == 'video')
        
        self.video_info = {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'fps': eval(video_stream['r_frame_rate']),
            'duration': float(video_info['format']['duration']),
            'codec': video_stream['codec_name']
        }
        
        self.logger.info(f"Resolution: {self.video_info['width']}x{self.video_info['height']}")
        self.logger.info(f"FPS: {self.video_info['fps']}")
        self.logger.info(f"Duration: {self.video_info['duration']:.2f} seconds")
        
        return self.video_info
    
    def setup_environment(self):
        """Set up processing environment and directories"""
        self.logger.info("Setting up processing environment...")
        
        # Create workspace directories
        dirs = ['frames', 'enhanced', 'temp', 'backup']
        for dir_name in dirs:
            (self.workspace / dir_name).mkdir(exist_ok=True)
        
        # Check for required tools
        tools = ['ffmpeg', 'python3']
        for tool in tools:
            try:
                self.run_command(f"which {tool}")
                self.logger.info(f"✅ {tool} available")
            except:
                raise RuntimeError(f"Required tool not found: {tool}")
    
    def extract_frames(self, input_video):
        """Extract video frames using FFmpeg"""
        self.logger.info("Extracting video frames...")
        
        frames_dir = self.workspace / "frames"
        audio_file = self.workspace / "audio.aac"
        
        # Extract frames
        fps = self.video_info['fps']
        frame_cmd = f'ffmpeg -i "{input_video}" -vf fps={fps} "{frames_dir}/frame_%08d.png" -y'
        self.run_command(frame_cmd)
        
        # Extract audio
        audio_cmd = f'ffmpeg -i "{input_video}" -vn -acodec copy "{audio_file}" -y'
        self.run_command(audio_cmd)
        
        # Count extracted frames
        frame_count = len(list(frames_dir.glob("*.png")))
        self.logger.info(f"Extracted {frame_count} frames and audio track")
        
        return frame_count
    
    def enhance_frames_realesrgan(self):
        """Enhance frames using Real-ESRGAN"""
        self.logger.info("Attempting Real-ESRGAN enhancement...")
        
        frames_dir = self.workspace / "frames"
        enhanced_dir = self.workspace / "enhanced"
        
        config = self.config['processing_config']
        scale = config['scale_factor']
        tile_size = config['tile_size']
        model = config['ai_model']
        
        try:
            cmd = f"realesrgan-ncnn-vulkan -i {frames_dir} -o {enhanced_dir} -n {model} -s {scale} -t {tile_size}"
            self.run_command(cmd)
            
            # Verify output
            enhanced_count = len(list(enhanced_dir.glob("*.png")))
            if enhanced_count > 0:
                self.logger.info(f"✅ Real-ESRGAN processed {enhanced_count} frames")
                return True
            else:
                self.logger.warning("Real-ESRGAN produced no output")
                return False
                
        except Exception as e:
            self.logger.warning(f"Real-ESRGAN failed: {e}")
            return False
    
    def enhance_frames_opencv(self):
        """Fallback enhancement using OpenCV"""
        self.logger.info("Using OpenCV fallback enhancement...")
        
        frames_dir = self.workspace / "frames"
        enhanced_dir = self.workspace / "enhanced"
        
        scale_factor = self.config['processing_config']['scale_factor']
        frame_files = sorted(frames_dir.glob("*.png"))
        
        processed = 0
        for frame_file in frame_files:
            try:
                # Load image
                img = cv2.imread(str(frame_file))
                if img is None:
                    continue
                
                # Upscale using INTER_CUBIC
                height, width = img.shape[:2]
                new_width = width * scale_factor
                new_height = height * scale_factor
                
                enhanced = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                # Save enhanced frame
                output_file = enhanced_dir / frame_file.name
                cv2.imwrite(str(output_file), enhanced)
                processed += 1
                
                if processed % 10 == 0:
                    progress = (processed / len(frame_files)) * 100
                    self.logger.info(f"OpenCV Progress: {progress:.1f}% ({processed}/{len(frame_files)})")
                    
            except Exception as e:
                self.logger.error(f"Failed to process {frame_file}: {e}")
        
        self.logger.info(f"✅ OpenCV processed {processed} frames")
        return processed > 0
    
    def enhance_frames(self):
        """Enhance frames using AI upscaling with fallback"""
        self.logger.info("Starting frame enhancement process...")
        
        # Try Real-ESRGAN first
        if self.enhance_frames_realesrgan():
            self.enhancement_method = "real-esrgan"
            return True
        
        # Fallback to OpenCV
        if self.enhance_frames_opencv():
            self.enhancement_method = "opencv"
            return True
        
        raise RuntimeError("All enhancement methods failed")
    
    def apply_temporal_consistency(self):
        """Apply temporal consistency filtering to prevent flickering"""
        if not self.config['quality_settings']['temporal_consistency']:
            return
        
        self.logger.info("Applying temporal consistency filtering...")
        
        enhanced_dir = self.workspace / "enhanced"
        backup_dir = self.workspace / "backup"
        
        # Create backup
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(enhanced_dir, backup_dir)
        
        # Simple temporal filtering (can be enhanced)
        frame_files = sorted(enhanced_dir.glob("*.png"))
        
        for i in range(1, len(frame_files) - 1):
            try:
                # Load three consecutive frames
                prev_img = cv2.imread(str(frame_files[i-1]))
                curr_img = cv2.imread(str(frame_files[i]))
                next_img = cv2.imread(str(frame_files[i+1]))
                
                if prev_img is None or curr_img is None or next_img is None:
                    continue
                
                # Apply simple temporal averaging
                filtered = cv2.addWeighted(
                    cv2.addWeighted(prev_img, 0.25, curr_img, 0.5, 0),
                    1.0, next_img, 0.25, 0
                )
                
                # Save filtered frame
                cv2.imwrite(str(frame_files[i]), filtered)
                
            except Exception as e:
                self.logger.warning(f"Temporal filtering failed for frame {i}: {e}")
        
        self.logger.info("✅ Temporal consistency filtering applied")
    
    def reassemble_video(self, output_file):
        """Reassemble enhanced frames into 4K video"""
        self.logger.info("Reassembling enhanced video...")
        
        enhanced_dir = self.workspace / "enhanced"
        audio_file = self.workspace / "audio.aac"
        
        fps = self.video_info['fps']
        
        # High-quality video encoding settings
        video_cmd = f'''ffmpeg -framerate {fps} -i "{enhanced_dir}/frame_%08d.png" -i "{audio_file}" \
                       -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p \
                       -profile:v high -level:v 5.1 \
                       -c:a aac -b:a 192k -ar 48000 \
                       -movflags +faststart -map 0:v:0 -map 1:a:0 \
                       -shortest -y "{output_file}"'''
        
        self.run_command(video_cmd)
        
        if Path(output_file).exists():
            file_size = Path(output_file).stat().st_size / (1024*1024)  # MB
            self.logger.info(f"✅ Video reassembled: {output_file} ({file_size:.1f} MB)")
            return True
        else:
            raise RuntimeError("Video reassembly failed")
    
    def validate_output(self, output_file):
        """Validate final 4K output quality"""
        self.logger.info("Validating output quality...")
        
        if not Path(output_file).exists():
            raise RuntimeError("Output file not found")
        
        # Check resolution using ffprobe
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "{output_file}"'
        result = self.run_command(cmd)
        width, height = map(int, result.stdout.strip().split(','))
        
        self.logger.info(f"Output resolution: {width}x{height}")
        
        # Validate 4K resolution
        if width == 3840 and height == 2160:
            self.logger.info("✅ 4K resolution confirmed")
            return True
        else:
            self.logger.warning(f"❌ Expected 4K (3840x2160), got {width}x{height}")
            return False
    
    def cleanup_workspace(self):
        """Clean up intermediate files if configured"""
        if not self.config['quality_settings']['intermediate_cleanup']:
            return
        
        self.logger.info("Cleaning up intermediate files...")
        
        # Keep only essential files
        cleanup_dirs = ['frames', 'temp']
        for dir_name in cleanup_dirs:
            dir_path = self.workspace / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)
                self.logger.info(f"Cleaned up {dir_name}")
    
    def generate_report(self, output_file, start_time):
        """Generate processing report"""
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        report = {
            "timestamp": end_time.isoformat(),
            "processing_time_seconds": processing_time,
            "enhancement_method": getattr(self, 'enhancement_method', 'unknown'),
            "configuration": self.config,
            "video_info": self.video_info,
            "output_file": str(output_file),
            "success": True
        }
        
        report_file = f"enhancement_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Processing report saved: {report_file}")
        return report_file
    
    def run_enhancement(self, input_video, output_file=None):
        """Execute complete enhancement workflow"""
        start_time = datetime.now()
        
        if output_file is None:
            output_file = f"enhanced_4k_{Path(input_video).stem}.mp4"
        
        try:
            self.logger.info("🎬 Starting theater video 4K enhancement workflow")
            self.logger.info(f"Input: {input_video}")
            self.logger.info(f"Output: {output_file}")
            
            # Execute workflow phases
            self.analyze_system()
            self.analyze_video(input_video)
            self.setup_environment()
            
            frame_count = self.extract_frames(input_video)
            self.enhance_frames()
            self.apply_temporal_consistency()
            self.reassemble_video(output_file)
            
            # Validate and report
            self.validate_output(output_file)
            self.generate_report(output_file, start_time)
            self.cleanup_workspace()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"🎉 Enhancement completed successfully in {processing_time:.1f} seconds!")
            self.logger.info(f"✅ Output: {output_file}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"❌ Enhancement failed: {str(e)}")
            raise

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python automated_theater_video_enhancer.py input_video.mp4 [config.json]")
        sys.exit(1)
    
    input_video = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else "processing_config.json"
    
    enhancer = TheaterVideoEnhancer(config_file)
    enhancer.run_enhancement(input_video)

if __name__ == "__main__":
    main()