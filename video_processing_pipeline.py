#!/usr/bin/env python3
"""
Video Processing Pipeline Implementation

This script implements the complete video processing pipeline for 4K enhancement,
including frame extraction, preprocessing, AI upscaling, and video reassembly.
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add video_enhancer to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'video_enhancer'))

from video_enhancer.config import ConfigManager
from video_enhancer.models import ProcessingConfig, ProcessingTool, ProcessingEnvironment
from video_enhancer.frame_processor import FrameProcessor
from video_enhancer.ai_upscaler import AIUpscaler
from video_enhancer.video_analyzer import VideoAnalyzer


class VideoProcessingPipeline:
    """Complete video processing pipeline for 4K enhancement."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the processing pipeline."""
        self.config_manager = config_manager
        self.logger = self._setup_logging()
        
        # Initialize components
        self.frame_processor = FrameProcessor(config_manager)
        self.ai_upscaler = AIUpscaler(config_manager)
        self.video_analyzer = VideoAnalyzer()
        
        # Processing state
        self.processing_stats = {}
        self.workspace_dir = "workspace"
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('video_processing.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def process_video_to_4k(self, input_video_path: str, output_video_path: str,
                           processing_config: Optional[ProcessingConfig] = None) -> bool:
        """
        Complete pipeline to process video from input to 4K output.
        
        Args:
            input_video_path: Path to input video file
            output_video_path: Path for final 4K output
            processing_config: Optional processing configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("="*70)
            self.logger.info("STARTING VIDEO PROCESSING PIPELINE")
            self.logger.info("="*70)
            
            # Step 1: Analyze video if no config provided
            if processing_config is None:
                self.logger.info("Step 1: Analyzing video for optimal processing strategy...")
                video_metadata = self.video_analyzer.analyze_video(input_video_path)
                processing_config = self._determine_processing_config(video_metadata)
            
            self.logger.info(f"Processing configuration:")
            self.logger.info(f"  Tool: {processing_config.tool.value}")
            self.logger.info(f"  Model: {processing_config.model}")
            self.logger.info(f"  Scale Factor: {processing_config.scale_factor}x")
            self.logger.info(f"  Tile Size: {processing_config.tile_size}")
            self.logger.info(f"  GPU Acceleration: {processing_config.gpu_acceleration}")
            
            # Step 2: Frame extraction and preprocessing
            success = self.extract_and_preprocess_frames(input_video_path, processing_config)
            if not success:
                self.logger.error("Frame extraction and preprocessing failed")
                return False
            
            # Step 3: AI upscaling
            success = self.upscale_frames(processing_config)
            if not success:
                self.logger.error("AI upscaling failed")
                return False
            
            # Step 4: Video reassembly
            success = self.reassemble_video(input_video_path, output_video_path, processing_config)
            if not success:
                self.logger.error("Video reassembly failed")
                return False
            
            self.logger.info("="*70)
            self.logger.info("VIDEO PROCESSING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*70)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            return False
    
    def extract_and_preprocess_frames(self, input_video_path: str, 
                                    processing_config: ProcessingConfig) -> bool:
        """
        Extract frames from video and apply preprocessing.
        
        Args:
            input_video_path: Path to input video
            processing_config: Processing configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("="*50)
            self.logger.info("STEP 1: FRAME EXTRACTION AND PREPROCESSING")
            self.logger.info("="*50)
            
            # Create workspace directories
            frames_dir = os.path.join(self.workspace_dir, "frames")
            audio_path = os.path.join(self.workspace_dir, "temp", "audio.aac")
            
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            
            # Extract frames
            self.logger.info("Extracting video frames...")
            success, frame_count = self.frame_processor.extract_frames(
                input_video_path, frames_dir, format="png"
            )
            
            if not success:
                self.logger.error("Frame extraction failed")
                return False
            
            self.processing_stats['total_frames'] = frame_count
            self.logger.info(f"Successfully extracted {frame_count} frames")
            
            # Extract audio separately
            self.logger.info("Extracting audio track...")
            success = self.frame_processor.extract_audio(input_video_path, audio_path)
            
            if not success:
                self.logger.error("Audio extraction failed")
                return False
            
            self.logger.info("Successfully extracted audio track")
            
            # Apply preprocessing filters
            self.logger.info("Applying preprocessing filters...")
            success = self.frame_processor.apply_preprocessing(frames_dir, processing_config)
            
            if not success:
                self.logger.error("Preprocessing failed")
                return False
            
            self.logger.info("Preprocessing completed successfully")
            
            # Validate frame sequence
            self.logger.info("Validating frame sequence...")
            success = self.frame_processor.validate_frame_sequence(frames_dir, frame_count)
            
            if not success:
                self.logger.error("Frame sequence validation failed")
                return False
            
            self.logger.info("Frame sequence validation passed")
            
            # Store paths for next steps
            self.processing_stats['frames_dir'] = frames_dir
            self.processing_stats['audio_path'] = audio_path
            
            return True
            
        except Exception as e:
            self.logger.error(f"Frame extraction and preprocessing failed: {str(e)}")
            return False
    
    def upscale_frames(self, processing_config: ProcessingConfig) -> bool:
        """
        Upscale frames using AI model.
        
        Args:
            processing_config: Processing configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("="*50)
            self.logger.info("STEP 2: AI UPSCALING PROCESS")
            self.logger.info("="*50)
            
            # Initialize AI model
            self.logger.info(f"Initializing AI model: {processing_config.model}")
            success = self.ai_upscaler.initialize_model(
                processing_config.model, 
                processing_config.scale_factor
            )
            
            if not success:
                self.logger.error("AI model initialization failed")
                return False
            
            # Set up directories
            input_frames_dir = self.processing_stats['frames_dir']
            output_frames_dir = os.path.join(self.workspace_dir, "enhanced")
            os.makedirs(output_frames_dir, exist_ok=True)
            
            # Process frames in batches
            self.logger.info("Starting AI upscaling process...")
            success, upscaling_stats = self.ai_upscaler.upscale_frames_batch(
                input_frames_dir, output_frames_dir, processing_config
            )
            
            if not success:
                self.logger.error("AI upscaling failed")
                return False
            
            # Update processing stats
            self.processing_stats.update(upscaling_stats)
            self.processing_stats['enhanced_frames_dir'] = output_frames_dir
            
            self.logger.info("AI upscaling completed successfully")
            self.logger.info(f"Processed {upscaling_stats.get('processed_frames', 0)} frames")
            self.logger.info(f"Failed frames: {upscaling_stats.get('failed_frames', 0)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"AI upscaling failed: {str(e)}")
            return False
    
    def reassemble_video(self, input_video_path: str, output_video_path: str,
                        processing_config: ProcessingConfig) -> bool:
        """
        Reassemble enhanced frames into final 4K video.
        
        Args:
            input_video_path: Original video path (for metadata)
            output_video_path: Final output path
            processing_config: Processing configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("="*50)
            self.logger.info("STEP 3: VIDEO REASSEMBLY")
            self.logger.info("="*50)
            
            # Get video metadata for reassembly
            video_metadata = self.video_analyzer.analyze_video(input_video_path)
            
            # Calculate output resolution
            input_width, input_height = video_metadata['resolution']
            output_width = input_width * processing_config.scale_factor
            output_height = input_height * processing_config.scale_factor
            output_resolution = (output_width, output_height)
            
            self.logger.info(f"Input resolution: {input_width}x{input_height}")
            self.logger.info(f"Output resolution: {output_width}x{output_height}")
            
            # Reassemble video
            enhanced_frames_dir = self.processing_stats['enhanced_frames_dir']
            audio_path = self.processing_stats['audio_path']
            frame_rate = video_metadata['frame_rate']
            
            self.logger.info("Reassembling enhanced frames into 4K video...")
            success = self.frame_processor.reassemble_video(
                enhanced_frames_dir, audio_path, output_video_path,
                frame_rate, output_resolution
            )
            
            if not success:
                self.logger.error("Video reassembly failed")
                return False
            
            self.logger.info(f"Successfully created 4K video: {output_video_path}")
            
            # Validate output
            if os.path.exists(output_video_path):
                output_size = os.path.getsize(output_video_path)
                self.logger.info(f"Output file size: {output_size / (1024*1024*1024):.2f} GB")
                
                # Quick validation of output video
                try:
                    output_metadata = self.video_analyzer.analyze_video(output_video_path)
                    output_res = output_metadata['resolution']
                    self.logger.info(f"Verified output resolution: {output_res[0]}x{output_res[1]}")
                    
                    if output_res[0] >= 3840 and output_res[1] >= 2160:
                        self.logger.info("✅ 4K resolution target achieved!")
                    else:
                        self.logger.warning("⚠️ Output resolution is below 4K target")
                        
                except Exception as e:
                    self.logger.warning(f"Could not validate output video: {str(e)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Video reassembly failed: {str(e)}")
            return False
    
    def _determine_processing_config(self, video_metadata: Dict[str, Any]) -> ProcessingConfig:
        """Determine optimal processing configuration based on video analysis."""
        # For theater content, use Real-ESRGAN with quality settings
        return ProcessingConfig(
            tool=ProcessingTool.REAL_ESRGAN,
            model="RealESRGAN_x4plus",
            scale_factor=2,  # 1080p to 4K requires 2x scaling
            tile_size=512,   # Conservative for 4GB GPU
            gpu_acceleration=True,
            denoise_level=2,  # Moderate denoising for theater content
            environment=ProcessingEnvironment.WSL,
            color_correction=True  # Enable theater-specific preprocessing
        )
    
    def cleanup_intermediate_files(self, keep_enhanced: bool = True) -> None:
        """
        Clean up intermediate processing files.
        
        Args:
            keep_enhanced: Whether to keep enhanced frames for inspection
        """
        try:
            cleanup_dirs = []
            
            # Always clean up original frames and temp files
            if 'frames_dir' in self.processing_stats:
                cleanup_dirs.append(self.processing_stats['frames_dir'])
            
            temp_dir = os.path.join(self.workspace_dir, "temp")
            if os.path.exists(temp_dir):
                cleanup_dirs.append(temp_dir)
            
            # Optionally clean up enhanced frames
            if not keep_enhanced and 'enhanced_frames_dir' in self.processing_stats:
                cleanup_dirs.append(self.processing_stats['enhanced_frames_dir'])
            
            self.frame_processor.cleanup_intermediate_files(cleanup_dirs)
            self.logger.info("Intermediate file cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {str(e)}")
    
    def generate_processing_report(self, output_path: str = "processing_report.json") -> None:
        """Generate a detailed processing report."""
        try:
            report = {
                "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline_version": "1.0",
                "processing_stats": self.processing_stats,
                "success": True
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Processing report saved to: {output_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not generate processing report: {str(e)}")


def main():
    """Main function to run the video processing pipeline."""
    # Configuration
    input_video = "videos/大学生原创话剧《自杀既遂》.mp4"
    output_video = "workspace/enhanced/theater_drama_4k.mp4"
    
    # Verify input file exists
    if not os.path.exists(input_video):
        print(f"Error: Input video file not found: {input_video}")
        return False
    
    # Initialize pipeline
    config_manager = ConfigManager()
    pipeline = VideoProcessingPipeline(config_manager)
    
    try:
        # Run complete pipeline
        success = pipeline.process_video_to_4k(input_video, output_video)
        
        if success:
            print("\n" + "="*70)
            print("🎉 VIDEO PROCESSING COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"Input:  {input_video}")
            print(f"Output: {output_video}")
            
            # Generate processing report
            pipeline.generate_processing_report()
            
            # Optional cleanup
            print("\nCleaning up intermediate files...")
            pipeline.cleanup_intermediate_files(keep_enhanced=False)
            
            return True
        else:
            print("\n" + "="*70)
            print("❌ VIDEO PROCESSING FAILED")
            print("="*70)
            print("Check the logs for detailed error information.")
            return False
            
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)