#!/usr/bin/env python3
"""
Execute AI Upscaling Process - Simplified Version

This script implements task 4.2: Execute AI upscaling process
Based on the working test_ai_upscaling.py approach
"""

import os
import sys
import logging
import json
import time
import cv2
import numpy as np

# Add video_enhancer to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'video_enhancer'))

from video_enhancer.config import ConfigManager
from video_enhancer.models import ProcessingConfig, ProcessingTool, ProcessingEnvironment
from video_enhancer.ai_upscaler import AIUpscaler


def monitor_gpu_memory():
    """Monitor GPU memory usage if available."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "usage_percent": (reserved / total) * 100
            }
    except ImportError:
        pass
    return None


def execute_ai_upscaling():
    """Execute AI upscaling process with monitoring and error handling."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ai_upscaling_execution.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Configuration
    input_frames_dir = "workspace/test/frames"  # From task 4.1
    output_frames_dir = "workspace/enhanced"
    
    # Load processing strategy
    strategy_file = "theater_video_processing_strategy.json"
    if os.path.exists(strategy_file):
        with open(strategy_file, 'r', encoding='utf-8') as f:
            strategy = json.load(f)
        logger.info(f"Loaded processing strategy from {strategy_file}")
    else:
        # Default strategy
        strategy = {
            "ai_model": "RealESRGAN_x4plus",
            "scale_factor": 2,
            "tile_size": 640,
            "gpu_acceleration": True,
            "denoise_level": 2
        }
        logger.warning("Using default processing strategy")
    
    # Verify input frames exist
    if not os.path.exists(input_frames_dir):
        logger.error(f"Input frames directory not found: {input_frames_dir}")
        logger.info("Please ensure task 4.1 (frame extraction) has been completed first.")
        return False
    
    frame_files = [f for f in os.listdir(input_frames_dir) if f.endswith('.png')]
    if not frame_files:
        logger.error("No frame files found in input directory")
        return False
    
    # Create output directory
    os.makedirs(output_frames_dir, exist_ok=True)
    
    try:
        # Initialize components
        config_manager = ConfigManager()
        ai_upscaler = AIUpscaler(config_manager)
        
        logger.info("="*70)
        logger.info("EXECUTING AI UPSCALING PROCESS")
        logger.info("="*70)
        
        logger.info(f"Input frames: {len(frame_files)} files")
        logger.info(f"Input directory: {input_frames_dir}")
        logger.info(f"Output directory: {output_frames_dir}")
        
        # Create processing configuration
        processing_config = ProcessingConfig(
            tool=ProcessingTool.REAL_ESRGAN,
            model=strategy.get("ai_model", "RealESRGAN_x4plus"),
            scale_factor=strategy.get("scale_factor", 2),
            tile_size=strategy.get("tile_size", 640),
            gpu_acceleration=strategy.get("gpu_acceleration", True),
            denoise_level=strategy.get("denoise_level", 2),
            environment=ProcessingEnvironment.WSL,
            color_correction=True
        )
        
        logger.info(f"Processing configuration:")
        logger.info(f"  Model: {processing_config.model}")
        logger.info(f"  Scale factor: {processing_config.scale_factor}x")
        logger.info(f"  Tile size: {processing_config.tile_size}")
        logger.info(f"  GPU acceleration: {processing_config.gpu_acceleration}")
        
        # Monitor initial GPU memory
        initial_gpu = monitor_gpu_memory()
        if initial_gpu:
            logger.info(f"Initial GPU memory usage: {initial_gpu['usage_percent']:.1f}%")
        
        # Step 1: Try to initialize Real-ESRGAN model
        logger.info("Step 1: Initializing AI upscaling model...")
        start_time = time.time()
        
        model_success = ai_upscaler.initialize_model(
            processing_config.model, 
            processing_config.scale_factor
        )
        
        if model_success:
            logger.info("✅ AI model initialized successfully")
            
            # Get model info
            model_info = ai_upscaler.get_model_info()
            logger.info(f"Model info: {model_info}")
            
            # Step 2: Process frames with AI upscaling
            logger.info("Step 2: Processing frames with AI upscaling...")
            success, stats = ai_upscaler.upscale_frames_batch(
                input_frames_dir, output_frames_dir, processing_config
            )
        else:
            logger.warning("❌ AI model initialization failed")
            success = False
        
        # Step 3: Use fallback if AI upscaling failed
        if not success:
            logger.info("Step 3: Using fallback upscaling method...")
            success, stats = ai_upscaler.upscale_with_fallback(
                input_frames_dir, output_frames_dir, processing_config
            )
        
        if not success:
            logger.error("Both AI upscaling and fallback methods failed")
            return False
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Step 4: Validate results and handle failed frames
        logger.info("Step 4: Validating upscaling results...")
        
        output_files = [f for f in os.listdir(output_frames_dir) if f.endswith('.png')]
        logger.info(f"Output frames created: {len(output_files)}")
        
        # Validate frame integrity and attempt recovery for failed frames
        validated_frames = 0
        failed_frames = []
        
        for frame_file in frame_files:
            output_path = os.path.join(output_frames_dir, frame_file)
            
            if not os.path.exists(output_path):
                failed_frames.append(frame_file)
                continue
            
            # Check frame integrity
            frame = cv2.imread(output_path)
            if frame is None:
                failed_frames.append(frame_file)
                continue
            
            validated_frames += 1
        
        logger.info(f"Validated frames: {validated_frames}")
        logger.info(f"Failed frames: {len(failed_frames)}")
        
        # Attempt recovery for failed frames
        recovered_frames = 0
        if failed_frames:
            logger.info(f"Attempting recovery for {len(failed_frames)} failed frames...")
            
            for frame_file in failed_frames:
                input_path = os.path.join(input_frames_dir, frame_file)
                output_path = os.path.join(output_frames_dir, frame_file)
                
                # Try single frame upscaling
                recovery_success = ai_upscaler._upscale_single_frame(input_path, output_path)
                
                if recovery_success:
                    # Validate recovered frame
                    frame = cv2.imread(output_path)
                    if frame is not None:
                        recovered_frames += 1
                        logger.debug(f"Recovered frame: {frame_file}")
            
            logger.info(f"Recovered frames: {recovered_frames}")
        
        # Step 5: Final validation and quality check
        if output_files:
            # Check dimensions of sample frames
            sample_input = os.path.join(input_frames_dir, frame_files[0])
            sample_output = os.path.join(output_frames_dir, output_files[0])
            
            input_img = cv2.imread(sample_input)
            output_img = cv2.imread(sample_output)
            
            if input_img is not None and output_img is not None:
                input_h, input_w = input_img.shape[:2]
                output_h, output_w = output_img.shape[:2]
                
                logger.info(f"Input dimensions: {input_w}x{input_h}")
                logger.info(f"Output dimensions: {output_w}x{output_h}")
                
                scale_w = output_w / input_w
                scale_h = output_h / input_h
                
                logger.info(f"Actual scale factor: {scale_w:.1f}x{scale_h:.1f}")
                
                if abs(scale_w - processing_config.scale_factor) < 0.1:
                    logger.info("✅ Scale factor verification passed")
                else:
                    logger.warning("⚠️ Scale factor verification failed")
            
            # Check file sizes
            input_size = os.path.getsize(sample_input)
            output_size = os.path.getsize(sample_output)
            
            logger.info(f"Sample input size: {input_size / 1024:.1f} KB")
            logger.info(f"Sample output size: {output_size / 1024:.1f} KB")
        
        # Monitor final GPU memory
        final_gpu = monitor_gpu_memory()
        if final_gpu:
            logger.info(f"Final GPU memory usage: {final_gpu['usage_percent']:.1f}%")
        
        # Step 6: Generate processing report
        logger.info("Step 5: Generating processing report...")
        
        final_stats = {
            "total_frames": len(frame_files),
            "processed_frames": validated_frames + recovered_frames,
            "failed_frames": len(failed_frames) - recovered_frames,
            "recovered_frames": recovered_frames,
            "processing_time_minutes": total_time / 60,
            "processing_rate_fps": len(frame_files) / total_time if total_time > 0 else 0,
            "method_used": stats.get('method', 'real-esrgan'),
            "scale_factor": processing_config.scale_factor,
            "tile_size": processing_config.tile_size
        }
        
        if stats:
            final_stats.update(stats)
        
        # Save processing report
        report = {
            "process": "ai_upscaling_execution",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                "model": processing_config.model,
                "scale_factor": processing_config.scale_factor,
                "tile_size": processing_config.tile_size,
                "gpu_acceleration": processing_config.gpu_acceleration
            },
            "statistics": final_stats,
            "success": final_stats["failed_frames"] == 0
        }
        
        report_path = "ai_upscaling_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processing report saved to: {report_path}")
        
        # Final results
        success_rate = (final_stats["processed_frames"] / final_stats["total_frames"]) * 100
        
        logger.info("="*70)
        logger.info("AI UPSCALING PROCESS COMPLETED")
        logger.info("="*70)
        logger.info(f"Total frames: {final_stats['total_frames']}")
        logger.info(f"Successfully processed: {final_stats['processed_frames']}")
        logger.info(f"Failed frames: {final_stats['failed_frames']}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Processing time: {final_stats['processing_time_minutes']:.1f} minutes")
        logger.info(f"Processing rate: {final_stats['processing_rate_fps']:.2f} fps")
        logger.info(f"Method used: {final_stats['method_used']}")
        
        # Consider successful if > 95% of frames processed
        overall_success = success_rate > 95.0
        
        if overall_success:
            logger.info("✅ AI upscaling process completed successfully!")
        else:
            logger.warning("⚠️ AI upscaling completed with some failures")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"AI upscaling execution failed: {str(e)}")
        return False


def main():
    """Main function to execute AI upscaling process."""
    
    try:
        success = execute_ai_upscaling()
        
        if success:
            print("\n" + "="*70)
            print("🎉 AI UPSCALING PROCESS COMPLETED SUCCESSFULLY!")
            print("="*70)
            print("Enhanced frames are ready for video reassembly (task 4.3)")
            return True
        else:
            print("\n" + "="*70)
            print("❌ AI UPSCALING PROCESS FAILED")
            print("="*70)
            print("Check the logs for detailed error information.")
            return False
            
    except KeyboardInterrupt:
        print("\n\nUpscaling process interrupted by user.")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)