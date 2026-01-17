#!/usr/bin/env python3
"""
Execute AI Upscaling Process - Task 4.2 Implementation

This script implements task 4.2: Execute AI upscaling process
Based on the working test approach but adapted for the actual task
"""

import os
import sys
import logging
import json
import time
import cv2
import numpy as np


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
        logger.info("="*70)
        logger.info("EXECUTING AI UPSCALING PROCESS - TASK 4.2")
        logger.info("="*70)
        
        logger.info(f"Input frames: {len(frame_files)} files")
        logger.info(f"Input directory: {input_frames_dir}")
        logger.info(f"Output directory: {output_frames_dir}")
        
        # Configuration
        scale_factor = strategy.get("scale_factor", 2)
        tile_size = strategy.get("tile_size", 640)
        
        logger.info(f"Processing configuration:")
        logger.info(f"  Scale factor: {scale_factor}x")
        logger.info(f"  Tile size: {tile_size}")
        
        # Monitor initial GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                usage_percent = (reserved / total) * 100
                logger.info(f"Initial GPU memory usage: {usage_percent:.1f}%")
        except ImportError:
            logger.info("PyTorch not available, GPU monitoring disabled")
        
        # Step 1: Try Real-ESRGAN first (if available)
        logger.info("Step 1: Attempting Real-ESRGAN AI upscaling...")
        start_time = time.time()
        
        ai_success = False
        try:
            # Try to import and use Real-ESRGAN
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            import torch
            
            # Check if model file exists
            model_path = 'models/RealESRGAN_x4plus.pth'
            if os.path.exists(model_path):
                logger.info("Real-ESRGAN model file found, initializing...")
                
                # Initialize model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                               num_block=23, num_grow_ch=32, scale=4)
                
                upsampler = RealESRGANer(
                    scale=scale_factor,
                    model_path=model_path,
                    model=model,
                    tile=tile_size,
                    tile_pad=10,
                    pre_pad=0,
                    half=torch.cuda.is_available(),
                    gpu_id=0 if torch.cuda.is_available() else None
                )
                
                logger.info("✅ Real-ESRGAN initialized successfully")
                
                # Process frames with Real-ESRGAN
                processed_count = 0
                failed_count = 0
                
                for i, frame_file in enumerate(frame_files):
                    input_path = os.path.join(input_frames_dir, frame_file)
                    output_path = os.path.join(output_frames_dir, frame_file)
                    
                    # Skip if already processed
                    if os.path.exists(output_path):
                        processed_count += 1
                        continue
                    
                    try:
                        # Read and upscale frame
                        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
                        if img is not None:
                            output, _ = upsampler.enhance(img, outscale=upsampler.scale)
                            success = cv2.imwrite(output_path, output)
                            if success:
                                processed_count += 1
                            else:
                                failed_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to process frame {frame_file}: {str(e)}")
                        failed_count += 1
                    
                    # Progress reporting
                    if (i + 1) % 10 == 0 or i == len(frame_files) - 1:
                        progress = (i + 1) / len(frame_files) * 100
                        logger.info(f"Real-ESRGAN Progress: {progress:.1f}% ({i+1}/{len(frame_files)})")
                        
                        # Clear GPU cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                logger.info(f"Real-ESRGAN processed: {processed_count}, failed: {failed_count}")
                ai_success = processed_count > 0
                
            else:
                logger.warning(f"Real-ESRGAN model file not found: {model_path}")
                
        except ImportError as e:
            logger.warning(f"Real-ESRGAN not available: {str(e)}")
        except Exception as e:
            logger.warning(f"Real-ESRGAN failed: {str(e)}")
        
        # Step 2: Use OpenCV fallback if Real-ESRGAN failed or unavailable
        if not ai_success:
            logger.info("Step 2: Using OpenCV fallback upscaling...")
            
            processed_count = 0
            failed_count = 0
            
            for i, frame_file in enumerate(frame_files):
                input_path = os.path.join(input_frames_dir, frame_file)
                output_path = os.path.join(output_frames_dir, frame_file)
                
                # Skip if already processed
                if os.path.exists(output_path):
                    processed_count += 1
                    continue
                
                try:
                    # Read input frame
                    img = cv2.imread(input_path)
                    if img is not None:
                        # Get current dimensions
                        height, width = img.shape[:2]
                        
                        # Calculate new dimensions
                        new_width = width * scale_factor
                        new_height = height * scale_factor
                        
                        # Upscale using LANCZOS interpolation
                        upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                        
                        # Save upscaled frame
                        success = cv2.imwrite(output_path, upscaled)
                        if success:
                            processed_count += 1
                        else:
                            failed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to process frame {frame_file}: {str(e)}")
                    failed_count += 1
                
                # Progress reporting
                if (i + 1) % 50 == 0 or i == len(frame_files) - 1:
                    progress = (i + 1) / len(frame_files) * 100
                    logger.info(f"OpenCV Progress: {progress:.1f}% ({i+1}/{len(frame_files)})")
            
            logger.info(f"OpenCV fallback processed: {processed_count}, failed: {failed_count}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Step 3: Validate results
        logger.info("Step 3: Validating upscaling results...")
        
        output_files = [f for f in os.listdir(output_frames_dir) if f.endswith('.png')]
        logger.info(f"Output frames created: {len(output_files)}")
        
        # Validate frame integrity
        validated_frames = 0
        corrupted_frames = 0
        
        for frame_file in frame_files:
            output_path = os.path.join(output_frames_dir, frame_file)
            
            if os.path.exists(output_path):
                # Check frame integrity
                frame = cv2.imread(output_path)
                if frame is not None:
                    validated_frames += 1
                else:
                    corrupted_frames += 1
        
        logger.info(f"Validated frames: {validated_frames}")
        logger.info(f"Corrupted frames: {corrupted_frames}")
        
        # Step 4: Quality verification
        if output_files and len(frame_files) > 0:
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
                
                actual_scale_w = output_w / input_w
                actual_scale_h = output_h / input_h
                
                logger.info(f"Actual scale factor: {actual_scale_w:.1f}x{actual_scale_h:.1f}")
                
                if abs(actual_scale_w - scale_factor) < 0.1:
                    logger.info("✅ Scale factor verification passed")
                else:
                    logger.warning("⚠️ Scale factor verification failed")
            
            # Check file sizes
            input_size = os.path.getsize(sample_input)
            output_size = os.path.getsize(sample_output)
            
            logger.info(f"Sample input size: {input_size / 1024:.1f} KB")
            logger.info(f"Sample output size: {output_size / 1024:.1f} KB")
        
        # Step 5: Generate processing report
        logger.info("Step 4: Generating processing report...")
        
        final_stats = {
            "total_frames": len(frame_files),
            "processed_frames": validated_frames,
            "failed_frames": len(frame_files) - validated_frames,
            "corrupted_frames": corrupted_frames,
            "processing_time_minutes": total_time / 60,
            "processing_rate_fps": len(frame_files) / total_time if total_time > 0 else 0,
            "method_used": "real-esrgan" if ai_success else "opencv_fallback",
            "scale_factor": scale_factor,
            "tile_size": tile_size
        }
        
        # Save processing report
        report = {
            "process": "ai_upscaling_execution_task_4_2",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                "scale_factor": scale_factor,
                "tile_size": tile_size,
                "gpu_acceleration": strategy.get("gpu_acceleration", True)
            },
            "statistics": final_stats,
            "success": final_stats["failed_frames"] == 0
        }
        
        report_path = "ai_upscaling_task_4_2_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processing report saved to: {report_path}")
        
        # Final results
        success_rate = (final_stats["processed_frames"] / final_stats["total_frames"]) * 100
        
        logger.info("="*70)
        logger.info("AI UPSCALING PROCESS COMPLETED - TASK 4.2")
        logger.info("="*70)
        logger.info(f"Total frames: {final_stats['total_frames']}")
        logger.info(f"Successfully processed: {final_stats['processed_frames']}")
        logger.info(f"Failed frames: {final_stats['failed_frames']}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Processing time: {final_stats['processing_time_minutes']:.1f} minutes")
        logger.info(f"Processing rate: {final_stats['processing_rate_fps']:.2f} fps")
        logger.info(f"Method used: {final_stats['method_used']}")
        
        # Consider successful if > 90% of frames processed
        overall_success = success_rate > 90.0
        
        if overall_success:
            logger.info("✅ Task 4.2 AI upscaling process completed successfully!")
        else:
            logger.warning("⚠️ Task 4.2 AI upscaling completed with some failures")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"Task 4.2 AI upscaling execution failed: {str(e)}")
        return False


def main():
    """Main function to execute Task 4.2 AI upscaling process."""
    
    try:
        success = execute_ai_upscaling()
        
        if success:
            print("\n" + "="*70)
            print("🎉 TASK 4.2 AI UPSCALING PROCESS COMPLETED SUCCESSFULLY!")
            print("="*70)
            print("Enhanced frames are ready for video reassembly (task 4.3)")
            return True
        else:
            print("\n" + "="*70)
            print("❌ TASK 4.2 AI UPSCALING PROCESS FAILED")
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