#!/usr/bin/env python3
"""
Execute Video Reassembly Process - Task 4.3 Implementation (Simplified)

This script implements task 4.3: Video reassembly and output generation
- Reassemble enhanced frames into 4K video
- Merge original audio track with enhanced video
- Generate final 4K output file
"""

import os
import sys
import logging
import json
import time
import subprocess


def execute_video_reassembly():
    """Execute video reassembly process - Task 4.3 implementation."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('video_reassembly_execution.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("="*70)
        logger.info("EXECUTING VIDEO REASSEMBLY PROCESS - TASK 4.3")
        logger.info("="*70)
        
        # Configuration
        enhanced_frames_dir = "workspace/enhanced"
        audio_file = "workspace/test/audio.aac"
        output_video = "enhanced_4k_theater_video.mp4"
        
        # From theater video analysis report
        original_frame_rate = 25.0  # fps
        
        logger.info(f"Enhanced frames directory: {enhanced_frames_dir}")
        logger.info(f"Audio file: {audio_file}")
        logger.info(f"Output video: {output_video}")
        logger.info(f"Target frame rate: {original_frame_rate} fps")
        
        # Step 1: Validate inputs
        logger.info("Step 1: Validating inputs...")
        
        # Check enhanced frames
        if not os.path.exists(enhanced_frames_dir):
            logger.error(f"Enhanced frames directory not found: {enhanced_frames_dir}")
            return False
        
        frame_files = [f for f in os.listdir(enhanced_frames_dir) 
                      if f.startswith('frame_') and f.endswith('.png')]
        frame_count = len(frame_files)
        
        if frame_count == 0:
            logger.error("No enhanced frames found")
            return False
        
        logger.info(f"Found {frame_count} enhanced frames")
        
        # Check audio file
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return False
        
        audio_size = os.path.getsize(audio_file)
        logger.info(f"Audio file size: {audio_size / (1024*1024):.1f} MB")
        
        # Step 2: Get frame resolution
        logger.info("Step 2: Checking frame resolution...")
        
        try:
            import cv2
            first_frame_path = os.path.join(enhanced_frames_dir, sorted(frame_files)[0])
            first_frame = cv2.imread(first_frame_path)
            
            if first_frame is not None:
                height, width = first_frame.shape[:2]
                logger.info(f"Enhanced frame resolution: {width}x{height}")
                
                if width >= 3840 and height >= 2160:
                    logger.info("✅ 4K resolution confirmed")
                else:
                    logger.info(f"Resolution: {width}x{height} (not 4K but proceeding)")
            else:
                logger.warning("Could not read first frame for resolution check")
                width, height = 3840, 2160  # Assume 4K
        except ImportError:
            logger.warning("OpenCV not available, assuming 4K resolution")
            width, height = 3840, 2160
        
        # Step 3: Reassemble video with FFmpeg
        logger.info("Step 3: Reassembling video with audio...")
        
        # Construct frame input pattern for FFmpeg
        frame_pattern = os.path.join(enhanced_frames_dir, "frame_%08d.png")
        
        # Construct FFmpeg command for high-quality video reassembly
        cmd = [
            'ffmpeg',
            '-framerate', str(original_frame_rate),
            '-i', frame_pattern,
            '-i', audio_file,
            '-c:v', 'libx264',  # H.264 video codec
            '-preset', 'medium',  # Balance between speed and quality
            '-crf', '18',  # High quality (lower CRF = higher quality)
            '-pix_fmt', 'yuv420p',  # Compatible pixel format
            '-c:a', 'aac',  # AAC audio codec
            '-b:a', '192k',  # High quality audio bitrate
            '-movflags', '+faststart',  # Optimize for streaming
            '-shortest',  # End when shortest stream ends
            '-y',  # Overwrite existing files
            output_video
        ]
        
        logger.info("FFmpeg command:")
        logger.info(" ".join(cmd))
        
        # Run FFmpeg
        logger.info("Starting video reassembly...")
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"✅ Video reassembly completed successfully!")
            logger.info(f"Processing time: {processing_time:.1f} seconds")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed with return code: {e.returncode}")
            logger.error("FFmpeg stderr output:")
            logger.error(e.stderr)
            return False
        
        # Step 4: Validate output
        logger.info("Step 4: Validating output...")
        
        if not os.path.exists(output_video):
            logger.error("Output video file was not created")
            return False
        
        final_size = os.path.getsize(output_video)
        logger.info(f"Final video file size: {final_size / (1024*1024):.1f} MB")
        
        # Get output video info
        try:
            info_cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                output_video
            ]
            
            result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
            video_info = json.loads(result.stdout)
            
            # Extract video properties
            for stream in video_info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    out_width = stream.get('width', 0)
                    out_height = stream.get('height', 0)
                    fps = eval(stream.get('r_frame_rate', '0/1'))
                    
                    logger.info(f"Final video resolution: {out_width}x{out_height}")
                    logger.info(f"Final video frame rate: {fps:.2f} fps")
                    
                    # Verify 4K resolution achievement
                    if out_width >= 3840 and out_height >= 2160:
                        logger.info("🎉 4K resolution achieved!")
                    else:
                        logger.info(f"Output resolution: {out_width}x{out_height}")
                    break
            
        except subprocess.CalledProcessError:
            logger.warning("Could not get output video info")
        
        # Step 5: Generate processing report
        logger.info("Step 5: Generating processing report...")
        
        processing_report = {
            "process": "video_reassembly_task_4_3",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_data": {
                "enhanced_frames_count": frame_count,
                "audio_file": audio_file,
                "frame_rate": original_frame_rate
            },
            "output": {
                "file_path": output_video,
                "file_size_mb": final_size / (1024*1024),
                "processing_time_seconds": processing_time
            },
            "success": True
        }
        
        report_path = "video_reassembly_task_4_3_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(processing_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processing report saved to: {report_path}")
        
        # Final results summary
        logger.info("="*70)
        logger.info("VIDEO REASSEMBLY PROCESS COMPLETED - TASK 4.3")
        logger.info("="*70)
        logger.info(f"✅ Enhanced frames processed: {frame_count}")
        logger.info(f"✅ Audio track merged successfully")
        logger.info(f"✅ Final video generated: {output_video}")
        logger.info(f"✅ File size: {final_size / (1024*1024):.1f} MB")
        logger.info(f"✅ Processing time: {processing_time:.1f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Task 4.3 video reassembly execution failed: {str(e)}")
        return False


def main():
    """Main function to execute Task 4.3 video reassembly process."""
    
    try:
        success = execute_video_reassembly()
        
        if success:
            print("\n" + "="*70)
            print("🎉 TASK 4.3 VIDEO REASSEMBLY COMPLETED SUCCESSFULLY!")
            print("="*70)
            print("Final enhanced theater video is ready!")
            print("Output file: enhanced_4k_theater_video.mp4")
            return True
        else:
            print("\n" + "="*70)
            print("❌ TASK 4.3 VIDEO REASSEMBLY FAILED")
            print("="*70)
            print("Check the logs for detailed error information.")
            return False
            
    except KeyboardInterrupt:
        print("\n\nVideo reassembly process interrupted by user.")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)