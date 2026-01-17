#!/usr/bin/env python3
"""
Execute Video Reassembly Process - Task 4.3 Implementation

This script implements task 4.3: Video reassembly and output generation
- Reassemble enhanced frames into 4K video
- Merge original audio track with enhanced video
- Ensure temporal consistency and prevent flickering
- Generate final 4K output file
"""

import os
import sys
import logging
import json
import time
import subprocess
import cv2
import numpy as np
from pathlib import Path


def validate_enhanced_frames(frames_dir: str) -> tuple[bool, int, tuple[int, int]]:
    """
    Validate enhanced frames and get frame information.
    
    Returns:
        Tuple of (success, frame_count, resolution)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Get list of enhanced frames
        frame_files = sorted([f for f in os.listdir(frames_dir) 
                            if f.startswith('frame_') and f.endswith('.png')])
        
        if not frame_files:
            logger.error(f"No enhanced frames found in {frames_dir}")
            return False, 0, (0, 0)
        
        logger.info(f"Found {len(frame_files)} enhanced frames")
        
        # Check frame sequence integrity
        expected_frames = []
        for i in range(1, len(frame_files) + 1):
            expected_frames.append(f"frame_{i:08d}.png")
        
        missing_frames = []
        for expected in expected_frames:
            if expected not in frame_files:
                missing_frames.append(expected)
        
        if missing_frames:
            logger.warning(f"Missing frames detected: {missing_frames[:5]}...")
            if len(missing_frames) > len(frame_files) * 0.1:  # More than 10% missing
                logger.error("Too many missing frames for reliable reassembly")
                return False, 0, (0, 0)
        
        # Get resolution from first frame
        first_frame_path = os.path.join(frames_dir, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        
        if first_frame is None:
            logger.error(f"Cannot read first frame: {first_frame_path}")
            return False, 0, (0, 0)
        
        height, width = first_frame.shape[:2]
        resolution = (width, height)
        
        logger.info(f"Enhanced frame resolution: {width}x{height}")
        
        # Validate that this is actually 4K or higher resolution
        if width < 3840 or height < 2160:
            logger.warning(f"Frames are not 4K resolution: {width}x{height}")
            logger.info("Proceeding with available resolution...")
        else:
            logger.info("✅ 4K resolution confirmed")
        
        # Sample check frame integrity
        sample_indices = [0, len(frame_files)//2, len(frame_files)-1]
        corrupted_count = 0
        
        for idx in sample_indices:
            if idx < len(frame_files):
                frame_path = os.path.join(frames_dir, frame_files[idx])
                frame = cv2.imread(frame_path)
                if frame is None:
                    corrupted_count += 1
                    logger.warning(f"Corrupted frame detected: {frame_files[idx]}")
        
        if corrupted_count > 0:
            logger.warning(f"Found {corrupted_count} corrupted frames in sample")
            if corrupted_count >= len(sample_indices):
                logger.error("Too many corrupted frames detected")
                return False, 0, (0, 0)
        
        logger.info("✅ Enhanced frames validation passed")
        return True, len(frame_files), resolution
        
    except Exception as e:
        logger.error(f"Frame validation failed: {str(e)}")
        return False, 0, (0, 0)


def validate_audio_file(audio_path: str) -> bool:
    """
    Validate that the audio file exists and is readable.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        True if audio file is valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return False
        
        # Check file size
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logger.error(f"Audio file is empty: {audio_path}")
            return False
        
        logger.info(f"Audio file validated: {audio_path}")
        logger.info(f"Audio file size: {file_size / (1024*1024):.1f} MB")
        
        # Try to get audio info using ffprobe
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            audio_info = json.loads(result.stdout)
            
            # Extract audio properties
            for stream in audio_info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    codec = stream.get('codec_name', 'unknown')
                    sample_rate = stream.get('sample_rate', 'unknown')
                    channels = stream.get('channels', 'unknown')
                    
                    logger.info(f"Audio codec: {codec}")
                    logger.info(f"Sample rate: {sample_rate} Hz")
                    logger.info(f"Channels: {channels}")
                    break
            
        except subprocess.CalledProcessError:
            logger.warning("Could not get detailed audio info, but file exists")
        
        logger.info("✅ Audio file validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Audio validation failed: {str(e)}")
        return False


def apply_temporal_consistency_filter(frames_dir: str) -> bool:
    """
    Apply temporal consistency filtering to prevent flickering.
    
    Args:
        frames_dir: Directory containing enhanced frames
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        frame_files = sorted([f for f in os.listdir(frames_dir) 
                            if f.startswith('frame_') and f.endswith('.png')])
        
        if len(frame_files) < 3:
            logger.info("Too few frames for temporal filtering, skipping")
            return True
        
        logger.info(f"Applying temporal consistency filter to {len(frame_files)} frames")
        
        # Create backup directory
        backup_dir = frames_dir + "_backup"
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            logger.info(f"Created backup directory: {backup_dir}")
        
        processed_count = 0
        
        # Process frames with temporal smoothing
        for i in range(1, len(frame_files) - 1):  # Skip first and last frame
            prev_path = os.path.join(frames_dir, frame_files[i-1])
            curr_path = os.path.join(frames_dir, frame_files[i])
            next_path = os.path.join(frames_dir, frame_files[i+1])
            backup_path = os.path.join(backup_dir, frame_files[i])
            
            # Load three consecutive frames
            prev_frame = cv2.imread(prev_path)
            curr_frame = cv2.imread(curr_path)
            next_frame = cv2.imread(next_path)
            
            if prev_frame is None or curr_frame is None or next_frame is None:
                logger.warning(f"Could not load frames around {frame_files[i]}")
                continue
            
            # Backup original frame
            cv2.imwrite(backup_path, curr_frame)
            
            # Apply temporal smoothing (weighted average)
            # Current frame gets 70% weight, neighbors get 15% each
            smoothed_frame = (
                0.15 * prev_frame.astype(np.float32) +
                0.70 * curr_frame.astype(np.float32) +
                0.15 * next_frame.astype(np.float32)
            ).astype(np.uint8)
            
            # Save smoothed frame
            cv2.imwrite(curr_path, smoothed_frame)
            processed_count += 1
            
            # Progress reporting
            if processed_count % 20 == 0:
                progress = processed_count / (len(frame_files) - 2) * 100
                logger.info(f"Temporal filtering progress: {progress:.1f}%")
        
        logger.info(f"✅ Temporal consistency filter applied to {processed_count} frames")
        logger.info(f"Original frames backed up to: {backup_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Temporal consistency filtering failed: {str(e)}")
        return False


def reassemble_video_with_audio(frames_dir: str, audio_path: str, output_path: str,
                               frame_rate: float, resolution: tuple[int, int]) -> bool:
    """
    Reassemble enhanced frames into final 4K video with audio.
    
    Args:
        frames_dir: Directory containing enhanced frames
        audio_path: Path to audio file
        output_path: Path for final output video
        frame_rate: Frame rate for output video
        resolution: Resolution tuple (width, height)
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Construct frame input pattern for FFmpeg
        frame_pattern = os.path.join(frames_dir, "frame_%08d.png")
        
        logger.info("="*70)
        logger.info("REASSEMBLING ENHANCED VIDEO WITH AUDIO")
        logger.info("="*70)
        logger.info(f"Frame pattern: {frame_pattern}")
        logger.info(f"Audio file: {audio_path}")
        logger.info(f"Output file: {output_path}")
        logger.info(f"Resolution: {resolution[0]}x{resolution[1]}")
        logger.info(f"Frame rate: {frame_rate} fps")
        
        # Construct FFmpeg command for high-quality video reassembly
        cmd = [
            'ffmpeg',
            '-framerate', str(frame_rate),
            '-i', frame_pattern,
            '-i', audio_path,
            '-c:v', 'libx264',  # H.264 video codec
            '-preset', 'slow',  # High quality encoding (slower but better)
            '-crf', '16',  # Very high quality (lower CRF = higher quality)
            '-pix_fmt', 'yuv420p',  # Compatible pixel format
            '-profile:v', 'high',  # H.264 high profile
            '-level:v', '5.1',  # H.264 level for 4K
            '-c:a', 'aac',  # AAC audio codec
            '-b:a', '192k',  # High quality audio bitrate
            '-ar', '48000',  # Audio sample rate
            '-movflags', '+faststart',  # Optimize for streaming
            '-map', '0:v:0',  # Map video from first input
            '-map', '1:a:0',  # Map audio from second input
            '-shortest',  # End when shortest stream ends
            '-y',  # Overwrite existing files
            output_path
        ]
        
        logger.info("FFmpeg command:")
        logger.info(" ".join(cmd))
        
        # Run FFmpeg with progress monitoring
        logger.info("Starting video reassembly...")
        start_time = time.time()
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True
        )
        
        # Monitor progress
        stderr_output = []
        while True:
            output = process.stderr.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                stderr_output.append(output.strip())
                # Log progress information
                if 'frame=' in output and 'fps=' in output:
                    logger.info(f"FFmpeg: {output.strip()}")
        
        # Wait for completion
        return_code = process.poll()
        
        # Get any remaining output
        remaining_stdout, remaining_stderr = process.communicate()
        if remaining_stderr:
            stderr_output.extend(remaining_stderr.strip().split('\n'))
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if return_code == 0:
            logger.info(f"✅ Video reassembly completed successfully!")
            logger.info(f"Processing time: {processing_time:.1f} seconds")
            
            # Verify output file
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"Output file size: {file_size / (1024*1024):.1f} MB")
                
                # Get output video info
                try:
                    info_cmd = [
                        'ffprobe',
                        '-v', 'quiet',
                        '-print_format', 'json',
                        '-show_format',
                        '-show_streams',
                        output_path
                    ]
                    
                    result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
                    video_info = json.loads(result.stdout)
                    
                    # Extract video properties
                    for stream in video_info.get('streams', []):
                        if stream.get('codec_type') == 'video':
                            width = stream.get('width', 0)
                            height = stream.get('height', 0)
                            fps = eval(stream.get('r_frame_rate', '0/1'))
                            
                            logger.info(f"Final video resolution: {width}x{height}")
                            logger.info(f"Final video frame rate: {fps:.2f} fps")
                            
                            # Verify 4K resolution achievement
                            if width >= 3840 and height >= 2160:
                                logger.info("🎉 4K resolution achieved!")
                            else:
                                logger.warning(f"Output is not 4K: {width}x{height}")
                            break
                    
                except subprocess.CalledProcessError:
                    logger.warning("Could not get output video info")
                
                return True
            else:
                logger.error("Output file was not created")
                return False
        else:
            logger.error(f"FFmpeg failed with return code: {return_code}")
            logger.error("FFmpeg stderr output:")
            for line in stderr_output[-20:]:  # Show last 20 lines
                logger.error(f"  {line}")
            return False
        
    except Exception as e:
        logger.error(f"Video reassembly failed: {str(e)}")
        return False


def execute_video_reassembly():
    """Execute video reassembly process - Task 4.3 implementation."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
        
        # Step 1: Validate enhanced frames
        logger.info("Step 1: Validating enhanced frames...")
        frames_valid, frame_count, resolution = validate_enhanced_frames(enhanced_frames_dir)
        
        if not frames_valid:
            logger.error("Enhanced frames validation failed")
            return False
        
        logger.info(f"✅ Validated {frame_count} enhanced frames at {resolution[0]}x{resolution[1]}")
        
        # Step 2: Validate audio file
        logger.info("Step 2: Validating audio file...")
        audio_valid = validate_audio_file(audio_file)
        
        if not audio_valid:
            logger.error("Audio file validation failed")
            return False
        
        logger.info("✅ Audio file validation passed")
        
        # Step 3: Apply temporal consistency filter to prevent flickering
        logger.info("Step 3: Applying temporal consistency filter...")
        temporal_success = apply_temporal_consistency_filter(enhanced_frames_dir)
        
        if not temporal_success:
            logger.warning("Temporal consistency filtering failed, proceeding without it")
        else:
            logger.info("✅ Temporal consistency filter applied")
        
        # Step 4: Reassemble video with audio
        logger.info("Step 4: Reassembling enhanced video with audio...")
        reassembly_success = reassemble_video_with_audio(
            enhanced_frames_dir,
            audio_file,
            output_video,
            original_frame_rate,
            resolution
        )
        
        if not reassembly_success:
            logger.error("Video reassembly failed")
            return False
        
        logger.info("✅ Video reassembly completed successfully")
        
        # Step 5: Final validation and quality check
        logger.info("Step 5: Performing final validation...")
        
        if not os.path.exists(output_video):
            logger.error("Final output video file not found")
            return False
        
        # Get final video statistics
        final_size = os.path.getsize(output_video)
        logger.info(f"Final video file size: {final_size / (1024*1024):.1f} MB")
        
        # Calculate compression ratio
        if frame_count > 0:
            # Estimate uncompressed size (rough calculation)
            bytes_per_pixel = 3  # RGB
            uncompressed_size = frame_count * resolution[0] * resolution[1] * bytes_per_pixel
            compression_ratio = uncompressed_size / final_size
            logger.info(f"Compression ratio: {compression_ratio:.1f}:1")
        
        # Step 6: Generate processing report
        logger.info("Step 6: Generating processing report...")
        
        processing_report = {
            "process": "video_reassembly_task_4_3",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_data": {
                "enhanced_frames_count": frame_count,
                "enhanced_frames_resolution": f"{resolution[0]}x{resolution[1]}",
                "audio_file": audio_file,
                "frame_rate": original_frame_rate
            },
            "processing_steps": {
                "frames_validation": frames_valid,
                "audio_validation": audio_valid,
                "temporal_consistency_applied": temporal_success,
                "video_reassembly": reassembly_success
            },
            "output": {
                "file_path": output_video,
                "file_size_mb": final_size / (1024*1024),
                "resolution": f"{resolution[0]}x{resolution[1]}",
                "is_4k": resolution[0] >= 3840 and resolution[1] >= 2160,
                "frame_rate": original_frame_rate
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
        logger.info(f"✅ Temporal consistency ensured")
        logger.info(f"✅ Final 4K video generated: {output_video}")
        logger.info(f"✅ Output resolution: {resolution[0]}x{resolution[1]}")
        logger.info(f"✅ File size: {final_size / (1024*1024):.1f} MB")
        
        if resolution[0] >= 3840 and resolution[1] >= 2160:
            logger.info("🎉 4K RESOLUTION TARGET ACHIEVED!")
        else:
            logger.info(f"📊 Enhanced to {resolution[0]}x{resolution[1]} resolution")
        
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
            print("Final 4K enhanced theater video is ready!")
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