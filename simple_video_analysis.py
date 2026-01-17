#!/usr/bin/env python3
"""
Simplified video analysis script for the theater drama video.
"""

import os
import subprocess
import json
import cv2
import numpy as np
from pathlib import Path


def analyze_video_with_ffprobe(video_path):
    """Extract video metadata using FFprobe."""
    try:
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
        
        return {
            'resolution': (width, height),
            'frame_rate': frame_rate,
            'duration': duration,
            'codec': video_stream.get('codec_name', 'unknown'),
            'bitrate': bitrate,
            'audio_tracks': len(audio_streams),
            'file_size': os.path.getsize(video_path)
        }
        
    except subprocess.CalledProcessError as e:
        print(f"FFprobe failed: {e.stderr}")
        return None
    except Exception as e:
        print(f"Failed to parse FFprobe output: {str(e)}")
        return None


def analyze_theater_characteristics(video_path):
    """Analyze theater-specific characteristics using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Could not open video file for analysis")
            return None
        
        # Sample frames for analysis (every 30 seconds)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = int(frame_rate * 30)  # Every 30 seconds
        
        frames_analyzed = 0
        brightness_levels = []
        complexity_scores = []
        
        for frame_num in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frames_analyzed += 1
            
            # Analyze brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightness_levels.append(brightness)
            
            # Analyze complexity using edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            complexity_scores.append(edge_density)
            
            # Limit analysis to prevent excessive processing
            if frames_analyzed >= 10:
                break
        
        cap.release()
        
        if frames_analyzed == 0:
            return None
        
        # Calculate statistics
        avg_brightness = np.mean(brightness_levels)
        avg_complexity = np.mean(complexity_scores)
        
        # Classify lighting type
        if avg_brightness > 120:
            lighting_type = "natural"
        elif avg_brightness < 80:
            lighting_type = "stage"
        else:
            lighting_type = "mixed"
        
        # Classify scene complexity
        if avg_complexity > 0.15:
            scene_complexity = "complex"
        elif avg_complexity > 0.08:
            scene_complexity = "moderate"
        else:
            scene_complexity = "simple"
        
        return {
            'lighting_type': lighting_type,
            'scene_complexity': scene_complexity,
            'avg_brightness': avg_brightness,
            'avg_complexity': avg_complexity,
            'frames_analyzed': frames_analyzed,
            'recommended_model': 'realesrgan-x4plus'  # Default recommendation
        }
        
    except Exception as e:
        print(f"Theater analysis failed: {str(e)}")
        return None


def main():
    """Main function to analyze the theater video."""
    video_path = "videos/大学生原创话剧《自杀既遂》.mp4"
    
    print("Theater Drama Video Analysis")
    print("=" * 50)
    print(f"Analyzing: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False
    
    # Get basic metadata
    print("\nExtracting video metadata...")
    metadata = analyze_video_with_ffprobe(video_path)
    
    if not metadata:
        print("Failed to extract video metadata")
        return False
    
    # Analyze theater characteristics
    print("Analyzing theater-specific characteristics...")
    theater_analysis = analyze_theater_characteristics(video_path)
    
    # Print results
    print("\n" + "="*60)
    print("VIDEO ANALYSIS REPORT")
    print("="*60)
    
    print(f"\nBASIC SPECIFICATIONS:")
    print(f"  Resolution: {metadata['resolution'][0]}x{metadata['resolution'][1]}")
    print(f"  Frame Rate: {metadata['frame_rate']:.2f} fps")
    print(f"  Duration: {metadata['duration']:.2f} seconds ({metadata['duration']/60:.1f} minutes)")
    print(f"  Video Codec: {metadata['codec']}")
    print(f"  Bitrate: {metadata['bitrate']:,} bps ({metadata['bitrate']/1000000:.1f} Mbps)")
    print(f"  File Size: {metadata['file_size']:,} bytes ({metadata['file_size']/1024/1024:.1f} MB)")
    print(f"  Audio Tracks: {metadata['audio_tracks']}")
    
    # Check if 4K
    is_4k = metadata['resolution'][0] >= 3840 and metadata['resolution'][1] >= 2160
    print(f"  4K Status: {'Already 4K' if is_4k else 'Needs upscaling'}")
    
    if theater_analysis:
        print(f"\nTHEATER CHARACTERISTICS:")
        print(f"  Lighting Type: {theater_analysis['lighting_type'].title()}")
        print(f"  Scene Complexity: {theater_analysis['scene_complexity'].title()}")
        print(f"  Average Brightness: {theater_analysis['avg_brightness']:.1f}")
        print(f"  Complexity Score: {theater_analysis['avg_complexity']:.3f}")
        print(f"  Frames Analyzed: {theater_analysis['frames_analyzed']}")
        print(f"  Recommended Model: {theater_analysis['recommended_model']}")
    
    print("\nANALYSIS SUMMARY:")
    print("-" * 30)
    
    if not is_4k:
        scale_factor = 3840 / metadata['resolution'][0]
        print(f"→ Video needs {scale_factor:.1f}x upscaling to reach 4K")
    else:
        print("✓ Video is already 4K resolution")
    
    if theater_analysis:
        print(f"→ Recommended AI model: {theater_analysis['recommended_model']}")
        
        if theater_analysis['lighting_type'] == "stage":
            print("→ Stage lighting detected - use high-quality enhancement")
        if theater_analysis['scene_complexity'] == "complex":
            print("→ Complex scenes detected - prioritize quality over speed")
    
    print("\n✓ Analysis completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)