#!/usr/bin/env python3
"""
Enhanced video analysis script for the theater drama video.
Provides comprehensive analysis including detailed audio track information,
noise assessment, compression artifacts detection, and theater-specific characteristics.
"""

import os
import subprocess
import json
import cv2
import numpy as np
from pathlib import Path


def analyze_video_with_ffprobe(video_path):
    """Extract comprehensive video metadata using FFprobe."""
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
        
        # Parse detailed audio information
        audio_tracks = []
        for i, audio_stream in enumerate(audio_streams):
            audio_info = {
                'track_number': i + 1,
                'codec': audio_stream.get('codec_name', 'unknown'),
                'codec_long_name': audio_stream.get('codec_long_name', 'Unknown'),
                'bitrate': int(audio_stream.get('bit_rate', 0)),
                'channels': int(audio_stream.get('channels', 2)),
                'channel_layout': audio_stream.get('channel_layout', 'stereo'),
                'sample_rate': int(audio_stream.get('sample_rate', 44100)),
                'sample_fmt': audio_stream.get('sample_fmt', 'unknown'),
                'language': audio_stream.get('tags', {}).get('language', 'unknown')
            }
            audio_tracks.append(audio_info)
        
        # Additional video stream details
        video_details = {
            'pixel_format': video_stream.get('pix_fmt', 'unknown'),
            'color_range': video_stream.get('color_range', 'unknown'),
            'color_space': video_stream.get('color_space', 'unknown'),
            'profile': video_stream.get('profile', 'unknown'),
            'level': video_stream.get('level', 'unknown')
        }
        
        return {
            'resolution': (width, height),
            'frame_rate': frame_rate,
            'duration': duration,
            'codec': video_stream.get('codec_name', 'unknown'),
            'codec_long_name': video_stream.get('codec_long_name', 'Unknown'),
            'bitrate': bitrate,
            'audio_tracks': audio_tracks,
            'video_details': video_details,
            'file_size': os.path.getsize(video_path),
            'format_name': probe_data.get('format', {}).get('format_name', 'unknown'),
            'format_long_name': probe_data.get('format', {}).get('format_long_name', 'Unknown')
        }
        
    except subprocess.CalledProcessError as e:
        print(f"FFprobe failed: {e.stderr}")
        return None
    except Exception as e:
        print(f"Failed to parse FFprobe output: {str(e)}")
        return None


def analyze_video_quality_and_artifacts(video_path):
    """Analyze video quality, noise levels, and compression artifacts."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Could not open video file for quality analysis")
            return None
        
        # Sample frames for analysis (every 60 seconds for more comprehensive analysis)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = int(frame_rate * 60)  # Every 60 seconds
        
        frames_analyzed = 0
        noise_levels = []
        sharpness_scores = []
        compression_artifacts = []
        brightness_levels = []
        contrast_levels = []
        
        for frame_num in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frames_analyzed += 1
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Noise level estimation using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            noise_levels.append(laplacian_var)
            
            # Sharpness estimation using gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sharpness = np.mean(np.sqrt(grad_x**2 + grad_y**2))
            sharpness_scores.append(sharpness)
            
            # Compression artifact detection using block analysis
            artifact_score = detect_compression_artifacts(gray)
            compression_artifacts.append(artifact_score)
            
            # Brightness and contrast analysis
            brightness = np.mean(gray)
            contrast = np.std(gray)
            brightness_levels.append(brightness)
            contrast_levels.append(contrast)
            
            # Limit analysis to prevent excessive processing
            if frames_analyzed >= 15:
                break
        
        cap.release()
        
        if frames_analyzed == 0:
            return None
        
        # Calculate quality metrics
        avg_noise = np.mean(noise_levels)
        avg_sharpness = np.mean(sharpness_scores)
        avg_artifacts = np.mean(compression_artifacts)
        avg_brightness = np.mean(brightness_levels)
        avg_contrast = np.mean(contrast_levels)
        
        # Quality assessment
        noise_level = "low" if avg_noise > 500 else "moderate" if avg_noise > 200 else "high"
        sharpness_level = "high" if avg_sharpness > 30 else "moderate" if avg_sharpness > 15 else "low"
        artifact_level = "low" if avg_artifacts < 0.1 else "moderate" if avg_artifacts < 0.2 else "high"
        
        return {
            'noise_level': noise_level,
            'avg_noise_score': avg_noise,
            'sharpness_level': sharpness_level,
            'avg_sharpness_score': avg_sharpness,
            'compression_artifacts': artifact_level,
            'avg_artifact_score': avg_artifacts,
            'avg_brightness': avg_brightness,
            'avg_contrast': avg_contrast,
            'frames_analyzed': frames_analyzed,
            'quality_summary': get_quality_summary(noise_level, sharpness_level, artifact_level)
        }
        
    except Exception as e:
        print(f"Quality analysis failed: {str(e)}")
        return None


def detect_compression_artifacts(gray_frame):
    """Detect compression artifacts using block-based analysis."""
    height, width = gray_frame.shape
    block_size = 8  # Standard DCT block size
    artifact_score = 0
    block_count = 0
    
    # Analyze 8x8 blocks for compression artifacts
    for y in range(0, height - block_size, block_size):
        for x in range(0, width - block_size, block_size):
            block = gray_frame[y:y+block_size, x:x+block_size]
            
            # Calculate block variance (low variance indicates compression)
            block_var = np.var(block)
            
            # Check for blocking artifacts at block boundaries
            if x + block_size < width:
                right_block = gray_frame[y:y+block_size, x+block_size:x+2*block_size]
                boundary_diff = np.mean(np.abs(block[:, -1] - right_block[:, 0]))
                artifact_score += boundary_diff / 255.0
            
            if y + block_size < height:
                bottom_block = gray_frame[y+block_size:y+2*block_size, x:x+block_size]
                boundary_diff = np.mean(np.abs(block[-1, :] - bottom_block[0, :]))
                artifact_score += boundary_diff / 255.0
            
            block_count += 1
            
            # Limit analysis for performance
            if block_count >= 100:
                break
        if block_count >= 100:
            break
    
    return artifact_score / max(block_count, 1)


def analyze_theater_characteristics(video_path):
    """Analyze theater-specific characteristics with enhanced detail."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Could not open video file for theater analysis")
            return None
        
        # Sample frames for analysis (every 45 seconds for theater analysis)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = int(frame_rate * 45)  # Every 45 seconds
        
        frames_analyzed = 0
        lighting_scores = {'stage': 0, 'natural': 0, 'mixed': 0}
        complexity_scores = {'simple': 0, 'moderate': 0, 'complex': 0}
        stage_setup_scores = {'proscenium': 0, 'thrust': 0, 'arena': 0}
        actor_visibility_scores = []
        scene_depth_scores = []
        
        for frame_num in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frames_analyzed += 1
            
            # Analyze lighting characteristics
            lighting_type = analyze_frame_lighting(frame)
            lighting_scores[lighting_type] += 1
            
            # Analyze scene complexity
            complexity = analyze_frame_complexity(frame)
            complexity_scores[complexity] += 1
            
            # Analyze stage setup
            stage_setup = analyze_stage_setup(frame)
            stage_setup_scores[stage_setup] += 1
            
            # Analyze actor visibility
            visibility_score = analyze_actor_visibility(frame)
            actor_visibility_scores.append(visibility_score)
            
            # Analyze scene depth
            depth_score = analyze_scene_depth(frame)
            scene_depth_scores.append(depth_score)
            
            # Limit analysis to prevent excessive processing
            if frames_analyzed >= 12:
                break
        
        cap.release()
        
        if frames_analyzed == 0:
            return None
        
        # Determine dominant characteristics
        lighting_type = max(lighting_scores, key=lighting_scores.get)
        scene_complexity = max(complexity_scores, key=complexity_scores.get)
        stage_setup = max(stage_setup_scores, key=stage_setup_scores.get)
        
        # Calculate averages
        avg_visibility = np.mean(actor_visibility_scores)
        avg_depth = np.mean(scene_depth_scores)
        
        # Estimate actor count based on complexity and visibility
        actor_count = estimate_actor_count(complexity_scores, avg_visibility, frames_analyzed)
        
        return {
            'lighting_type': lighting_type,
            'lighting_distribution': lighting_scores,
            'scene_complexity': scene_complexity,
            'complexity_distribution': complexity_scores,
            'stage_setup': stage_setup,
            'stage_setup_distribution': stage_setup_scores,
            'estimated_actor_count': actor_count,
            'avg_actor_visibility': avg_visibility,
            'avg_scene_depth': avg_depth,
            'frames_analyzed': frames_analyzed,
            'recommended_model': get_recommended_model(lighting_type, scene_complexity),
            'enhancement_priority': get_enhancement_priority(scene_complexity, avg_visibility)
        }
        
    except Exception as e:
        print(f"Theater analysis failed: {str(e)}")
        return None


def analyze_frame_lighting(frame):
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
    
    # Analyze lighting uniformity
    brightness_hist = cv2.calcHist([brightness], [0], None, [256], [0, 256])
    hist_peaks = len([i for i in range(1, 255) if brightness_hist[i] > brightness_hist[i-1] and brightness_hist[i] > brightness_hist[i+1]])
    
    # Enhanced heuristic classification
    if brightness_std > 60 and color_temp_ratio > 1.2 and hist_peaks <= 3:
        return "stage"  # High contrast, warm lighting, few peaks typical of stage
    elif mean_brightness > 120 and brightness_std < 40 and hist_peaks <= 2:
        return "natural"  # Even, bright lighting
    else:
        return "mixed"  # Mixed or complex lighting


def analyze_frame_complexity(frame):
    """Analyze visual complexity of a frame."""
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Calculate texture complexity using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate color complexity
    color_std = np.std(frame, axis=(0, 1))
    color_complexity = np.mean(color_std)
    
    # Enhanced heuristic classification
    if edge_density > 0.15 or laplacian_var > 1000 or color_complexity > 40:
        return "complex"
    elif edge_density > 0.08 or laplacian_var > 500 or color_complexity > 25:
        return "moderate"
    else:
        return "simple"


def analyze_stage_setup(frame):
    """Analyze stage setup type based on visual cues."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Analyze horizontal and vertical line distributions
    horizontal_lines = cv2.HoughLines(cv2.Canny(gray, 50, 150), 1, np.pi/180, threshold=100)
    vertical_lines = cv2.HoughLines(cv2.Canny(gray, 50, 150), 1, np.pi/180, threshold=100)
    
    # Analyze frame composition
    center_activity = np.mean(gray[height//3:2*height//3, width//3:2*width//3])
    edge_activity = np.mean(np.concatenate([
        gray[:height//3, :].flatten(),
        gray[2*height//3:, :].flatten(),
        gray[:, :width//3].flatten(),
        gray[:, 2*width//3:].flatten()
    ]))
    
    composition_ratio = center_activity / (edge_activity + 1)
    
    # Heuristic classification
    if composition_ratio > 1.3:
        return "proscenium"  # Activity concentrated in center
    elif composition_ratio > 0.8:
        return "thrust"  # Moderate center focus
    else:
        return "arena"  # Activity distributed


def analyze_actor_visibility(frame):
    """Analyze actor visibility and detail level."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use face detection as a proxy for actor visibility
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Calculate visibility score based on face detection and image sharpness
    face_count = len(faces)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Combine face detection with overall image quality
    visibility_score = min(face_count * 0.3 + sharpness / 1000, 1.0)
    
    return visibility_score


def analyze_scene_depth(frame):
    """Analyze perceived depth of the scene."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Analyze depth cues using gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Analyze vertical distribution of features (depth perception)
    height = gray.shape[0]
    top_third = gradient_magnitude[:height//3, :]
    middle_third = gradient_magnitude[height//3:2*height//3, :]
    bottom_third = gradient_magnitude[2*height//3:, :]
    
    top_activity = np.mean(top_third)
    middle_activity = np.mean(middle_third)
    bottom_activity = np.mean(bottom_third)
    
    # Calculate depth score
    depth_score = (top_activity + middle_activity) / (bottom_activity + 1)
    
    return min(depth_score / 2.0, 1.0)  # Normalize to 0-1


def estimate_actor_count(complexity_scores, avg_visibility, frames_analyzed):
    """Estimate number of actors based on scene complexity and visibility."""
    complex_ratio = complexity_scores['complex'] / frames_analyzed
    moderate_ratio = complexity_scores['moderate'] / frames_analyzed
    
    # Base estimate on complexity
    if complex_ratio > 0.6:
        base_count = 5
    elif complex_ratio > 0.3 or moderate_ratio > 0.7:
        base_count = 3
    else:
        base_count = 2
    
    # Adjust based on visibility score
    if avg_visibility > 0.7:
        base_count += 1
    elif avg_visibility < 0.3:
        base_count = max(1, base_count - 1)
    
    return base_count


def get_recommended_model(lighting_type, scene_complexity):
    """Get recommended AI model based on content characteristics."""
    if lighting_type == "stage" and scene_complexity == "complex":
        return "realesrgan-x4plus"  # Best for complex stage lighting
    elif lighting_type == "natural":
        return "real-cugan"  # Good for natural lighting
    elif scene_complexity == "simple":
        return "waifu2x"  # Efficient for simple content
    else:
        return "realesrgan-x4plus"  # Default robust choice


def get_enhancement_priority(scene_complexity, avg_visibility):
    """Determine enhancement priority based on content characteristics."""
    if scene_complexity == "complex" or avg_visibility < 0.4:
        return "quality"  # Complex content or poor visibility needs quality focus
    elif scene_complexity == "simple" and avg_visibility > 0.7:
        return "speed"  # Simple, clear content can prioritize speed
    else:
        return "balanced"  # Balanced approach for moderate content


def get_quality_summary(noise_level, sharpness_level, artifact_level):
    """Generate quality summary based on analysis."""
    issues = []
    if noise_level == "high":
        issues.append("high noise levels")
    if sharpness_level == "low":
        issues.append("low sharpness")
    if artifact_level == "high":
        issues.append("compression artifacts")
    
    if not issues:
        return "Good quality - suitable for enhancement"
    elif len(issues) == 1:
        return f"Moderate quality - {issues[0]} detected"
    else:
        return f"Quality issues detected - {', '.join(issues)}"


def main():
    """Main function to perform comprehensive theater video analysis."""
    video_path = "videos/大学生原创话剧《自杀既遂》.mp4"
    
    print("Enhanced Theater Drama Video Analysis")
    print("=" * 60)
    print(f"Analyzing: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False
    
    # Get comprehensive metadata
    print("\nExtracting comprehensive video metadata...")
    metadata = analyze_video_with_ffprobe(video_path)
    
    if not metadata:
        print("Failed to extract video metadata")
        return False
    
    # Analyze video quality and artifacts
    print("Analyzing video quality and compression artifacts...")
    quality_analysis = analyze_video_quality_and_artifacts(video_path)
    
    # Analyze theater characteristics
    print("Analyzing theater-specific characteristics...")
    theater_analysis = analyze_theater_characteristics(video_path)
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("COMPREHENSIVE VIDEO ANALYSIS REPORT")
    print("="*80)
    
    # Basic specifications
    print(f"\nBASIC SPECIFICATIONS:")
    print(f"  Resolution: {metadata['resolution'][0]}x{metadata['resolution'][1]}")
    print(f"  Frame Rate: {metadata['frame_rate']:.2f} fps")
    print(f"  Duration: {metadata['duration']:.2f} seconds ({metadata['duration']/60:.1f} minutes)")
    print(f"  Video Codec: {metadata['codec']} ({metadata['codec_long_name']})")
    print(f"  Bitrate: {metadata['bitrate']:,} bps ({metadata['bitrate']/1000000:.1f} Mbps)")
    print(f"  File Size: {metadata['file_size']:,} bytes ({metadata['file_size']/1024/1024:.1f} MB)")
    print(f"  Container Format: {metadata['format_name']} ({metadata['format_long_name']})")
    
    # Video details
    print(f"\nVIDEO ENCODING DETAILS:")
    vd = metadata['video_details']
    print(f"  Pixel Format: {vd['pixel_format']}")
    print(f"  Color Range: {vd['color_range']}")
    print(f"  Color Space: {vd['color_space']}")
    print(f"  Profile: {vd['profile']}")
    print(f"  Level: {vd['level']}")
    
    # Check if 4K
    is_4k = metadata['resolution'][0] >= 3840 and metadata['resolution'][1] >= 2160
    print(f"  4K Status: {'Already 4K' if is_4k else 'Needs upscaling'}")
    
    # Audio tracks
    print(f"\nAUDIO TRACKS ({len(metadata['audio_tracks'])}):")
    for audio in metadata['audio_tracks']:
        print(f"  Track {audio['track_number']}: {audio['codec']} ({audio['codec_long_name']})")
        print(f"    Channels: {audio['channels']} ({audio['channel_layout']})")
        print(f"    Sample Rate: {audio['sample_rate']} Hz")
        print(f"    Sample Format: {audio['sample_fmt']}")
        print(f"    Bitrate: {audio['bitrate']:,} bps")
        print(f"    Language: {audio['language']}")
    
    # Quality analysis
    if quality_analysis:
        print(f"\nQUALITY ASSESSMENT:")
        print(f"  Noise Level: {quality_analysis['noise_level'].title()} (score: {quality_analysis['avg_noise_score']:.1f})")
        print(f"  Sharpness Level: {quality_analysis['sharpness_level'].title()} (score: {quality_analysis['avg_sharpness_score']:.1f})")
        print(f"  Compression Artifacts: {quality_analysis['compression_artifacts'].title()} (score: {quality_analysis['avg_artifact_score']:.3f})")
        print(f"  Average Brightness: {quality_analysis['avg_brightness']:.1f}")
        print(f"  Average Contrast: {quality_analysis['avg_contrast']:.1f}")
        print(f"  Quality Summary: {quality_analysis['quality_summary']}")
        print(f"  Frames Analyzed: {quality_analysis['frames_analyzed']}")
    
    # Theater characteristics
    if theater_analysis:
        print(f"\nTHEATER CHARACTERISTICS:")
        print(f"  Lighting Type: {theater_analysis['lighting_type'].title()}")
        print(f"    Distribution: {theater_analysis['lighting_distribution']}")
        print(f"  Scene Complexity: {theater_analysis['scene_complexity'].title()}")
        print(f"    Distribution: {theater_analysis['complexity_distribution']}")
        print(f"  Stage Setup: {theater_analysis['stage_setup'].title()}")
        print(f"    Distribution: {theater_analysis['stage_setup_distribution']}")
        print(f"  Estimated Actor Count: {theater_analysis['estimated_actor_count']}")
        print(f"  Average Actor Visibility: {theater_analysis['avg_actor_visibility']:.2f}")
        print(f"  Average Scene Depth: {theater_analysis['avg_scene_depth']:.2f}")
        print(f"  Frames Analyzed: {theater_analysis['frames_analyzed']}")
        print(f"  Recommended Model: {theater_analysis['recommended_model']}")
        print(f"  Enhancement Priority: {theater_analysis['enhancement_priority'].title()}")
    
    # Enhancement recommendations
    print(f"\nENHANCEMENT RECOMMENDATIONS:")
    print("-" * 40)
    
    if not is_4k:
        scale_factor = 3840 / metadata['resolution'][0]
        print(f"→ Video needs {scale_factor:.1f}x upscaling to reach 4K")
    else:
        print("✓ Video is already 4K resolution")
    
    if theater_analysis:
        print(f"→ Recommended AI model: {theater_analysis['recommended_model']}")
        print(f"→ Processing priority: {theater_analysis['enhancement_priority']}")
        
        if theater_analysis['lighting_type'] == "stage":
            print("→ Stage lighting detected - use high-quality enhancement")
        if theater_analysis['scene_complexity'] == "complex":
            print("→ Complex scenes detected - prioritize quality over speed")
        if theater_analysis['estimated_actor_count'] > 3:
            print("→ Multiple actors detected - ensure facial detail preservation")
        if theater_analysis['avg_actor_visibility'] < 0.5:
            print("→ Low actor visibility - focus on detail enhancement")
    
    if quality_analysis:
        if quality_analysis['noise_level'] == "high":
            print("→ High noise detected - apply denoising before upscaling")
        if quality_analysis['compression_artifacts'] == "high":
            print("→ Compression artifacts detected - use artifact reduction")
        if quality_analysis['sharpness_level'] == "low":
            print("→ Low sharpness - prioritize edge enhancement")
    
    print("\n✓ Comprehensive analysis completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)