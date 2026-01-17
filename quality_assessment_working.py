#!/usr/bin/env python3
"""
Quality Assessment Task 5.1: Perform quality comparison analysis
"""

import cv2
import numpy as np
import json
import os
import subprocess
from datetime import datetime

def get_video_info(video_path):
    """Extract video information using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error getting video info for {video_path}: {e}")
        return None

def extract_sample_frames(video_path, num_frames=5):
    """Extract sample frames for quality comparison"""
    print(f"Extracting frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Total frames: {total_frames}")
    
    if total_frames == 0:
        print("  No frames found in video")
        cap.release()
        return []
    
    frame_indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            print(f"  Extracted frame {idx}: {frame.shape}")
        else:
            print(f"  Failed to extract frame {idx}")
    
    cap.release()
    print(f"  Successfully extracted {len(frames)} frames")
    return frames

def calculate_basic_metrics(original_frames, enhanced_frames):
    """Calculate basic quality metrics"""
    if not original_frames or not enhanced_frames:
        return None
    
    print("Calculating quality metrics...")
    
    # Use minimum number of frames
    min_frames = min(len(original_frames), len(enhanced_frames))
    print(f"  Comparing {min_frames} frame pairs")
    
    psnr_values = []
    ssim_values = []
    
    try:
        from skimage.metrics import structural_similarity as ssim_func
        from skimage.metrics import peak_signal_noise_ratio as psnr_func
        
        for i in range(min_frames):
            orig_frame = original_frames[i]
            enh_frame = enhanced_frames[i]
            
            # Convert to grayscale
            orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
            enh_gray = cv2.cvtColor(enh_frame, cv2.COLOR_BGR2GRAY)
            
            # Resize enhanced frame to match original for comparison
            if orig_gray.shape != enh_gray.shape:
                enh_gray = cv2.resize(enh_gray, (orig_gray.shape[1], orig_gray.shape[0]))
            
            # Calculate PSNR
            psnr_val = psnr_func(orig_gray, enh_gray, data_range=255)
            psnr_values.append(psnr_val)
            
            # Calculate SSIM
            ssim_val = ssim_func(orig_gray, enh_gray, data_range=255)
            ssim_values.append(ssim_val)
            
            print(f"  Frame {i}: PSNR={psnr_val:.2f}dB, SSIM={ssim_val:.3f}")
        
        return {
            'psnr_values': psnr_values,
            'ssim_values': ssim_values,
            'psnr_mean': np.mean(psnr_values),
            'psnr_std': np.std(psnr_values),
            'ssim_mean': np.mean(ssim_values),
            'ssim_std': np.std(ssim_values)
        }
    
    except ImportError as e:
        print(f"  Error importing scikit-image: {e}")
        return None

def analyze_file_sizes(original_path, enhanced_path):
    """Analyze file sizes and basic video properties"""
    print("Analyzing file properties...")
    
    # Get file sizes
    orig_size = os.path.getsize(original_path)
    enh_size = os.path.getsize(enhanced_path)
    
    analysis = {
        'original_size_mb': orig_size / (1024 * 1024),
        'enhanced_size_mb': enh_size / (1024 * 1024),
        'size_ratio': enh_size / orig_size if orig_size > 0 else 0,
        'size_increase_mb': (enh_size - orig_size) / (1024 * 1024)
    }
    
    print(f"  Original: {analysis['original_size_mb']:.1f} MB")
    print(f"  Enhanced: {analysis['enhanced_size_mb']:.1f} MB")
    print(f"  Size ratio: {analysis['size_ratio']:.2f}x")
    
    # Get video information
    orig_info = get_video_info(original_path)
    enh_info = get_video_info(enhanced_path)
    
    if orig_info and enh_info:
        orig_video = next((s for s in orig_info['streams'] if s['codec_type'] == 'video'), None)
        enh_video = next((s for s in enh_info['streams'] if s['codec_type'] == 'video'), None)
        
        if orig_video and enh_video:
            analysis['original_resolution'] = f"{orig_video['width']}x{orig_video['height']}"
            analysis['enhanced_resolution'] = f"{enh_video['width']}x{enh_video['height']}"
            
            print(f"  Original resolution: {analysis['original_resolution']}")
            print(f"  Enhanced resolution: {analysis['enhanced_resolution']}")
            
            # Calculate pixel ratio
            orig_pixels = orig_video['width'] * orig_video['height']
            enh_pixels = enh_video['width'] * enh_video['height']
            analysis['pixel_ratio'] = enh_pixels / orig_pixels if orig_pixels > 0 else 0
            
            print(f"  Pixel ratio: {analysis['pixel_ratio']:.2f}x")
    
    return analysis

def main():
    """Main function"""
    print("="*60)
    print("QUALITY ASSESSMENT TASK 5.1")
    print("="*60)
    
    # Define video paths
    original_video = "videos/大学生原创话剧《自杀既遂》.mp4"
    enhanced_video = "enhanced_4k_theater_video.mp4"
    
    # Check if files exist
    if not os.path.exists(original_video):
        print(f"Error: Original video not found at {original_video}")
        return
    
    if not os.path.exists(enhanced_video):
        print(f"Error: Enhanced video not found at {enhanced_video}")
        return
    
    # Initialize results
    results = {
        'timestamp': datetime.now().isoformat(),
        'original_video': original_video,
        'enhanced_video': enhanced_video,
        'file_analysis': {},
        'quality_metrics': {},
        'recommendations': []
    }
    
    # Analyze file sizes and properties
    results['file_analysis'] = analyze_file_sizes(original_video, enhanced_video)
    
    # Extract sample frames
    original_frames = extract_sample_frames(original_video, num_frames=5)
    enhanced_frames = extract_sample_frames(enhanced_video, num_frames=5)
    
    # Calculate quality metrics if frames were extracted
    if original_frames and enhanced_frames:
        metrics = calculate_basic_metrics(original_frames, enhanced_frames)
        if metrics:
            results['quality_metrics'] = metrics
    
    # Generate recommendations
    recommendations = []
    
    if 'psnr_mean' in results['quality_metrics']:
        psnr_mean = results['quality_metrics']['psnr_mean']
        if psnr_mean < 25:
            recommendations.append("PSNR is below 25dB - quality may be degraded")
        elif psnr_mean > 35:
            recommendations.append("Excellent PSNR results - high quality preservation")
        else:
            recommendations.append("Good PSNR results - acceptable quality")
    
    if 'ssim_mean' in results['quality_metrics']:
        ssim_mean = results['quality_metrics']['ssim_mean']
        if ssim_mean < 0.8:
            recommendations.append("SSIM below 0.8 - structural similarity could be improved")
        elif ssim_mean > 0.9:
            recommendations.append("Excellent SSIM results - structural details well preserved")
        else:
            recommendations.append("Good SSIM results - reasonable structural preservation")
    
    if 'size_ratio' in results['file_analysis']:
        size_ratio = results['file_analysis']['size_ratio']
        if size_ratio > 10:
            recommendations.append("File size increased significantly - consider compression")
        elif size_ratio < 0.1:
            recommendations.append("File size decreased significantly - may indicate quality loss")
        else:
            recommendations.append("Reasonable file size change")
    
    results['recommendations'] = recommendations
    
    # Print summary
    print("\n" + "="*60)
    print("QUALITY ASSESSMENT SUMMARY")
    print("="*60)
    
    if 'quality_metrics' in results and results['quality_metrics']:
        metrics = results['quality_metrics']
        print(f"PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
        print(f"SSIM: {metrics['ssim_mean']:.3f} ± {metrics['ssim_std']:.3f}")
    
    if recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    # Save results
    output_file = 'quality_assessment_task_5_1_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")
    print("Quality assessment completed!")
    print("="*60)

if __name__ == "__main__":
    main()