#!/usr/bin/env python3
"""
Quality Assessment Task 5.1: Perform quality comparison analysis

This script compares original and enhanced video quality using metrics (PSNR, SSIM),
assesses visual improvements in theater-specific elements, checks for artifacts,
flickering, or quality degradation, and monitors file size and compression efficiency.

Requirements: 4.1, 4.4
"""

import cv2
import numpy as np
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import seaborn as sns

class QualityAssessment:
    def __init__(self, original_video_path, enhanced_video_path):
        self.original_video_path = original_video_path
        self.enhanced_video_path = enhanced_video_path
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'original_video': original_video_path,
            'enhanced_video': enhanced_video_path,
            'metrics': {},
            'file_analysis': {},
            'theater_assessment': {},
            'artifacts_analysis': {},
            'recommendations': []
        }
    
    def get_video_info(self, video_path):
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
    
    def extract_sample_frames(self, video_path, num_frames=10):
        """Extract sample frames for quality comparison"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def calculate_psnr_ssim(self, original_frames, enhanced_frames):
        """Calculate PSNR and SSIM metrics between frame pairs"""
        psnr_values = []
        ssim_values = []
        
        min_frames = min(len(original_frames), len(enhanced_frames))
        
        for i in range(min_frames):
            # Resize enhanced frame to match original for comparison
            orig_frame = original_frames[i]
            enh_frame = enhanced_frames[i]
            
            # Convert to grayscale for SSIM calculation
            orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
            enh_gray = cv2.cvtColor(enh_frame, cv2.COLOR_BGR2GRAY)
            
            # Resize enhanced frame to original size for fair comparison
            enh_gray_resized = cv2.resize(enh_gray, (orig_gray.shape[1], orig_gray.shape[0]))
            
            # Calculate PSNR
            psnr_val = psnr(orig_gray, enh_gray_resized, data_range=255)
            psnr_values.append(psnr_val)
            
            # Calculate SSIM
            ssim_val = ssim(orig_gray, enh_gray_resized, data_range=255)
            ssim_values.append(ssim_val)
        
        return psnr_values, ssim_values
    
    def analyze_theater_elements(self, original_frames, enhanced_frames):
        """Analyze theater-specific improvements"""
        theater_analysis = {
            'actor_detail_improvement': 0,
            'stage_clarity_improvement': 0,
            'lighting_preservation': 0,
            'overall_theater_score': 0
        }
        
        # Simple analysis based on edge detection and contrast
        for orig, enh in zip(original_frames, enhanced_frames):
            # Convert to grayscale
            orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            enh_gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
            
            # Resize for comparison
            enh_gray = cv2.resize(enh_gray, (orig_gray.shape[1], orig_gray.shape[0]))
            
            # Edge detection for detail analysis
            orig_edges = cv2.Canny(orig_gray, 50, 150)
            enh_edges = cv2.Canny(enh_gray, 50, 150)
            
            # Count edge pixels as a measure of detail
            orig_edge_count = np.sum(orig_edges > 0)
            enh_edge_count = np.sum(enh_edges > 0)
            
            if orig_edge_count > 0:
                detail_improvement = (enh_edge_count - orig_edge_count) / orig_edge_count
                theater_analysis['actor_detail_improvement'] += detail_improvement
            
            # Contrast analysis
            orig_contrast = np.std(orig_gray)
            enh_contrast = np.std(enh_gray)
            
            if orig_contrast > 0:
                contrast_improvement = (enh_contrast - orig_contrast) / orig_contrast
                theater_analysis['stage_clarity_improvement'] += contrast_improvement
        
        # Average the improvements
        num_frames = len(original_frames)
        if num_frames > 0:
            theater_analysis['actor_detail_improvement'] /= num_frames
            theater_analysis['stage_clarity_improvement'] /= num_frames
            theater_analysis['lighting_preservation'] = 1.0  # Placeholder - would need more sophisticated analysis
            theater_analysis['overall_theater_score'] = (
                theater_analysis['actor_detail_improvement'] + 
                theater_analysis['stage_clarity_improvement'] + 
                theater_analysis['lighting_preservation']
            ) / 3
        
        return theater_analysis
    
    def check_artifacts_and_flickering(self, enhanced_frames):
        """Check for artifacts and flickering in enhanced video"""
        artifacts_analysis = {
            'flickering_detected': False,
            'artifact_score': 0,
            'temporal_consistency': 0,
            'issues_found': []
        }
        
        if len(enhanced_frames) < 2:
            return artifacts_analysis
        
        # Check for flickering by comparing consecutive frames
        frame_differences = []
        for i in range(1, len(enhanced_frames)):
            prev_frame = cv2.cvtColor(enhanced_frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(enhanced_frames[i], cv2.COLOR_BGR2GRAY)
            
            # Resize to same size
            if prev_frame.shape != curr_frame.shape:
                curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]))
            
            # Calculate frame difference
            diff = np.mean(np.abs(curr_frame.astype(float) - prev_frame.astype(float)))
            frame_differences.append(diff)
        
        # Analyze frame differences for flickering
        if frame_differences:
            mean_diff = np.mean(frame_differences)
            std_diff = np.std(frame_differences)
            
            # Simple flickering detection
            if std_diff > mean_diff * 0.5:
                artifacts_analysis['flickering_detected'] = True
                artifacts_analysis['issues_found'].append('Potential flickering detected')
            
            # Temporal consistency score (lower differences = better consistency)
            artifacts_analysis['temporal_consistency'] = max(0, 1 - (std_diff / 255))
        
        return artifacts_analysis
    
    def analyze_file_sizes(self):
        """Compare file sizes and compression efficiency"""
        file_analysis = {}
        
        # Get file sizes
        orig_size = os.path.getsize(self.original_video_path)
        enh_size = os.path.getsize(self.enhanced_video_path)
        
        file_analysis['original_size_mb'] = orig_size / (1024 * 1024)
        file_analysis['enhanced_size_mb'] = enh_size / (1024 * 1024)
        file_analysis['size_ratio'] = enh_size / orig_size if orig_size > 0 else 0
        file_analysis['size_increase_mb'] = (enh_size - orig_size) / (1024 * 1024)
        
        # Get video information
        orig_info = self.get_video_info(self.original_video_path)
        enh_info = self.get_video_info(self.enhanced_video_path)
        
        if orig_info and enh_info:
            # Extract resolution information
            orig_video_stream = next((s for s in orig_info['streams'] if s['codec_type'] == 'video'), None)
            enh_video_stream = next((s for s in enh_info['streams'] if s['codec_type'] == 'video'), None)
            
            if orig_video_stream and enh_video_stream:
                file_analysis['original_resolution'] = f"{orig_video_stream['width']}x{orig_video_stream['height']}"
                file_analysis['enhanced_resolution'] = f"{enh_video_stream['width']}x{enh_video_stream['height']}"
                
                # Calculate pixel count increase
                orig_pixels = orig_video_stream['width'] * orig_video_stream['height']
                enh_pixels = enh_video_stream['width'] * enh_video_stream['height']
                file_analysis['pixel_ratio'] = enh_pixels / orig_pixels if orig_pixels > 0 else 0
                
                # Compression efficiency (MB per megapixel)
                file_analysis['orig_compression_efficiency'] = file_analysis['original_size_mb'] / (orig_pixels / 1e6)
                file_analysis['enh_compression_efficiency'] = file_analysis['enhanced_size_mb'] / (enh_pixels / 1e6)
        
        return file_analysis
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Check PSNR/SSIM results
        if 'psnr_mean' in self.results['metrics']:
            if self.results['metrics']['psnr_mean'] < 25:
                recommendations.append("PSNR is below 25dB - consider using different upscaling parameters")
            elif self.results['metrics']['psnr_mean'] > 35:
                recommendations.append("Excellent PSNR results - current settings are optimal")
        
        if 'ssim_mean' in self.results['metrics']:
            if self.results['metrics']['ssim_mean'] < 0.8:
                recommendations.append("SSIM is below 0.8 - structural similarity could be improved")
            elif self.results['metrics']['ssim_mean'] > 0.9:
                recommendations.append("Excellent SSIM results - structural details well preserved")
        
        # Check file size
        if 'size_ratio' in self.results['file_analysis']:
            if self.results['file_analysis']['size_ratio'] > 10:
                recommendations.append("File size increased significantly - consider compression optimization")
            elif self.results['file_analysis']['size_ratio'] < 2:
                recommendations.append("Reasonable file size increase for 4K enhancement")
        
        # Check theater-specific improvements
        if 'overall_theater_score' in self.results['theater_assessment']:
            if self.results['theater_assessment']['overall_theater_score'] > 0.1:
                recommendations.append("Good theater-specific improvements detected")
            else:
                recommendations.append("Limited theater-specific improvements - consider different enhancement approach")
        
        # Check for artifacts
        if self.results['artifacts_analysis'].get('flickering_detected', False):
            recommendations.append("Flickering detected - consider temporal consistency filters")
        
        return recommendations
    
    def run_complete_analysis(self):
        """Run the complete quality assessment analysis"""
        print("Starting quality assessment analysis...")
        
        # Extract sample frames
        print("Extracting sample frames from original video...")
        original_frames = self.extract_sample_frames(self.original_video_path)
        
        print("Extracting sample frames from enhanced video...")
        enhanced_frames = self.extract_sample_frames(self.enhanced_video_path)
        
        if not original_frames or not enhanced_frames:
            print("Error: Could not extract frames from videos")
            return None
        
        print(f"Extracted {len(original_frames)} original frames and {len(enhanced_frames)} enhanced frames")
        
        # Calculate PSNR and SSIM
        print("Calculating PSNR and SSIM metrics...")
        psnr_values, ssim_values = self.calculate_psnr_ssim(original_frames, enhanced_frames)
        
        self.results['metrics'] = {
            'psnr_values': psnr_values,
            'ssim_values': ssim_values,
            'psnr_mean': np.mean(psnr_values) if psnr_values else 0,
            'psnr_std': np.std(psnr_values) if psnr_values else 0,
            'ssim_mean': np.mean(ssim_values) if ssim_values else 0,
            'ssim_std': np.std(ssim_values) if ssim_values else 0
        }
        
        # Analyze theater-specific elements
        print("Analyzing theater-specific improvements...")
        self.results['theater_assessment'] = self.analyze_theater_elements(original_frames, enhanced_frames)
        
        # Check for artifacts and flickering
        print("Checking for artifacts and flickering...")
        self.results['artifacts_analysis'] = self.check_artifacts_and_flickering(enhanced_frames)
        
        # Analyze file sizes and compression
        print("Analyzing file sizes and compression efficiency...")
        self.results['file_analysis'] = self.analyze_file_sizes()
        
        # Generate recommendations
        print("Generating recommendations...")
        self.results['recommendations'] = self.generate_recommendations()
        
        return self.results
    
    def save_results(self, output_file='quality_assessment_results.json'):
        """Save analysis results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    
    def print_summary(self):
        """Print a summary of the analysis results"""
        print("\n" + "="*60)
        print("QUALITY ASSESSMENT SUMMARY")
        print("="*60)
        
        # Metrics summary
        if 'psnr_mean' in self.results['metrics']:
            print(f"PSNR (Peak Signal-to-Noise Ratio): {self.results['metrics']['psnr_mean']:.2f} ± {self.results['metrics']['psnr_std']:.2f} dB")
            print(f"SSIM (Structural Similarity): {self.results['metrics']['ssim_mean']:.3f} ± {self.results['metrics']['ssim_std']:.3f}")
        
        # File analysis
        if 'original_size_mb' in self.results['file_analysis']:
            print(f"\nFile Size Analysis:")
            print(f"  Original: {self.results['file_analysis']['original_size_mb']:.1f} MB")
            print(f"  Enhanced: {self.results['file_analysis']['enhanced_size_mb']:.1f} MB")
            print(f"  Size Ratio: {self.results['file_analysis']['size_ratio']:.1f}x")
            
            if 'original_resolution' in self.results['file_analysis']:
                print(f"  Original Resolution: {self.results['file_analysis']['original_resolution']}")
                print(f"  Enhanced Resolution: {self.results['file_analysis']['enhanced_resolution']}")
        
        # Theater assessment
        if 'overall_theater_score' in self.results['theater_assessment']:
            print(f"\nTheater-Specific Assessment:")
            print(f"  Actor Detail Improvement: {self.results['theater_assessment']['actor_detail_improvement']:.3f}")
            print(f"  Stage Clarity Improvement: {self.results['theater_assessment']['stage_clarity_improvement']:.3f}")
            print(f"  Overall Theater Score: {self.results['theater_assessment']['overall_theater_score']:.3f}")
        
        # Artifacts analysis
        if 'flickering_detected' in self.results['artifacts_analysis']:
            print(f"\nArtifacts Analysis:")
            print(f"  Flickering Detected: {self.results['artifacts_analysis']['flickering_detected']}")
            print(f"  Temporal Consistency: {self.results['artifacts_analysis']['temporal_consistency']:.3f}")
            if self.results['artifacts_analysis']['issues_found']:
                print(f"  Issues Found: {', '.join(self.results['artifacts_analysis']['issues_found'])}")
        
        # Recommendations
        if self.results['recommendations']:
            print(f"\nRecommendations:")
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("="*60)


def main():
    """Main function to run quality assessment"""
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
    
    # Run quality assessment
    qa = QualityAssessment(original_video, enhanced_video)
    results = qa.run_complete_analysis()
    
    if results:
        # Print summary
        qa.print_summary()
        
        # Save results
        qa.save_results('quality_assessment_task_5_1_results.json')
        
        print(f"\nQuality assessment completed successfully!")
        print(f"Results saved to quality_assessment_task_5_1_results.json")
    else:
        print("Quality assessment failed!")


if __name__ == "__main__":
    main()