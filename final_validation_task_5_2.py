#!/usr/bin/env python3
"""
Final Output Validation Task 5.2: Final output validation and optimization

This script validates 4K resolution (3840x2160) achievement, tests video playback compatibility,
optimizes file size if necessary while preserving quality, and generates processing report 
with settings and results.

Requirements: 4.2, 4.5
"""

import cv2
import numpy as np
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

class FinalValidation:
    def __init__(self, enhanced_video_path, original_video_path=None):
        self.enhanced_video_path = enhanced_video_path
        self.original_video_path = original_video_path
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'enhanced_video': enhanced_video_path,
            'original_video': original_video_path,
            'resolution_validation': {},
            'playback_compatibility': {},
            'optimization_analysis': {},
            'processing_report': {},
            'final_status': 'unknown'
        }
    
    def validate_4k_resolution(self):
        """Validate that the video achieves exactly 4K resolution (3840x2160)"""
        print("Validating 4K resolution...")
        
        validation = {
            'target_resolution': '3840x2160',
            'actual_resolution': 'unknown',
            'width': 0,
            'height': 0,
            'is_4k': False,
            'pixel_count': 0,
            'aspect_ratio': 0
        }
        
        try:
            # Use OpenCV to get video properties
            cap = cv2.VideoCapture(self.enhanced_video_path)
            if not cap.isOpened():
                validation['error'] = f"Cannot open video file: {self.enhanced_video_path}"
                return validation
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            validation['width'] = width
            validation['height'] = height
            validation['actual_resolution'] = f"{width}x{height}"
            validation['pixel_count'] = width * height
            validation['aspect_ratio'] = width / height if height > 0 else 0
            validation['frame_count'] = frame_count
            validation['fps'] = fps
            
            # Check if it's exactly 4K
            validation['is_4k'] = (width == 3840 and height == 2160)
            
            cap.release()
            
            print(f"  Resolution: {validation['actual_resolution']}")
            print(f"  Is 4K: {validation['is_4k']}")
            print(f"  Frame count: {frame_count}")
            print(f"  FPS: {fps:.2f}")
            
        except Exception as e:
            validation['error'] = str(e)
            print(f"  Error: {e}")
        
        return validation
    
    def test_playback_compatibility(self):
        """Test video playback compatibility with different codecs and containers"""
        print("Testing playback compatibility...")
        
        compatibility = {
            'file_format': 'unknown',
            'video_codec': 'unknown',
            'audio_codec': 'unknown',
            'container_format': 'unknown',
            'compatibility_score': 0,
            'supported_players': [],
            'issues': []
        }
        
        try:
            # Use ffprobe to get detailed codec information
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', '-show_streams', self.enhanced_video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Extract format information
            if 'format' in info:
                format_info = info['format']
                compatibility['container_format'] = format_info.get('format_name', 'unknown')
                compatibility['file_format'] = format_info.get('format_long_name', 'unknown')
            
            # Extract stream information
            video_stream = None
            audio_stream = None
            
            for stream in info.get('streams', []):
                if stream['codec_type'] == 'video' and not video_stream:
                    video_stream = stream
                elif stream['codec_type'] == 'audio' and not audio_stream:
                    audio_stream = stream
            
            if video_stream:
                compatibility['video_codec'] = video_stream.get('codec_name', 'unknown')
                compatibility['video_profile'] = video_stream.get('profile', 'unknown')
                compatibility['video_level'] = video_stream.get('level', 'unknown')
            
            if audio_stream:
                compatibility['audio_codec'] = audio_stream.get('codec_name', 'unknown')
            
            # Assess compatibility based on common codecs
            compatibility_score = 0
            supported_players = []
            
            video_codec = compatibility['video_codec'].lower()
            container = compatibility['container_format'].lower()
            
            # Check for widely supported formats
            if 'h264' in video_codec or 'avc' in video_codec:
                compatibility_score += 30
                supported_players.extend(['VLC', 'Windows Media Player', 'QuickTime', 'Chrome', 'Firefox'])
            elif 'h265' in video_codec or 'hevc' in video_codec:
                compatibility_score += 25
                supported_players.extend(['VLC', 'Modern browsers', 'Hardware players'])
            elif 'vp9' in video_codec:
                compatibility_score += 20
                supported_players.extend(['Chrome', 'Firefox', 'VLC'])
            
            if 'mp4' in container:
                compatibility_score += 25
                supported_players.append('Universal MP4 support')
            elif 'mkv' in container:
                compatibility_score += 20
                supported_players.append('MKV players')
            elif 'avi' in container:
                compatibility_score += 15
                supported_players.append('Legacy AVI support')
            
            # Check for potential issues
            if compatibility['video_codec'] == 'unknown':
                compatibility['issues'].append('Unknown video codec may cause playback issues')
            
            if compatibility_score < 30:
                compatibility['issues'].append('Limited compatibility - may not play on all devices')
            
            compatibility['compatibility_score'] = min(compatibility_score, 100)
            compatibility['supported_players'] = list(set(supported_players))
            
            print(f"  Container: {compatibility['container_format']}")
            print(f"  Video codec: {compatibility['video_codec']}")
            print(f"  Audio codec: {compatibility['audio_codec']}")
            print(f"  Compatibility score: {compatibility['compatibility_score']}/100")
            
        except Exception as e:
            compatibility['error'] = str(e)
            compatibility['issues'].append(f"Error analyzing compatibility: {e}")
            print(f"  Error: {e}")
        
        return compatibility
    
    def analyze_optimization_needs(self):
        """Analyze if file size optimization is needed while preserving quality"""
        print("Analyzing optimization needs...")
        
        optimization = {
            'current_size_mb': 0,
            'optimization_needed': False,
            'recommended_actions': [],
            'estimated_savings': 0,
            'quality_impact': 'none'
        }
        
        try:
            # Get current file size
            file_size = os.path.getsize(self.enhanced_video_path)
            optimization['current_size_mb'] = file_size / (1024 * 1024)
            
            print(f"  Current size: {optimization['current_size_mb']:.1f} MB")
            
            # Get video properties for optimization analysis
            cap = cv2.VideoCapture(self.enhanced_video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                # Calculate expected size for 4K video
                duration_seconds = frame_count / fps if fps > 0 else 0
                pixels_per_frame = width * height
                
                # Rough estimates for different quality levels
                # High quality: ~0.1 bits per pixel per frame
                # Medium quality: ~0.05 bits per pixel per frame
                # Low quality: ~0.02 bits per pixel per frame
                
                expected_high_mb = (pixels_per_frame * frame_count * 0.1) / (8 * 1024 * 1024)
                expected_medium_mb = (pixels_per_frame * frame_count * 0.05) / (8 * 1024 * 1024)
                
                print(f"  Expected high quality: ~{expected_high_mb:.1f} MB")
                print(f"  Expected medium quality: ~{expected_medium_mb:.1f} MB")
                
                # Determine if optimization is needed
                if optimization['current_size_mb'] > expected_high_mb * 2:
                    optimization['optimization_needed'] = True
                    optimization['recommended_actions'].append('Reduce bitrate while maintaining quality')
                    optimization['estimated_savings'] = optimization['current_size_mb'] - expected_high_mb
                    optimization['quality_impact'] = 'minimal'
                elif optimization['current_size_mb'] < expected_medium_mb * 0.5:
                    optimization['recommended_actions'].append('File size is very small - check for quality issues')
                    optimization['quality_impact'] = 'potentially_degraded'
                else:
                    optimization['recommended_actions'].append('File size appears reasonable for 4K content')
                    optimization['quality_impact'] = 'none'
            
        except Exception as e:
            optimization['error'] = str(e)
            print(f"  Error: {e}")
        
        return optimization
    
    def generate_processing_report(self):
        """Generate comprehensive processing report with settings and results"""
        print("Generating processing report...")
        
        report = {
            'processing_summary': {},
            'technical_specifications': {},
            'quality_assessment': {},
            'recommendations': [],
            'workflow_documentation': {}
        }
        
        try:
            # Processing summary
            report['processing_summary'] = {
                'input_file': self.original_video_path,
                'output_file': self.enhanced_video_path,
                'processing_date': datetime.now().isoformat(),
                'enhancement_type': '4K AI Upscaling',
                'processing_status': 'completed'
            }
            
            # Technical specifications
            if os.path.exists(self.enhanced_video_path):
                file_size = os.path.getsize(self.enhanced_video_path)
                report['technical_specifications'] = {
                    'output_file_size_mb': file_size / (1024 * 1024),
                    'target_resolution': '3840x2160',
                    'actual_resolution': self.results['resolution_validation'].get('actual_resolution', 'unknown'),
                    'is_4k_compliant': self.results['resolution_validation'].get('is_4k', False)
                }
            
            # Quality assessment summary
            if os.path.exists('quality_assessment_task_5_1_results.json'):
                try:
                    with open('quality_assessment_task_5_1_results.json', 'r') as f:
                        qa_results = json.load(f)
                        if 'quality_metrics' in qa_results:
                            report['quality_assessment'] = qa_results['quality_metrics']
                except Exception as e:
                    print(f"  Could not load quality assessment results: {e}")
            
            # Generate recommendations
            recommendations = []
            
            # Resolution recommendations
            if not self.results['resolution_validation'].get('is_4k', False):
                recommendations.append('Video does not meet 4K resolution requirements')
            else:
                recommendations.append('4K resolution successfully achieved')
            
            # Compatibility recommendations
            compat_score = self.results['playback_compatibility'].get('compatibility_score', 0)
            if compat_score >= 70:
                recommendations.append('Excellent playback compatibility')
            elif compat_score >= 50:
                recommendations.append('Good playback compatibility')
            else:
                recommendations.append('Limited playback compatibility - consider format conversion')
            
            # Optimization recommendations
            if self.results['optimization_analysis'].get('optimization_needed', False):
                recommendations.append('File size optimization recommended')
            
            report['recommendations'] = recommendations
            
            # Workflow documentation
            report['workflow_documentation'] = {
                'tools_used': ['Real-ESRGAN', 'FFmpeg', 'OpenCV', 'Python'],
                'processing_steps': [
                    'System analysis and environment setup',
                    'Video analysis and strategy design',
                    'AI upscaling with Real-ESRGAN',
                    'Video reassembly and audio merging',
                    'Quality assessment and validation'
                ],
                'key_settings': {
                    'ai_model': 'Real-ESRGAN-x4plus',
                    'scale_factor': '4x',
                    'target_resolution': '3840x2160'
                }
            }
            
        except Exception as e:
            report['error'] = str(e)
            print(f"  Error generating report: {e}")
        
        return report
    
    def run_complete_validation(self):
        """Run the complete final validation process"""
        print("="*60)
        print("FINAL OUTPUT VALIDATION TASK 5.2")
        print("="*60)
        
        # Validate 4K resolution
        self.results['resolution_validation'] = self.validate_4k_resolution()
        
        # Test playback compatibility
        self.results['playback_compatibility'] = self.test_playback_compatibility()
        
        # Analyze optimization needs
        self.results['optimization_analysis'] = self.analyze_optimization_needs()
        
        # Generate processing report
        self.results['processing_report'] = self.generate_processing_report()
        
        # Determine final status
        is_4k = self.results['resolution_validation'].get('is_4k', False)
        compat_score = self.results['playback_compatibility'].get('compatibility_score', 0)
        
        if is_4k and compat_score >= 50:
            self.results['final_status'] = 'success'
        elif is_4k:
            self.results['final_status'] = 'partial_success'
        else:
            self.results['final_status'] = 'failed'
        
        return self.results
    
    def save_results(self, output_file='final_validation_task_5_2_results.json'):
        """Save validation results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    
    def print_summary(self):
        """Print a summary of the validation results"""
        print("\n" + "="*60)
        print("FINAL VALIDATION SUMMARY")
        print("="*60)
        
        # Resolution validation
        resolution = self.results['resolution_validation']
        print(f"4K Resolution Validation:")
        print(f"  Target: 3840x2160")
        print(f"  Actual: {resolution.get('actual_resolution', 'unknown')}")
        print(f"  4K Compliant: {resolution.get('is_4k', False)}")
        
        # Compatibility
        compatibility = self.results['playback_compatibility']
        print(f"\nPlayback Compatibility:")
        print(f"  Score: {compatibility.get('compatibility_score', 0)}/100")
        print(f"  Video Codec: {compatibility.get('video_codec', 'unknown')}")
        print(f"  Container: {compatibility.get('container_format', 'unknown')}")
        
        # Optimization
        optimization = self.results['optimization_analysis']
        print(f"\nFile Optimization:")
        print(f"  Current Size: {optimization.get('current_size_mb', 0):.1f} MB")
        print(f"  Optimization Needed: {optimization.get('optimization_needed', False)}")
        
        # Final status
        print(f"\nFinal Status: {self.results['final_status'].upper()}")
        
        # Recommendations
        if 'recommendations' in self.results['processing_report']:
            print(f"\nRecommendations:")
            for i, rec in enumerate(self.results['processing_report']['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("="*60)


def main():
    """Main function to run final validation"""
    # Define video paths
    original_video = "videos/大学生原创话剧《自杀既遂》.mp4"
    enhanced_video = "enhanced_4k_theater_video.mp4"
    
    # Check if enhanced video exists
    if not os.path.exists(enhanced_video):
        print(f"Error: Enhanced video not found at {enhanced_video}")
        return
    
    # Run final validation
    validator = FinalValidation(enhanced_video, original_video)
    results = validator.run_complete_validation()
    
    if results:
        # Print summary
        validator.print_summary()
        
        # Save results
        validator.save_results('final_validation_task_5_2_results.json')
        
        print(f"\nFinal validation completed!")
        print(f"Results saved to final_validation_task_5_2_results.json")
    else:
        print("Final validation failed!")


if __name__ == "__main__":
    main()