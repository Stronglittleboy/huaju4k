#!/usr/bin/env python3
"""
Script to analyze the theater drama video file.

This script performs comprehensive analysis of the theater video including:
- Video specifications (resolution, codec, bitrate, duration)
- Audio track analysis
- Theater-specific characteristics assessment
- Quality and enhancement recommendations
"""

import sys
import logging
from pathlib import Path

# Add the video_enhancer module to the path
sys.path.insert(0, str(Path(__file__).parent))

from video_enhancer.video_analyzer import VideoAnalyzer
from video_enhancer.models import VideoMetadata


def main():
    """Main function to analyze the theater video."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Path to the theater drama video
    video_path = "videos/大学生原创话剧《自杀既遂》.mp4"
    
    print("Theater Drama Video Analysis")
    print("=" * 50)
    print(f"Analyzing: {video_path}")
    
    try:
        # Initialize analyzer
        analyzer = VideoAnalyzer()
        
        # Perform analysis
        print("\nStarting comprehensive video analysis...")
        metadata = analyzer.analyze_video_file(video_path)
        
        # Print detailed report
        analyzer.print_analysis_report(metadata)
        
        # Additional analysis summary
        print("\nANALYSIS SUMMARY:")
        print("-" * 30)
        
        # Check if video needs enhancement
        if metadata.is_4k:
            print("✓ Video is already 4K resolution")
        else:
            scale_factor = 3840 / metadata.width
            print(f"→ Video needs {scale_factor:.1f}x upscaling to reach 4K")
        
        # Enhancement recommendations
        if metadata.theater_characteristics:
            tc = metadata.theater_characteristics
            print(f"→ Recommended AI model: {tc.recommended_model}")
            print(f"→ Processing approach: {tc.processing_priority.value}")
            
            # Specific recommendations based on analysis
            if tc.lighting_type.value == "stage":
                print("→ Stage lighting detected - use high-quality enhancement")
            if tc.scene_complexity.value == "complex":
                print("→ Complex scenes detected - prioritize quality over speed")
            if tc.actor_count > 3:
                print("→ Multiple actors detected - ensure facial detail preservation")
        
        print("\n✓ Analysis completed successfully!")
        
        return metadata
        
    except FileNotFoundError:
        print(f"Error: Video file not found at {video_path}")
        print("Please ensure the video file exists in the videos/ directory")
        return None
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None


if __name__ == "__main__":
    result = main()
    if result is None:
        sys.exit(1)