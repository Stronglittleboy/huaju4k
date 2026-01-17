#!/usr/bin/env python3
"""
System Analysis Runner for Video Enhancement Project

This script runs the system analysis to assess hardware capabilities
for video enhancement processing.
"""

import sys
import os

# Add the video_enhancer module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'video_enhancer'))

from video_enhancer.system_analyzer import main

if __name__ == "__main__":
    print("Video Enhancement System Analysis")
    print("=" * 50)
    print()
    
    try:
        specs = main()
        print()
        print("System analysis completed successfully!")
        print("Check 'system_specs.json' for detailed specifications.")
        
    except Exception as e:
        print(f"Error during system analysis: {e}")
        sys.exit(1)