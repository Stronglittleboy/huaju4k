#!/usr/bin/env python3
"""
CLI Application Entry Point

Command-line entry point for the Video Enhancement Toolkit.
"""

import sys
import os

# Add the parent directory to the path so we can import the toolkit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_enhancement_toolkit.main import main

if __name__ == "__main__":
    sys.exit(main())