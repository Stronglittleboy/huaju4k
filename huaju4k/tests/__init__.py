"""
Test suite for huaju4k video enhancement tool.
"""

# Test configuration
TEST_CONFIG = {
    'test_data_dir': './test_data',
    'temp_test_dir': './test_temp',
    'mock_video_duration': 10.0,  # seconds
    'mock_video_resolution': (1920, 1080),
    'mock_video_framerate': 30.0
}

__all__ = ['TEST_CONFIG']