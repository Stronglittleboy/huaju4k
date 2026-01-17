#!/usr/bin/env python3
"""
调试视频分析问题的简单测试脚本
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_video_analysis():
    """测试视频分析功能"""
    video_path = "/mnt/c/Users/Administrator/Downloads/target.mp4"
    
    print(f"测试视频文件: {video_path}")
    print(f"文件存在: {os.path.exists(video_path)}")
    
    if os.path.exists(video_path):
        file_size = os.path.getsize(video_path)
        print(f"文件大小: {file_size / (1024*1024):.1f} MB")
    
    try:
        print("\n1. 测试导入...")
        from huaju4k.core.video_analyzer import VideoAnalyzer
        print("✅ VideoAnalyzer 导入成功")
        
        print("\n2. 初始化VideoAnalyzer...")
        analyzer = VideoAnalyzer()
        print("✅ VideoAnalyzer 初始化成功")
        
        print("\n3. 开始视频分析...")
        video_info = analyzer.analyze_video(video_path)
        print("✅ 视频分析完成")
        
        print(f"\n视频信息:")
        print(f"  分辨率: {video_info.resolution}")
        print(f"  时长: {video_info.duration:.1f} 秒")
        print(f"  帧率: {video_info.framerate:.1f} fps")
        print(f"  编解码器: {video_info.codec}")
        print(f"  有音频: {video_info.has_audio}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_video_analysis()
    sys.exit(0 if success else 1)