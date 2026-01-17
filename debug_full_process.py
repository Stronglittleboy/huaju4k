#!/usr/bin/env python3
"""
调试完整处理流程
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_full_process():
    """测试完整处理流程"""
    video_path = "/mnt/c/Users/Administrator/Downloads/target.mp4"
    
    try:
        print("初始化视频增强处理器...")
        from huaju4k.core.video_enhancement_processor import VideoEnhancementProcessor
        processor = VideoEnhancementProcessor(config_path=None)
        print("✅ 处理器初始化成功")
        
        print("\n开始处理...")
        result = processor.process(
            input_path=video_path,
            output_path=None,  # 使用默认输出路径
            preset="theater_medium",
            quality="balanced"
        )
        
        if result.success:
            print(f"✅ 处理成功！输出文件: {result.output_path}")
            print(f"处理时间: {result.processing_time:.1f} 秒")
        else:
            print(f"❌ 处理失败: {result.error}")
            
        return result.success
        
    except Exception as e:
        print(f"❌ 异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_process()
    sys.exit(0 if success else 1)