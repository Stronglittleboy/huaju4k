#!/usr/bin/env python3
"""
简化的处理测试，禁用进度条
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simple_process():
    """简化的处理测试"""
    video_path = "/mnt/c/Users/Administrator/Downloads/target.mp4"
    
    try:
        print("1. 初始化处理器...")
        from huaju4k.core.video_enhancement_processor import VideoEnhancementProcessor
        processor = VideoEnhancementProcessor(config_path=None)
        
        # 禁用进度条显示
        processor.progress_tracker.display_enabled = False
        print("✅ 处理器初始化成功，进度条已禁用")
        
        print("\n2. 开始视频分析...")
        video_info = processor.analyze_video(video_path)
        print(f"✅ 视频分析完成: {video_info.resolution[0]}x{video_info.resolution[1]}, {video_info.duration:.1f}s")
        
        print("\n3. 计算处理策略...")
        strategy = processor._calculate_processing_strategy(video_info, "balanced", "theater_medium")
        print(f"✅ 策略计算完成: 瓦片大小 {strategy.tile_size}, 目标分辨率 {strategy.target_resolution}")
        
        print("\n4. 测试AI模型选择...")
        available_memory = processor.memory_manager.get_available_memory()
        model_name = processor.ai_model_manager.auto_select_model(
            strategy.target_resolution,
            available_memory['system']
        )
        print(f"✅ 选择AI模型: {model_name}")
        
        print("\n✅ 所有核心功能测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_process()
    sys.exit(0 if success else 1)