#!/usr/bin/env python3
"""
调试VideoEnhancementProcessor初始化问题
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

def test_processor_components():
    """逐步测试处理器组件"""
    
    try:
        print("1. 测试配置管理器...")
        from huaju4k.configs.simple_config_manager import SimpleConfigManager
        config_manager = SimpleConfigManager()
        config = config_manager.load_config(None)
        print("✅ 配置管理器正常")
        
        print("\n2. 测试视频分析器...")
        from huaju4k.core.video_analyzer import VideoAnalyzer
        video_analyzer = VideoAnalyzer(
            config=config.get('video', {}),
            performance_config=config.get('performance', {})
        )
        print("✅ 视频分析器正常")
        
        print("\n3. 测试AI模型管理器...")
        from huaju4k.core.ai_model_manager import AIModelManager
        ai_model_manager = AIModelManager(
            models_dir=config.get('models_dir', './models'),
            cache_size=config.get('model_cache_size', 2)
        )
        print("✅ AI模型管理器正常")
        
        print("\n4. 测试音频增强器...")
        try:
            from huaju4k.core.theater_audio_enhancer import TheaterAudioEnhancer
            audio_enhancer = TheaterAudioEnhancer(
                theater_preset=config.get('theater_preset', 'medium'),
                config=config.get('audio', {})
            )
            print("✅ 音频增强器正常")
        except ImportError as e:
            print(f"⚠️ 音频增强器导入失败: {e}")
        
        print("\n5. 测试内存管理器...")
        from huaju4k.core.memory_manager import ConservativeMemoryManager
        memory_manager = ConservativeMemoryManager(
            safety_margin=config.get('memory_safety_margin', 0.7),
            temp_dir=config.get('temp_dir', None)
        )
        print("✅ 内存管理器正常")
        
        print("\n6. 测试进度跟踪器...")
        from huaju4k.core.progress_tracker import MultiStageProgressTracker
        progress_tracker = MultiStageProgressTracker(
            update_interval=config.get('progress_update_interval', 0.5)
        )
        print("✅ 进度跟踪器正常")
        
        print("\n7. 测试瓦片处理器...")
        from huaju4k.core.tile_processor import TileProcessor
        tile_processor = TileProcessor(
            memory_manager=memory_manager,
            progress_tracker=progress_tracker
        )
        print("✅ 瓦片处理器正常")
        
        print("\n8. 测试完整处理器初始化...")
        from huaju4k.core.video_enhancement_processor import VideoEnhancementProcessor
        processor = VideoEnhancementProcessor(config_path=None)
        print("✅ 视频增强处理器初始化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_processor_components()
    sys.exit(0 if success else 1)