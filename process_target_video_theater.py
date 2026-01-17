#!/usr/bin/env python3
"""
戏剧视频增强处理脚本

用于处理 /mnt/c/Users/Administrator/Downloads/target.mp4
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# 配置日志
log_file = f"theater_enhancement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_theater_video(input_path: str, output_path: str):
    """处理戏剧视频"""
    
    logger.info("=" * 60)
    logger.info("戏剧视频4K增强系统")
    logger.info("=" * 60)
    
    try:
        # 导入组件
        logger.info("加载组件...")
        from huaju4k.analysis.stage_structure_analyzer import StageStructureAnalyzer
        from huaju4k.strategy.enhancement_planner import EnhancementStrategyPlanner
        from huaju4k.core.ai_model_manager import StrategyDrivenModelManager
        from huaju4k.core.progress_tracker import MultiStageProgressTracker
        from huaju4k.core.three_stage_enhancer import ThreeStageVideoEnhancer
        
        # 验证输入
        if not Path(input_path).exists():
            logger.error(f"输入文件不存在: {input_path}")
            return False
        
        file_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
        logger.info(f"输入文件: {input_path} ({file_size_mb:.1f} MB)")
        logger.info(f"输出文件: {output_path}")
        
        # 步骤1: 分析视频
        logger.info("\n" + "=" * 40)
        logger.info("步骤 1/4: 分析视频结构")
        logger.info("=" * 40)
        
        analyzer = StageStructureAnalyzer()
        features = analyzer.analyze_structure(input_path)
        
        if not features:
            logger.error("视频分析失败")
            return False
        
        logger.info(f"分辨率: {features.resolution}")
        logger.info(f"帧率: {features.fps} fps")
        logger.info(f"时长: {features.duration:.1f} 秒 ({features.duration/60:.1f} 分钟)")
        logger.info(f"总帧数: {features.total_frames}")
        logger.info(f"静态相机: {features.is_static_camera}")
        logger.info(f"噪声分数: {features.noise_score:.3f}")
        logger.info(f"边缘密度: {features.edge_density:.3f}")
        
        # 步骤2: 生成策略
        logger.info("\n" + "=" * 40)
        logger.info("步骤 2/4: 生成增强策略")
        logger.info("=" * 40)
        
        planner = EnhancementStrategyPlanner()
        strategy = planner.generate_strategy(features)
        
        logger.info(f"分辨率路径: {strategy.resolution_plan}")
        logger.info(f"GAN增强: {'启用' if strategy.gan_policy.global_allowed else '禁用'}")
        logger.info(f"GAN强度: {strategy.gan_policy.strength}")
        logger.info(f"背景锁定: {'启用' if strategy.temporal_strategy.background_lock else '禁用'}")
        logger.info(f"瓦片大小: {strategy.memory_policy.tile_size}")
        
        # 步骤3: 初始化处理器
        logger.info("\n" + "=" * 40)
        logger.info("步骤 3/4: 初始化处理器")
        logger.info("=" * 40)
        
        model_manager = StrategyDrivenModelManager()
        progress_tracker = MultiStageProgressTracker()
        
        # 进度回调
        last_progress = [0]
        def progress_callback(stage: str, progress: float, message: str):
            current = int(progress * 100)
            if current > last_progress[0] or current == 0:
                logger.info(f"[{stage}] {current}% - {message}")
                last_progress[0] = current
        
        # 步骤4: 执行增强
        logger.info("\n" + "=" * 40)
        logger.info("步骤 4/4: 执行三阶段视频增强")
        logger.info("=" * 40)
        logger.info("注意: 这可能需要很长时间，请耐心等待...")
        
        enhancer = ThreeStageVideoEnhancer(model_manager, progress_tracker)
        
        success = enhancer.enhance_video(
            input_path=input_path,
            output_path=output_path,
            strategy=strategy,
            progress_callback=progress_callback
        )
        
        if success:
            logger.info("\n" + "=" * 60)
            logger.info("✅ 视频增强完成！")
            logger.info("=" * 60)
            
            stats = enhancer.get_enhancement_statistics()
            logger.info(f"总帧数: {stats['total_frames_processed']}")
            logger.info(f"总时间: {stats['total_processing_time']:.1f} 秒")
            logger.info(f"平均FPS: {stats['average_fps']:.2f}")
            logger.info(f"输出文件: {output_path}")
            
            if Path(output_path).exists():
                output_size = Path(output_path).stat().st_size / (1024 * 1024)
                logger.info(f"输出大小: {output_size:.1f} MB")
            
            return True
        else:
            logger.error("❌ 视频增强失败")
            return False
            
    except Exception as e:
        logger.error(f"处理错误: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # 默认路径
    default_input = "/mnt/c/Users/Administrator/Downloads/target.mp4"
    default_output = "target_enhanced_4k.mp4"
    
    if len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    elif len(sys.argv) == 2:
        input_path = sys.argv[1]
        output_path = default_output
    else:
        input_path = default_input
        output_path = default_output
    
    print(f"\n戏剧视频4K增强")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print()
    
    success = process_theater_video(input_path, output_path)
    sys.exit(0 if success else 1)
