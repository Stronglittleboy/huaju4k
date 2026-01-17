#!/usr/bin/env python3
"""
处理指定的target.mp4文件
"""

import os
import sys
import logging
from pathlib import Path

def main():
    """处理target.mp4文件"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # 输入和输出文件
    input_video = "/mnt/c/Users/Administrator/Downloads/target.mp4"
    output_video = "/mnt/c/Users/Administrator/Downloads/target_enhanced_4k.mp4"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_video):
        logger.error(f"输入视频文件不存在: {input_video}")
        return False
    
    # 获取文件信息
    file_size = os.path.getsize(input_video) / (1024 * 1024)
    logger.info(f"输入文件: {input_video}")
    logger.info(f"输入文件大小: {file_size:.2f} MB")
    
    try:
        # 添加项目路径
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from huaju4k.core.video_enhancement_processor import VideoEnhancementProcessor
        
        logger.info("="*60)
        logger.info("开始Huaju4K剧院级4K增强处理")
        logger.info("="*60)
        
        # 创建处理器
        processor = VideoEnhancementProcessor()
        
        # 执行处理
        result = processor.process(
            input_path=input_video,
            output_path=output_video,
            preset="theater_medium",  # 剧院中等预设
            quality="balanced"        # 平衡质量
        )
        
        if result.success:
            logger.info("="*60)
            logger.info("✅ 视频增强处理成功完成！")
            logger.info("="*60)
            logger.info(f"输出文件: {result.output_path}")
            logger.info(f"处理时间: {result.processing_time:.1f}秒")
            
            # 检查输出文件
            if os.path.exists(result.output_path):
                output_size = os.path.getsize(result.output_path) / (1024 * 1024)
                logger.info(f"输出文件大小: {output_size:.2f} MB")
                logger.info(f"文件大小增长: {output_size/file_size:.1f}x")
                
                # 显示质量指标
                if result.quality_metrics:
                    logger.info("\n🎯 质量指标:")
                    key_metrics = [
                        'overall_score', 'resolution_improvement_ratio', 
                        'brightness_stability', 'edge_stability'
                    ]
                    for key in key_metrics:
                        if key in result.quality_metrics:
                            value = result.quality_metrics[key]
                            if isinstance(value, float):
                                logger.info(f"  {key}: {value:.3f}")
                            else:
                                logger.info(f"  {key}: {value}")
                
                logger.info(f"\n🎉 增强完成！输出文件保存在:")
                logger.info(f"   {os.path.abspath(result.output_path)}")
                
            return True
        else:
            logger.error("="*60)
            logger.error("❌ 视频增强处理失败")
            logger.error("="*60)
            logger.error(f"错误信息: {result.error}")
            return False
            
    except Exception as e:
        logger.error(f"处理过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)