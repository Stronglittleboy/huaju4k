#!/usr/bin/env python3
"""
优化的视频增强处理脚本 - 使用FFmpeg流式处理

避免提取所有帧到磁盘，直接使用FFmpeg进行4K放大
"""

import sys
import os
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# 配置日志
log_file = f"ffmpeg_enhancement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_video_info(input_path: str) -> dict:
    """获取视频信息"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            input_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            
            video_stream = None
            audio_stream = None
            for stream in info.get('streams', []):
                if stream['codec_type'] == 'video' and not video_stream:
                    video_stream = stream
                elif stream['codec_type'] == 'audio' and not audio_stream:
                    audio_stream = stream
            
            return {
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': eval(video_stream.get('r_frame_rate', '25/1')),
                'duration': float(info['format'].get('duration', 0)),
                'has_audio': audio_stream is not None,
                'video_codec': video_stream.get('codec_name', 'unknown'),
                'audio_codec': audio_stream.get('codec_name', 'unknown') if audio_stream else None
            }
    except Exception as e:
        logger.error(f"获取视频信息失败: {e}")
    return None

def enhance_video_ffmpeg(input_path: str, output_path: str, target_width: int = 3840, target_height: int = 2160):
    """
    使用FFmpeg进行视频增强
    
    使用高质量的缩放算法和滤镜进行4K放大
    """
    logger.info("=" * 60)
    logger.info("FFmpeg优化视频增强")
    logger.info("=" * 60)
    
    # 检查输入文件
    if not Path(input_path).exists():
        logger.error(f"输入文件不存在: {input_path}")
        return False
    
    # 获取视频信息
    info = get_video_info(input_path)
    if not info:
        logger.error("无法获取视频信息")
        return False
    
    logger.info(f"输入视频: {input_path}")
    logger.info(f"  分辨率: {info['width']}x{info['height']}")
    logger.info(f"  帧率: {info['fps']:.2f} fps")
    logger.info(f"  时长: {info['duration']:.1f} 秒 ({info['duration']/60:.1f} 分钟)")
    logger.info(f"  视频编码: {info['video_codec']}")
    logger.info(f"  音频: {'有' if info['has_audio'] else '无'}")
    
    logger.info(f"\n目标分辨率: {target_width}x{target_height}")
    logger.info(f"输出文件: {output_path}")
    
    # 构建FFmpeg滤镜链
    # 使用lanczos缩放 + unsharp锐化 + 轻微降噪
    filter_complex = (
        f"scale={target_width}:{target_height}:flags=lanczos,"  # 高质量缩放
        f"unsharp=5:5:0.8:5:5:0.4,"  # 锐化增强
        f"hqdn3d=1.5:1.5:6:6"  # 轻微降噪
    )
    
    # 构建FFmpeg命令
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vf', filter_complex,
        '-c:v', 'libx264',
        '-preset', 'medium',  # 平衡速度和质量
        '-crf', '18',  # 高质量
        '-pix_fmt', 'yuv420p',
    ]
    
    # 处理音频
    if info['has_audio']:
        cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
    
    # 添加进度显示
    cmd.extend(['-progress', 'pipe:1', '-stats'])
    
    cmd.append(output_path)
    
    logger.info(f"\n开始处理...")
    logger.info(f"FFmpeg命令: {' '.join(cmd[:10])}...")
    
    try:
        # 运行FFmpeg
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # 监控进度
        frame_count = 0
        total_frames = int(info['duration'] * info['fps'])
        
        for line in process.stdout:
            line = line.strip()
            if 'frame=' in line:
                try:
                    # 解析帧数
                    parts = line.split()
                    for part in parts:
                        if part.startswith('frame='):
                            frame_count = int(part.split('=')[1])
                            break
                    
                    if total_frames > 0:
                        progress = (frame_count / total_frames) * 100
                        if frame_count % 500 == 0:  # 每500帧报告一次
                            logger.info(f"进度: {progress:.1f}% ({frame_count}/{total_frames} 帧)")
                except:
                    pass
            elif 'error' in line.lower():
                logger.warning(f"FFmpeg: {line}")
        
        process.wait()
        
        if process.returncode == 0:
            logger.info("\n" + "=" * 60)
            logger.info("✅ 视频增强完成！")
            logger.info("=" * 60)
            
            if Path(output_path).exists():
                output_size = Path(output_path).stat().st_size / (1024 * 1024)
                logger.info(f"输出文件: {output_path}")
                logger.info(f"输出大小: {output_size:.1f} MB")
            
            return True
        else:
            logger.error(f"FFmpeg处理失败，返回码: {process.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"处理错误: {e}", exc_info=True)
        return False

def enhance_video_two_pass(input_path: str, output_path: str):
    """
    两阶段增强：先2x放大，再2x放大到4K
    
    这种方式可以获得更好的质量
    """
    logger.info("=" * 60)
    logger.info("两阶段4K增强 (1080p -> 2160p -> 4320p -> 4K)")
    logger.info("=" * 60)
    
    info = get_video_info(input_path)
    if not info:
        return False
    
    # 临时文件
    temp_2x = str(Path(output_path).parent / "temp_2x_upscale.mp4")
    
    try:
        # 第一阶段: 2x放大 (1080p -> 2160p)
        logger.info("\n阶段 1/2: 2x放大...")
        width_2x = info['width'] * 2
        height_2x = info['height'] * 2
        
        cmd1 = [
            'ffmpeg', '-y', '-i', input_path,
            '-vf', f"scale={width_2x}:{height_2x}:flags=lanczos,unsharp=3:3:0.5",
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
            '-an',  # 暂时不处理音频
            temp_2x
        ]
        
        result = subprocess.run(cmd1, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"第一阶段失败: {result.stderr}")
            return False
        
        logger.info(f"第一阶段完成: {width_2x}x{height_2x}")
        
        # 第二阶段: 缩放到4K并添加音频
        logger.info("\n阶段 2/2: 缩放到4K并合并音频...")
        
        cmd2 = [
            'ffmpeg', '-y',
            '-i', temp_2x,
            '-i', input_path,  # 原始音频
            '-filter_complex',
            f"[0:v]scale=3840:2160:flags=lanczos,unsharp=5:5:0.8,hqdn3d=1.5:1.5:6:6[v]",
            '-map', '[v]',
            '-map', '1:a?',  # 可选音频
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-c:a', 'aac', '-b:a', '192k',
            output_path
        ]
        
        result = subprocess.run(cmd2, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"第二阶段失败: {result.stderr}")
            return False
        
        logger.info("✅ 两阶段增强完成！")
        return True
        
    finally:
        # 清理临时文件
        if Path(temp_2x).exists():
            os.remove(temp_2x)
            logger.info("已清理临时文件")

if __name__ == "__main__":
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
    
    print(f"\n视频4K增强 (FFmpeg优化版)")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print()
    
    # 使用单阶段快速处理
    success = enhance_video_ffmpeg(input_path, output_path)
    
    # 或者使用两阶段高质量处理（取消注释下面这行）
    # success = enhance_video_two_pass(input_path, output_path)
    
    sys.exit(0 if success else 1)
