#!/usr/bin/env python3
"""
多磁盘分段4K视频增强处理器
解决空间不足问题，使用多磁盘存储和及时清理策略

特性:
- 分段处理视频，避免一次性占用大量空间
- 利用多个磁盘存储临时文件
- 及时清理中间文件
- 支持断点续传
- 最小化存储占用
"""

import os
import sys
import json
import subprocess
import logging
import shutil
import time
import psutil
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import gc

class MultiDiskSegmentedEnhancer:
    def __init__(self):
        self.setup_logging()
        
        # 多磁盘配置
        self.disk_config = {
            'primary': '/mnt/d/workProject/huaju4k',  # 当前工作目录
            'storage_e': '/mnt/e/video_temp',         # E盘临时存储
            'storage_g': '/mnt/g/video_temp',         # G盘临时存储
            'output': '/mnt/e/video_output'           # 输出目录
        }
        
        # 创建工作目录
        self.setup_directories()
        
        # 分段配置
        self.segment_config = {
            'frames_per_segment': 1000,  # 每段处理1000帧
            'max_concurrent_segments': 2,  # 最多同时处理2段
            'cleanup_after_segment': True,  # 每段处理完立即清理
            'keep_checkpoints': True       # 保留检查点
        }
        
        # 进度跟踪
        self.progress_file = Path(self.disk_config['primary']) / "segmented_progress.json"
        self.load_progress()
        
    def setup_logging(self):
        """设置日志系统"""
        log_file = f"multi_disk_enhancement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """设置多磁盘工作目录"""
        for key, path in self.disk_config.items():
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
                self.logger.info(f"✅ 创建目录: {path}")
            except Exception as e:
                self.logger.error(f"❌ 创建目录失败 {path}: {e}")
                
    def check_disk_space(self):
        """检查各磁盘可用空间"""
        disk_info = {}
        for name, path in self.disk_config.items():
            try:
                total, used, free = shutil.disk_usage(path)
                disk_info[name] = {
                    'path': path,
                    'free_gb': free / (1024**3),
                    'total_gb': total / (1024**3),
                    'used_percent': (used / total) * 100
                }
                self.logger.info(f"💾 {name}: {disk_info[name]['free_gb']:.1f}GB 可用")
            except Exception as e:
                self.logger.error(f"❌ 检查磁盘空间失败 {path}: {e}")
                
        return disk_info
        
    def load_progress(self):
        """加载处理进度"""
        self.progress = {
            'stage': 'not_started',
            'total_frames': 0,
            'processed_segments': [],
            'current_segment': 0,
            'failed_segments': [],
            'video_info': None,
            'audio_extracted': False
        }
        
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    saved_progress = json.load(f)
                    self.progress.update(saved_progress)
                    self.logger.info(f"📋 已加载进度: 段 {self.progress['current_segment']}")
            except Exception as e:
                self.logger.warning(f"⚠️ 进度文件加载失败: {e}")
                
    def save_progress(self):
        """保存处理进度"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"❌ 进度保存失败: {e}")
            
    def get_video_info(self, video_file):
        """获取视频信息"""
        if self.progress['video_info']:
            return self.progress['video_info']
            
        self.logger.info("📹 分析视频信息...")
        cmd = f'ffprobe -v quiet -print_format json -show_streams "{video_file}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        video_info = json.loads(result.stdout)
        
        video_stream = next(s for s in video_info['streams'] if s['codec_type'] == 'video')
        
        info = {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'fps': eval(video_stream['r_frame_rate']),
            'duration': float(video_stream.get('duration', 0)),
            'total_frames': int(float(video_stream.get('duration', 0)) * eval(video_stream['r_frame_rate']))
        }
        
        self.progress['video_info'] = info
        self.progress['total_frames'] = info['total_frames']
        self.save_progress()
        
        self.logger.info(f"📊 视频信息: {info['width']}x{info['height']}, {info['fps']}fps, {info['total_frames']}帧")
        return info
        
    def extract_audio_once(self, video_file):
        """一次性提取音频到E盘"""
        if self.progress['audio_extracted']:
            audio_file = Path(self.disk_config['storage_e']) / "original_audio.wav"
            if audio_file.exists():
                self.logger.info("🎵 音频已提取，跳过")
                return str(audio_file)
                
        self.logger.info("🎵 提取音频到E盘...")
        audio_file = Path(self.disk_config['storage_e']) / "original_audio.wav"
        
        cmd = f'ffmpeg -i "{video_file}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{audio_file}" -y'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if audio_file.exists():
            self.logger.info(f"✅ 音频提取完成: {audio_file}")
            self.progress['audio_extracted'] = True
            self.save_progress()
            return str(audio_file)
        else:
            raise RuntimeError("音频提取失败")
            
    def extract_segment_frames(self, video_file, segment_idx, start_frame, end_frame):
        """提取指定段的帧到临时目录"""
        # 使用E盘和G盘轮换存储
        storage_disk = 'storage_e' if segment_idx % 2 == 0 else 'storage_g'
        segment_dir = Path(self.disk_config[storage_disk]) / f"segment_{segment_idx:04d}"
        segment_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"📹 提取段 {segment_idx}: 帧 {start_frame}-{end_frame} 到 {storage_disk}")
        
        # 计算时间范围
        video_info = self.progress['video_info']
        fps = video_info['fps']
        start_time = start_frame / fps
        duration = (end_frame - start_frame + 1) / fps
        
        # 提取帧
        cmd = f'ffmpeg -ss {start_time} -i "{video_file}" -t {duration} -vf fps={fps} "{segment_dir}/frame_%08d.png" -y'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 检查提取的帧数
        frame_files = list(segment_dir.glob("*.png"))
        actual_frames = len(frame_files)
        
        self.logger.info(f"✅ 段 {segment_idx} 提取完成: {actual_frames} 帧")
        return segment_dir, frame_files
        
    def enhance_segment_frames(self, segment_dir, frame_files, segment_idx):
        """增强指定段的帧"""
        enhanced_dir = segment_dir.parent / f"enhanced_{segment_idx:04d}"
        enhanced_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"🚀 增强段 {segment_idx}: {len(frame_files)} 帧")
        
        processed = 0
        failed = 0
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for frame_file in frame_files:
                future = executor.submit(self.enhance_single_frame, frame_file, enhanced_dir)
                futures.append(future)
                
            for future in as_completed(futures):
                try:
                    success = future.result()
                    if success:
                        processed += 1
                    else:
                        failed += 1
                        
                    # 每处理100帧显示进度
                    if (processed + failed) % 100 == 0:
                        progress = ((processed + failed) / len(frame_files)) * 100
                        self.logger.info(f"📊 段 {segment_idx} 进度: {progress:.1f}% ({processed}/{len(frame_files)})")
                        
                except Exception as e:
                    self.logger.error(f"❌ 帧处理失败: {e}")
                    failed += 1
                    
        self.logger.info(f"✅ 段 {segment_idx} 增强完成: {processed}/{len(frame_files)} 成功")
        return enhanced_dir, processed, failed
        
    def enhance_single_frame(self, frame_path, output_dir):
        """增强单个帧"""
        try:
            output_path = output_dir / frame_path.name
            
            # 加载图像
            img = cv2.imread(str(frame_path))
            if img is None:
                return False
                
            # 使用高质量插值放大到4K
            height, width = img.shape[:2]
            new_width = width * 2  # 1920 -> 3840
            new_height = height * 2  # 1080 -> 2160
            
            enhanced = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # 保存增强帧
            success = cv2.imwrite(str(output_path), enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            return success
            
        except Exception as e:
            self.logger.error(f"❌ 单帧处理失败 {frame_path}: {e}")
            return False
            
    def assemble_segment_video(self, enhanced_dir, segment_idx, fps):
        """将增强帧组装成视频段"""
        segment_video = enhanced_dir.parent / f"segment_{segment_idx:04d}.mp4"
        
        self.logger.info(f"🎞️ 组装段 {segment_idx} 视频...")
        
        cmd = f'ffmpeg -framerate {fps} -i "{enhanced_dir}/frame_%08d.png" -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p "{segment_video}" -y'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if segment_video.exists():
            file_size = segment_video.stat().st_size / (1024*1024)
            self.logger.info(f"✅ 段 {segment_idx} 视频完成: {file_size:.1f}MB")
            return str(segment_video)
        else:
            raise RuntimeError(f"段 {segment_idx} 视频组装失败")
            
    def cleanup_segment_files(self, segment_dir, enhanced_dir):
        """清理段的临时文件"""
        try:
            if segment_dir.exists():
                shutil.rmtree(segment_dir)
                self.logger.info(f"🧹 已清理原始帧目录: {segment_dir.name}")
                
            if enhanced_dir.exists():
                shutil.rmtree(enhanced_dir)
                self.logger.info(f"🧹 已清理增强帧目录: {enhanced_dir.name}")
                
            # 强制垃圾回收
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"❌ 清理失败: {e}")
            
    def merge_all_segments(self, segment_videos, audio_file, output_video):
        """合并所有视频段和音频"""
        self.logger.info("🎬 合并所有视频段...")
        
        # 创建文件列表
        filelist_path = Path(self.disk_config['primary']) / "segments_list.txt"
        with open(filelist_path, 'w') as f:
            for video in segment_videos:
                f.write(f"file '{video}'\n")
                
        # 合并视频段
        temp_video = Path(self.disk_config['output']) / "temp_merged.mp4"
        cmd = f'ffmpeg -f concat -safe 0 -i "{filelist_path}" -c copy "{temp_video}" -y'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if not temp_video.exists():
            raise RuntimeError("视频段合并失败")
            
        # 添加音频
        cmd = f'ffmpeg -i "{temp_video}" -i "{audio_file}" -c:v copy -c:a aac -b:a 192k -shortest "{output_video}" -y'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 清理临时文件
        if temp_video.exists():
            temp_video.unlink()
        if filelist_path.exists():
            filelist_path.unlink()
            
        if Path(output_video).exists():
            file_size = Path(output_video).stat().st_size / (1024*1024)
            self.logger.info(f"✅ 最终视频完成: {file_size:.1f}MB")
            return True
        else:
            raise RuntimeError("最终视频合并失败")
            
    def process_video_segmented(self, input_video, output_video=None):
        """分段处理完整视频"""
        start_time = time.time()
        
        if output_video is None:
            output_video = str(Path(self.disk_config['output']) / "final_4k_theater_video.mp4")
            
        try:
            self.logger.info("🎬 开始多磁盘分段4K视频增强")
            self.logger.info(f"📁 输入: {input_video}")
            self.logger.info(f"📁 输出: {output_video}")
            
            # 检查磁盘空间
            disk_info = self.check_disk_space()
            
            # 获取视频信息
            video_info = self.get_video_info(input_video)
            
            # 提取音频（一次性）
            audio_file = self.extract_audio_once(input_video)
            
            # 计算分段
            frames_per_segment = self.segment_config['frames_per_segment']
            total_segments = (video_info['total_frames'] + frames_per_segment - 1) // frames_per_segment
            
            self.logger.info(f"📊 总计 {total_segments} 段，每段 {frames_per_segment} 帧")
            
            # 处理各段
            segment_videos = []
            
            for segment_idx in range(total_segments):
                if segment_idx in self.progress['processed_segments']:
                    # 跳过已处理的段
                    segment_video = Path(self.disk_config['storage_e' if segment_idx % 2 == 0 else 'storage_g']) / f"segment_{segment_idx:04d}.mp4"
                    if segment_video.exists():
                        segment_videos.append(str(segment_video))
                        self.logger.info(f"⏭️ 跳过已处理段 {segment_idx}")
                        continue
                
                start_frame = segment_idx * frames_per_segment
                end_frame = min(start_frame + frames_per_segment - 1, video_info['total_frames'] - 1)
                
                self.logger.info(f"🔄 处理段 {segment_idx + 1}/{total_segments}")
                
                try:
                    # 提取帧
                    segment_dir, frame_files = self.extract_segment_frames(
                        input_video, segment_idx, start_frame, end_frame)
                    
                    # 增强帧
                    enhanced_dir, processed, failed = self.enhance_segment_frames(
                        segment_dir, frame_files, segment_idx)
                    
                    # 组装视频段
                    segment_video = self.assemble_segment_video(
                        enhanced_dir, segment_idx, video_info['fps'])
                    
                    segment_videos.append(segment_video)
                    
                    # 清理临时文件
                    if self.segment_config['cleanup_after_segment']:
                        self.cleanup_segment_files(segment_dir, enhanced_dir)
                    
                    # 更新进度
                    self.progress['processed_segments'].append(segment_idx)
                    self.progress['current_segment'] = segment_idx + 1
                    self.save_progress()
                    
                    self.logger.info(f"✅ 段 {segment_idx + 1} 完成")
                    
                except Exception as e:
                    self.logger.error(f"❌ 段 {segment_idx} 处理失败: {e}")
                    self.progress['failed_segments'].append(segment_idx)
                    self.save_progress()
                    continue
                    
            # 合并所有段
            self.logger.info("🎬 开始最终合并...")
            self.merge_all_segments(segment_videos, audio_file, output_video)
            
            # 清理段视频文件
            for segment_video in segment_videos:
                try:
                    Path(segment_video).unlink()
                    self.logger.info(f"🧹 已清理段视频: {Path(segment_video).name}")
                except:
                    pass
                    
            total_time = time.time() - start_time
            self.logger.info(f"🎉 分段处理完成! 总用时: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
            self.logger.info(f"✅ 输出文件: {output_video}")
            
            return output_video
            
        except Exception as e:
            self.logger.error(f"❌ 分段处理失败: {str(e)}")
            raise

def main():
    """主程序入口"""
    if len(sys.argv) < 2:
        print("使用方法: python3 multi_disk_4k_enhancer.py input_video.mp4 [output_video.mp4]")
        print("特性:")
        print("  - 多磁盘存储，避免空间不足")
        print("  - 分段处理，及时清理")
        print("  - 支持断点续传")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else None
    
    enhancer = MultiDiskSegmentedEnhancer()
    enhancer.process_video_segmented(input_video, output_video)

if __name__ == "__main__":
    main()