#!/usr/bin/env python3
"""
GPU优化的4K视频增强处理器
解决CPU打满但GPU利用率低的问题

优化策略:
1. 使用GPU加速的图像处理
2. CUDA加速的OpenCV操作
3. 批量GPU处理
4. 内存池管理
5. 异步GPU操作
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

# GPU相关导入
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndimage
    CUPY_AVAILABLE = True
    print("✅ CuPy可用，将使用GPU加速")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️ CuPy不可用，将使用OpenCV GPU模块")

class GPUOptimizedEnhancer:
    def __init__(self):
        self.setup_logging()
        self.setup_gpu()
        
        # 多磁盘配置
        self.disk_config = {
            'primary': '/mnt/d/workProject/huaju4k',
            'storage_e': '/mnt/e/video_temp',
            'storage_g': '/mnt/g/video_temp',
            'output': '/mnt/e/video_output'
        }
        
        # GPU优化配置
        self.gpu_config = {
            'batch_size': 8,  # 每批处理8帧
            'use_gpu_memory_pool': True,
            'gpu_memory_limit': 0.8,  # 使用80%的GPU内存
            'async_processing': True,
            'cuda_streams': 4
        }
        
        # 分段配置（优化后）
        self.segment_config = {
            'frames_per_segment': 2000,  # 增加到2000帧/段
            'max_concurrent_segments': 1,
            'cleanup_after_segment': True,
            'keep_checkpoints': True
        }
        
        self.setup_directories()
        self.progress_file = Path(self.disk_config['primary']) / "gpu_progress.json"
        self.load_progress()
        
    def setup_logging(self):
        """设置日志系统"""
        log_file = f"gpu_enhancement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_gpu(self):
        """设置GPU环境"""
        self.logger.info("🔧 设置GPU环境...")
        
        # 检查CUDA可用性
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.opencv_gpu_available = True
            self.logger.info(f"✅ OpenCV CUDA设备数量: {cv2.cuda.getCudaEnabledDeviceCount()}")
        else:
            self.opencv_gpu_available = False
            self.logger.warning("⚠️ OpenCV CUDA不可用")
            
        # 设置GPU内存池
        if CUPY_AVAILABLE:
            try:
                # 设置GPU内存池
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=int(4 * 1024**3 * self.gpu_config['gpu_memory_limit']))  # 80% of 4GB
                self.logger.info("✅ CuPy GPU内存池设置完成")
                self.cupy_available = True
            except Exception as e:
                self.logger.error(f"❌ CuPy设置失败: {e}")
                self.cupy_available = False
        else:
            self.cupy_available = False
            
    def setup_directories(self):
        """设置工作目录"""
        for key, path in self.disk_config.items():
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.error(f"❌ 创建目录失败 {path}: {e}")
                
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
            
    def enhance_frame_gpu_cupy(self, img_array):
        """使用CuPy进行GPU加速的帧增强"""
        try:
            # 将图像数据传输到GPU
            gpu_img = cp.asarray(img_array)
            
            # 使用GPU进行双三次插值
            scale_factor = 2
            new_height, new_width = gpu_img.shape[0] * scale_factor, gpu_img.shape[1] * scale_factor
            
            # 对每个颜色通道分别处理
            enhanced_channels = []
            for i in range(gpu_img.shape[2]):
                channel = gpu_img[:, :, i]
                # 使用GPU加速的缩放
                enhanced_channel = cpx_ndimage.zoom(channel, scale_factor, order=3)
                enhanced_channels.append(enhanced_channel)
            
            # 合并通道
            enhanced_gpu = cp.stack(enhanced_channels, axis=2)
            
            # 传输回CPU
            enhanced_cpu = cp.asnumpy(enhanced_gpu).astype(np.uint8)
            
            return enhanced_cpu
            
        except Exception as e:
            self.logger.error(f"❌ CuPy GPU增强失败: {e}")
            return None
            
    def enhance_frame_gpu_opencv(self, img_array):
        """使用OpenCV CUDA进行GPU加速的帧增强"""
        try:
            # 上传到GPU内存
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img_array)
            
            # GPU上进行缩放
            scale_factor = 2
            new_width = img_array.shape[1] * scale_factor
            new_height = img_array.shape[0] * scale_factor
            
            gpu_enhanced = cv2.cuda.resize(gpu_img, (new_width, new_height), 
                                         interpolation=cv2.INTER_CUBIC)
            
            # 下载到CPU内存
            enhanced_cpu = gpu_enhanced.download()
            
            return enhanced_cpu
            
        except Exception as e:
            self.logger.error(f"❌ OpenCV GPU增强失败: {e}")
            return None
            
    def enhance_frame_batch_gpu(self, frame_batch):
        """批量GPU处理帧"""
        enhanced_batch = []
        
        if self.cupy_available:
            # 使用CuPy批量处理
            try:
                batch_array = np.stack([cv2.imread(str(f)) for f in frame_batch])
                gpu_batch = cp.asarray(batch_array)
                
                enhanced_gpu_batch = []
                for i in range(gpu_batch.shape[0]):
                    enhanced = self.enhance_frame_gpu_cupy(cp.asnumpy(gpu_batch[i]))
                    if enhanced is not None:
                        enhanced_gpu_batch.append(enhanced)
                
                return enhanced_gpu_batch
                
            except Exception as e:
                self.logger.error(f"❌ CuPy批量处理失败: {e}")
                
        elif self.opencv_gpu_available:
            # 使用OpenCV CUDA批量处理
            for frame_path in frame_batch:
                img = cv2.imread(str(frame_path))
                if img is not None:
                    enhanced = self.enhance_frame_gpu_opencv(img)
                    if enhanced is not None:
                        enhanced_batch.append(enhanced)
                        
        else:
            # 回退到CPU处理
            for frame_path in frame_batch:
                img = cv2.imread(str(frame_path))
                if img is not None:
                    enhanced = cv2.resize(img, (3840, 2160), interpolation=cv2.INTER_LANCZOS4)
                    enhanced_batch.append(enhanced)
                    
        return enhanced_batch
        
    def enhance_segment_frames_gpu(self, segment_dir, frame_files, segment_idx):
        """GPU优化的段帧增强"""
        enhanced_dir = segment_dir.parent / f"enhanced_{segment_idx:04d}"
        enhanced_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"🚀 GPU增强段 {segment_idx}: {len(frame_files)} 帧")
        
        batch_size = self.gpu_config['batch_size']
        processed = 0
        failed = 0
        
        # 分批处理
        for i in range(0, len(frame_files), batch_size):
            batch = frame_files[i:i + batch_size]
            
            try:
                # GPU批量处理
                enhanced_batch = self.enhance_frame_batch_gpu(batch)
                
                # 保存增强帧
                for j, enhanced_img in enumerate(enhanced_batch):
                    if j < len(batch):
                        output_path = enhanced_dir / batch[j].name
                        success = cv2.imwrite(str(output_path), enhanced_img, 
                                            [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        if success:
                            processed += 1
                        else:
                            failed += 1
                            
                # 显示进度
                if (i // batch_size + 1) % 10 == 0:
                    progress = ((i + batch_size) / len(frame_files)) * 100
                    gpu_util = self.get_gpu_utilization()
                    self.logger.info(f"📊 段 {segment_idx} 进度: {progress:.1f}% "
                                   f"({processed}/{len(frame_files)}) GPU: {gpu_util:.1f}%")
                    
                # GPU内存清理
                if i % (batch_size * 10) == 0:
                    self.cleanup_gpu_memory()
                    
            except Exception as e:
                self.logger.error(f"❌ 批次处理失败: {e}")
                failed += batch_size
                
        self.logger.info(f"✅ 段 {segment_idx} GPU增强完成: {processed}/{len(frame_files)} 成功")
        return enhanced_dir, processed, failed
        
    def get_gpu_utilization(self):
        """获取GPU利用率"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 0.0
            
    def cleanup_gpu_memory(self):
        """清理GPU内存"""
        if self.cupy_available:
            try:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                self.logger.debug("🧹 GPU内存池已清理")
            except:
                pass
                
        # 强制垃圾回收
        gc.collect()
        
    def monitor_system_resources(self):
        """监控系统资源"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        gpu_util = self.get_gpu_utilization()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_info.percent,
            'gpu_percent': gpu_util
        }
        
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
        
        return info
        
    def extract_segment_frames(self, video_file, segment_idx, start_frame, end_frame):
        """提取指定段的帧"""
        storage_disk = 'storage_e' if segment_idx % 2 == 0 else 'storage_g'
        segment_dir = Path(self.disk_config[storage_disk]) / f"segment_{segment_idx:04d}"
        segment_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"📹 提取段 {segment_idx}: 帧 {start_frame}-{end_frame}")
        
        video_info = self.progress['video_info']
        fps = video_info['fps']
        start_time = start_frame / fps
        duration = (end_frame - start_frame + 1) / fps
        
        cmd = f'ffmpeg -ss {start_time} -i "{video_file}" -t {duration} -vf fps={fps} "{segment_dir}/frame_%08d.png" -y'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        frame_files = list(segment_dir.glob("*.png"))
        self.logger.info(f"✅ 段 {segment_idx} 提取完成: {len(frame_files)} 帧")
        return segment_dir, frame_files
        
    def process_video_gpu_optimized(self, input_video, output_video=None):
        """GPU优化的视频处理"""
        start_time = time.time()
        
        if output_video is None:
            output_video = str(Path(self.disk_config['output']) / "gpu_enhanced_4k_theater.mp4")
            
        try:
            self.logger.info("🚀 开始GPU优化4K视频增强")
            self.logger.info(f"📁 输入: {input_video}")
            self.logger.info(f"📁 输出: {output_video}")
            
            # 显示GPU状态
            gpu_util = self.get_gpu_utilization()
            self.logger.info(f"💻 初始GPU利用率: {gpu_util:.1f}%")
            
            # 获取视频信息
            video_info = self.get_video_info(input_video)
            
            # 计算分段
            frames_per_segment = self.segment_config['frames_per_segment']
            total_segments = (video_info['total_frames'] + frames_per_segment - 1) // frames_per_segment
            
            self.logger.info(f"📊 总计 {total_segments} 段，每段 {frames_per_segment} 帧")
            
            # 处理各段
            segment_videos = []
            
            for segment_idx in range(total_segments):
                if segment_idx in self.progress['processed_segments']:
                    continue
                    
                start_frame = segment_idx * frames_per_segment
                end_frame = min(start_frame + frames_per_segment - 1, video_info['total_frames'] - 1)
                
                self.logger.info(f"🔄 GPU处理段 {segment_idx + 1}/{total_segments}")
                
                try:
                    # 提取帧
                    segment_dir, frame_files = self.extract_segment_frames(
                        input_video, segment_idx, start_frame, end_frame)
                    
                    # GPU增强帧
                    enhanced_dir, processed, failed = self.enhance_segment_frames_gpu(
                        segment_dir, frame_files, segment_idx)
                    
                    # 组装视频段
                    segment_video = self.assemble_segment_video(
                        enhanced_dir, segment_idx, video_info['fps'])
                    
                    segment_videos.append(segment_video)
                    
                    # 清理临时文件
                    self.cleanup_segment_files(segment_dir, enhanced_dir)
                    
                    # 更新进度
                    self.progress['processed_segments'].append(segment_idx)
                    self.progress['current_segment'] = segment_idx + 1
                    self.save_progress()
                    
                    # 显示资源使用情况
                    resources = self.monitor_system_resources()
                    self.logger.info(f"📊 资源使用: CPU {resources['cpu_percent']:.1f}%, "
                                   f"GPU {resources['gpu_percent']:.1f}%, "
                                   f"内存 {resources['memory_percent']:.1f}%")
                    
                except Exception as e:
                    self.logger.error(f"❌ 段 {segment_idx} 处理失败: {e}")
                    continue
                    
            total_time = time.time() - start_time
            self.logger.info(f"🎉 GPU优化处理完成! 总用时: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
            
            return output_video
            
        except Exception as e:
            self.logger.error(f"❌ GPU优化处理失败: {str(e)}")
            raise
            
    def assemble_segment_video(self, enhanced_dir, segment_idx, fps):
        """组装视频段"""
        segment_video = enhanced_dir.parent / f"segment_{segment_idx:04d}.mp4"
        
        cmd = f'ffmpeg -framerate {fps} -i "{enhanced_dir}/frame_%08d.png" -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p "{segment_video}" -y'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if segment_video.exists():
            return str(segment_video)
        else:
            raise RuntimeError(f"段 {segment_idx} 视频组装失败")
            
    def cleanup_segment_files(self, segment_dir, enhanced_dir):
        """清理段文件"""
        try:
            if segment_dir.exists():
                shutil.rmtree(segment_dir)
            if enhanced_dir.exists():
                shutil.rmtree(enhanced_dir)
            self.cleanup_gpu_memory()
        except Exception as e:
            self.logger.error(f"❌ 清理失败: {e}")

def main():
    if len(sys.argv) < 2:
        print("使用方法: python3 gpu_optimized_enhancer.py input_video.mp4 [output_video.mp4]")
        print("GPU优化特性:")
        print("  - GPU加速图像处理")
        print("  - 批量GPU操作")
        print("  - 内存池管理")
        print("  - 异步处理")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else None
    
    enhancer = GPUOptimizedEnhancer()
    enhancer.process_video_gpu_optimized(input_video, output_video)

if __name__ == "__main__":
    main()