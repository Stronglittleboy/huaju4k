"""
GPU Video Super Resolver - 真实 GPU 超分核心模块

这是 Stage 4.2 的核心实现，确保：
1. nvidia-smi 显示显存占用 > 1GB
2. GPU Util 波动 30%~90%
3. 输出视频可播放且分辨率提升

设计原则：
- 逐帧处理：decode → GPU → encode
- 失败回退：GPU 失败不影响整体流程
- 进度显示：实时显示处理进度
"""

import os
import sys
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable
import time

import cv2
import numpy as np

from .model_manager import GPUModelManager

logger = logging.getLogger(__name__)


class GPUVideoSuperResolver:
    """
    GPU 视频超分辨率处理器
    
    使用 Real-ESRGAN 进行真实 GPU 超分，确保：
    - GPU 确实参与像素计算
    - 可插拔、可回退
    - 进度可视化
    """
    
    def __init__(self,
                 model_name: str = "RealESRGAN_x4plus",
                 tile_size: int = 384,
                 device: str = "cuda",
                 models_dir: str = "./models"):
        """
        初始化 GPU 超分处理器
        
        Args:
            model_name: 模型名称 ("RealESRGAN_x4plus" 或 "RealESRGAN_x2plus")
            tile_size: 瓦片大小，影响显存占用 (推荐 512 for x2, 384 for x4)
            device: 设备 ("cuda" 或 "cpu")
            models_dir: 模型存储目录
        """
        self.model_name = model_name
        self.tile_size = tile_size
        self.device = device
        
        # 初始化模型管理器
        self.model_manager = GPUModelManager(models_dir=models_dir)
        
        # 模型加载状态
        self._model_loaded = False
        
        logger.info(f"GPUVideoSuperResolver 初始化: model={model_name}, tile={tile_size}")
    
    def _ensure_model_loaded(self) -> bool:
        """确保模型已加载"""
        if self._model_loaded:
            return True
        
        success = self.model_manager.load_model(
            model_name=self.model_name,
            tile_size=self.tile_size,
            half=True  # FP16 节省显存
        )
        
        if success:
            self._model_loaded = True
            
            # 打印 GPU 状态
            stats = self.model_manager.get_gpu_stats()
            if stats.get("available"):
                print(f"\n🎮 GPU 状态:")
                print(f"   设备: {stats['device']}")
                print(f"   总显存: {stats['total_memory_mb']} MB")
                print(f"   已分配: {stats['allocated_mb']} MB")
        
        return success
    
    def enhance_video(self,
                     input_video: str,
                     output_video: str,
                     progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """
        GPU 超分增强视频
        
        处理流程：
        1. FFmpeg 解码视频帧
        2. 每帧送入 GPU 进行 Real-ESRGAN 推理
        3. FFmpeg 编码输出视频
        
        Args:
            input_video: 输入视频路径
            output_video: 输出视频路径
            progress_callback: 进度回调函数
            
        Returns:
            处理成功返回 True
        """
        # 验证输入
        if not Path(input_video).exists():
            logger.error(f"输入视频不存在: {input_video}")
            return False
        
        # 确保模型加载
        if not self._ensure_model_loaded():
            logger.error("模型加载失败，无法进行 GPU 超分")
            return False
        
        try:
            # 获取视频信息
            cap = cv2.VideoCapture(input_video)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {input_video}")
                return False
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 计算输出分辨率
            scale = 4 if "x4" in self.model_name else 2
            out_width = width * scale
            out_height = height * scale
            
            print(f"\n🚀 GPU 超分增强开始")
            print(f"   输入: {width}x{height} @ {fps:.1f}fps")
            print(f"   输出: {out_width}x{out_height}")
            print(f"   总帧数: {total_frames}")
            print(f"   模型: {self.model_name}")
            print(f"   瓦片大小: {self.tile_size}")
            print(f"\n💡 提示:")
            print(f"   - 这是离线处理，适合关键片段的高质量增强")
            print(f"   - GPU 利用率会呈波形，这是正常现象")
            print(f"   - 可在另一终端运行 'watch -n 1 nvidia-smi' 监控 GPU")
            print(f"   - 处理速度约 0.4 fps，请耐心等待\n")
            
            # 创建输出目录
            Path(output_video).parent.mkdir(parents=True, exist_ok=True)
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (out_width, out_height))
            
            if not out.isOpened():
                logger.error("无法创建输出视频")
                cap.release()
                return False
            
            # 逐帧处理
            frame_idx = 0
            start_time = time.time()
            last_print_time = start_time
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # GPU 超分处理
                try:
                    enhanced_frame = self.model_manager.enhance_frame(frame)
                except Exception as e:
                    logger.error(f"帧 {frame_idx} GPU 处理失败: {e}")
                    # 回退到简单放大
                    enhanced_frame = cv2.resize(frame, (out_width, out_height), 
                                               interpolation=cv2.INTER_LANCZOS4)
                
                # 写入输出
                out.write(enhanced_frame)
                
                frame_idx += 1
                
                # 进度显示 (每秒更新一次)
                current_time = time.time()
                if current_time - last_print_time >= 1.0 or frame_idx == total_frames:
                    last_print_time = current_time
                    progress = frame_idx / total_frames
                    elapsed = current_time - start_time
                    fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                    eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
                    
                    # 格式化剩余时间
                    days = int(eta // 86400)
                    hours = int((eta % 86400) // 3600)
                    minutes = int((eta % 3600) // 60)
                    seconds = int(eta % 60)
                    
                    if days > 0:
                        eta_str = f"{days}天{hours}小时{minutes}分{seconds}秒"
                    elif hours > 0:
                        eta_str = f"{hours}小时{minutes}分{seconds}秒"
                    elif minutes > 0:
                        eta_str = f"{minutes}分{seconds}秒"
                    else:
                        eta_str = f"{seconds}秒"
                    
                    bar_width = 40
                    filled = int(bar_width * progress)
                    bar = '█' * filled + '░' * (bar_width - filled)
                    
                    print(f"\r   进度: [{bar}] {progress*100:.1f}% "
                          f"({frame_idx}/{total_frames}) "
                          f"速度: {fps_actual:.1f} fps "
                          f"剩余: {eta_str}", end='', flush=True)
                    
                    if progress_callback:
                        progress_callback(progress)
            
            print()  # 换行
            
            # 释放资源
            cap.release()
            out.release()
            
            # 验证输出
            if not Path(output_video).exists():
                logger.error("输出视频未生成")
                return False
            
            output_size = Path(output_video).stat().st_size
            if output_size < 1000:
                logger.error(f"输出视频太小: {output_size} bytes")
                return False
            
            # 打印统计
            total_time = time.time() - start_time
            avg_fps = frame_idx / total_time if total_time > 0 else 0
            
            print(f"\n✅ GPU 超分完成!")
            print(f"   处理帧数: {frame_idx}")
            print(f"   总耗时: {total_time:.1f}s")
            print(f"   平均速度: {avg_fps:.2f} fps")
            print(f"   临时文件: {output_video}")
            print(f"   文件大小: {output_size / (1024*1024):.1f} MB")
            
            # 合并音频
            print(f"\n🔊 合并音频...")
            temp_video = output_video + ".temp.mp4"
            Path(output_video).rename(temp_video)
            
            merge_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', input_video,
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-map', '0:v:0',
                '-map', '1:a:0?',
                output_video
            ]
            
            result = subprocess.run(merge_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # 删除临时文件
                Path(temp_video).unlink()
                print(f"✅ 音频合并完成: {output_video}")
            else:
                logger.warning(f"音频合并失败，保留无音频版本: {result.stderr}")
                Path(temp_video).rename(output_video)
            
            return True
            
        except Exception as e:
            logger.error(f"GPU 超分处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def enhance_video_ffmpeg_pipe(self,
                                  input_video: str,
                                  output_video: str,
                                  progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """
        使用 FFmpeg pipe 进行 GPU 超分 (更高效的版本)
        
        流程：
        FFmpeg decode (pipe) → GPU 推理 → FFmpeg encode (pipe)
        
        Args:
            input_video: 输入视频路径
            output_video: 输出视频路径
            progress_callback: 进度回调函数
            
        Returns:
            处理成功返回 True
        """
        # 验证输入
        if not Path(input_video).exists():
            logger.error(f"输入视频不存在: {input_video}")
            return False
        
        # 确保模型加载
        if not self._ensure_model_loaded():
            logger.error("模型加载失败")
            return False
        
        try:
            # 获取视频信息
            probe_cmd = [
                'ffprobe', '-v', 'quiet',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
                '-of', 'csv=p=0',
                input_video
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            parts = result.stdout.strip().split(',')
            
            width = int(parts[0])
            height = int(parts[1])
            fps_parts = parts[2].split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
            total_frames = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
            
            # 计算输出分辨率
            scale = 4 if "x4" in self.model_name else 2
            out_width = width * scale
            out_height = height * scale
            
            print(f"\n🚀 GPU 超分增强 (FFmpeg Pipe 模式)")
            print(f"   输入: {width}x{height} @ {fps:.1f}fps")
            print(f"   输出: {out_width}x{out_height}")
            print(f"   模型: {self.model_name}")
            
            # FFmpeg 解码进程
            decode_cmd = [
                'ffmpeg', '-i', input_video,
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-v', 'quiet',
                '-'
            ]
            
            # FFmpeg 编码进程
            encode_cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{out_width}x{out_height}',
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-v', 'quiet',
                output_video
            ]
            
            # 启动进程
            decode_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE)
            encode_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)
            
            frame_size = width * height * 3
            frame_idx = 0
            start_time = time.time()
            
            while True:
                # 读取一帧
                raw_frame = decode_proc.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    break
                
                # 转换为 numpy 数组
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                
                # GPU 超分
                try:
                    enhanced_frame = self.model_manager.enhance_frame(frame)
                except Exception as e:
                    logger.warning(f"帧 {frame_idx} GPU 失败，回退到 CPU: {e}")
                    enhanced_frame = cv2.resize(frame, (out_width, out_height),
                                               interpolation=cv2.INTER_LANCZOS4)
                
                # 写入编码器
                encode_proc.stdin.write(enhanced_frame.tobytes())
                
                frame_idx += 1
                
                # 进度显示
                if frame_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                    
                    if total_frames > 0:
                        progress = frame_idx / total_frames
                        eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
                        
                        # 格式化剩余时间
                        days = int(eta // 86400)
                        hours = int((eta % 86400) // 3600)
                        minutes = int((eta % 3600) // 60)
                        seconds = int(eta % 60)
                        
                        if days > 0:
                            eta_str = f"{days}天{hours}小时{minutes}分{seconds}秒"
                        elif hours > 0:
                            eta_str = f"{hours}小时{minutes}分{seconds}秒"
                        elif minutes > 0:
                            eta_str = f"{minutes}分{seconds}秒"
                        else:
                            eta_str = f"{seconds}秒"
                        
                        bar_width = 40
                        filled = int(bar_width * progress)
                        bar = '█' * filled + '░' * (bar_width - filled)
                        print(f"\r   进度: [{bar}] {progress*100:.1f}% "
                              f"速度: {fps_actual:.1f} fps "
                              f"剩余: {eta_str}", end='', flush=True)
                    else:
                        print(f"\r   已处理: {frame_idx} 帧, 速度: {fps_actual:.1f} fps", 
                              end='', flush=True)
            
            print()
            
            # 关闭进程
            decode_proc.stdout.close()
            encode_proc.stdin.close()
            decode_proc.wait()
            encode_proc.wait()
            
            # 验证输出
            if not Path(output_video).exists():
                logger.error("输出视频未生成")
                return False
            
            total_time = time.time() - start_time
            print(f"\n✅ GPU 超分完成! 处理 {frame_idx} 帧, 耗时 {total_time:.1f}s")
            
            # 合并音频
            print(f"\n🔊 合并音频...")
            temp_video = output_video + ".temp.mp4"
            Path(output_video).rename(temp_video)
            
            merge_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', input_video,
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-map', '0:v:0',
                '-map', '1:a:0?',
                output_video
            ]
            
            result = subprocess.run(merge_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # 删除临时文件
                Path(temp_video).unlink()
                print(f"✅ 音频合并完成: {output_video}")
                return True
            else:
                logger.warning(f"音频合并失败，保留无音频版本: {result.stderr}")
                Path(temp_video).rename(output_video)
                return True
                
        except Exception as e:
            logger.error(f"FFmpeg pipe 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup(self) -> None:
        """清理资源，释放 GPU 显存"""
        self.model_manager.unload_model()
        self._model_loaded = False
        logger.info("GPU 资源已释放")


def test_gpu_stage():
    """测试 GPU Stage 是否正常工作"""
    print("=" * 60)
    print("GPU Stage 测试")
    print("=" * 60)
    
    # 检查 GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print("❌ CUDA 不可用")
            return False
    except ImportError:
        print("❌ PyTorch 未安装")
        return False
    
    # 检查 Real-ESRGAN
    try:
        from realesrgan import RealESRGANer
        print("✅ Real-ESRGAN 可用")
    except ImportError:
        print("❌ Real-ESRGAN 未安装")
        print("   请运行: pip install realesrgan basicsr")
        return False
    
    # 创建测试图像
    print("\n创建测试图像...")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 测试模型加载
    print("\n测试模型加载...")
    resolver = GPUVideoSuperResolver(
        model_name="RealESRGAN_x4plus",
        tile_size=384
    )
    
    if resolver._ensure_model_loaded():
        print("✅ 模型加载成功")
        
        # 测试单帧推理
        print("\n测试单帧 GPU 推理...")
        try:
            start = time.time()
            enhanced = resolver.model_manager.enhance_frame(test_image)
            elapsed = time.time() - start
            
            print(f"✅ GPU 推理成功!")
            print(f"   输入: {test_image.shape}")
            print(f"   输出: {enhanced.shape}")
            print(f"   耗时: {elapsed:.2f}s")
            
            # 检查 GPU 使用
            allocated = torch.cuda.memory_allocated(0) / (1024**2)
            print(f"   显存占用: {allocated:.0f} MB")
            
            return True
        except Exception as e:
            print(f"❌ GPU 推理失败: {e}")
            return False
    else:
        print("❌ 模型加载失败")
        return False


if __name__ == "__main__":
    test_gpu_stage()
