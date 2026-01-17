#!/usr/bin/env python3
"""
基于OpenCV CUDA的GPU加速器
针对现有环境优化的GPU加速实现
"""

import os
import sys
import json
import subprocess
import logging
import time
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class OpenCVGPUAccelerator:
    def __init__(self):
        self.setup_logging()
        self.setup_gpu_environment()
        
        # GPU配置
        self.gpu_config = {
            'batch_size': 8,
            'use_gpu_memory_optimization': True,
            'gpu_streams': 2,
            'fallback_to_cpu': True
        }
        
    def setup_logging(self):
        """设置日志系统"""
        log_file = f"opencv_gpu_acceleration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_gpu_environment(self):
        """设置OpenCV GPU环境"""
        self.logger.info("🔧 设置OpenCV GPU环境...")
        
        # 检查CUDA设备
        self.cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
        self.gpu_available = self.cuda_device_count > 0
        
        if self.gpu_available:
            self.logger.info(f"✅ 检测到 {self.cuda_device_count} 个CUDA设备")
            
            # 获取GPU信息
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                       '--format=csv,noheader'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_info = result.stdout.strip().split(', ')
                    self.logger.info(f"   GPU: {gpu_info[0]}")
                    self.logger.info(f"   显存: {gpu_info[1]} MB")
            except:
                pass
                
            # 设置CUDA设备
            cv2.cuda.setDevice(0)
            self.logger.info("✅ OpenCV CUDA环境设置完成")
        else:
            self.logger.warning("⚠️ 未检测到CUDA设备，将使用CPU处理")
            
    def gpu_denoise_opencv(self, image):
        """使用OpenCV CUDA进行降噪"""
        if not self.gpu_available:
            return cv2.bilateralFilter(image, 9, 75, 75)
            
        try:
            # 上传到GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            
            # GPU双边滤波降噪
            gpu_result = cv2.cuda.bilateralFilter(gpu_img, -1, 50, 50)
            
            # 下载到CPU
            result = gpu_result.download()
            return result
            
        except Exception as e:
            self.logger.warning(f"GPU降噪失败，使用CPU: {e}")
            return cv2.bilateralFilter(image, 9, 75, 75)
            
    def gpu_upscale_opencv(self, image, scale_factor=2):
        """使用OpenCV CUDA进行放大"""
        if not self.gpu_available:
            height, width = image.shape[:2]
            new_size = (width * scale_factor, height * scale_factor)
            return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
            
        try:
            # 上传到GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            
            # GPU缩放
            height, width = image.shape[:2]
            new_size = (width * scale_factor, height * scale_factor)
            gpu_result = cv2.cuda.resize(gpu_img, new_size, interpolation=cv2.INTER_CUBIC)
            
            # 下载到CPU
            result = gpu_result.download()
            return result
            
        except Exception as e:
            self.logger.warning(f"GPU放大失败，使用CPU: {e}")
            height, width = image.shape[:2]
            new_size = (width * scale_factor, height * scale_factor)
            return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
            
    def gpu_enhance_image(self, image, operations=['upscale']):
        """GPU图像增强"""
        result = image.copy()
        
        for operation in operations:
            if operation == 'denoise':
                result = self.gpu_denoise_opencv(result)
            elif operation == 'upscale':
                result = self.gpu_upscale_opencv(result, 2)
            elif operation == 'sharpen':
                result = self.gpu_sharpen_opencv(result)
                
        return result
        
    def gpu_sharpen_opencv(self, image):
        """使用OpenCV CUDA进行锐化"""
        if not self.gpu_available:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            return cv2.filter2D(image, -1, kernel)
            
        try:
            # 上传到GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            
            # 创建锐化核
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
            
            # GPU滤波
            gpu_result = cv2.cuda.filter2D(gpu_img, -1, kernel)
            
            # 下载到CPU
            result = gpu_result.download()
            return result
            
        except Exception as e:
            self.logger.warning(f"GPU锐化失败，使用CPU: {e}")
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            return cv2.filter2D(image, -1, kernel)
            
    def process_frames_batch_gpu(self, frame_files, output_dir, operations=['upscale']):
        """GPU批量处理帧"""
        self.logger.info(f"🚀 GPU批量处理 {len(frame_files)} 帧...")
        
        processed_count = 0
        failed_count = 0
        
        for i, frame_file in enumerate(frame_files):
            try:
                # 加载图像
                image = cv2.imread(str(frame_file))
                if image is None:
                    failed_count += 1
                    continue
                
                # GPU增强处理
                enhanced = self.gpu_enhance_image(image, operations)
                
                # 保存结果
                output_file = Path(output_dir) / frame_file.name
                success = cv2.imwrite(str(output_file), enhanced)
                
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
                    
                # 进度显示
                if (i + 1) % 10 == 0:
                    progress = ((i + 1) / len(frame_files)) * 100
                    gpu_util = self.get_gpu_utilization()
                    self.logger.info(f"📊 GPU批量处理进度: {progress:.1f}% "
                                   f"({processed_count}/{len(frame_files)}) "
                                   f"GPU: {gpu_util['gpu_utilization']:.1f}%")
                    
            except Exception as e:
                self.logger.error(f"处理帧失败 {frame_file}: {e}")
                failed_count += 1
                
        return processed_count, failed_count
        
    def process_video_frames_directory(self, frames_dir, output_dir, operations=['upscale']):
        """处理视频帧目录"""
        start_time = time.time()
        
        self.logger.info("🎬 开始GPU加速视频帧处理...")
        self.logger.info(f"输入目录: {frames_dir}")
        self.logger.info(f"输出目录: {output_dir}")
        self.logger.info(f"操作序列: {operations}")
        
        # 获取所有帧文件
        frame_files = sorted(Path(frames_dir).glob("*.png"))
        total_frames = len(frame_files)
        
        if total_frames == 0:
            raise ValueError(f"在 {frames_dir} 中未找到PNG帧文件")
            
        self.logger.info(f"找到 {total_frames} 个帧文件")
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 分批处理
        batch_size = self.gpu_config['batch_size']
        total_processed = 0
        total_failed = 0
        
        for i in range(0, total_frames, batch_size):
            batch_files = frame_files[i:i + batch_size]
            
            self.logger.info(f"🔄 处理批次 {i//batch_size + 1}/{(total_frames + batch_size - 1)//batch_size}")
            
            # GPU批量处理
            processed, failed = self.process_frames_batch_gpu(batch_files, output_dir, operations)
            
            total_processed += processed
            total_failed += failed
            
            # 批次完成报告
            batch_progress = ((i + len(batch_files)) / total_frames) * 100
            elapsed = time.time() - start_time
            fps = total_processed / elapsed if elapsed > 0 else 0
            
            self.logger.info(f"📊 总进度: {batch_progress:.1f}% "
                           f"成功: {total_processed} 失败: {total_failed} "
                           f"速度: {fps:.1f} fps")
        
        # 最终统计
        total_time = time.time() - start_time
        success_rate = (total_processed / total_frames) * 100 if total_frames > 0 else 0
        avg_fps = total_processed / total_time if total_time > 0 else 0
        
        # 生成处理报告
        report = {
            "task": "11.1 GPU-accelerated algorithm integration",
            "processing_timestamp": datetime.now().isoformat(),
            "input_directory": str(frames_dir),
            "output_directory": str(output_dir),
            "operations": operations,
            "statistics": {
                "total_frames": total_frames,
                "processed_frames": total_processed,
                "failed_frames": total_failed,
                "success_rate_percent": success_rate,
                "processing_time_seconds": total_time,
                "average_fps": avg_fps
            },
            "gpu_info": {
                "cuda_devices": self.cuda_device_count,
                "gpu_available": self.gpu_available,
                "final_gpu_utilization": self.get_gpu_utilization()
            },
            "processing_successful": total_processed > 0
        }
        
        # 保存报告
        with open("task_11_1_gpu_acceleration_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        self.logger.info("✅ GPU加速处理完成!")
        self.logger.info(f"   总帧数: {total_frames}")
        self.logger.info(f"   成功处理: {total_processed}")
        self.logger.info(f"   失败: {total_failed}")
        self.logger.info(f"   成功率: {success_rate:.1f}%")
        self.logger.info(f"   总用时: {total_time:.1f}秒")
        self.logger.info(f"   平均速度: {avg_fps:.1f} fps")
        
        return report
        
    def get_gpu_utilization(self):
        """获取GPU利用率"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
                return {
                    'gpu_utilization': float(gpu_util),
                    'memory_used_mb': float(mem_used),
                    'memory_total_mb': float(mem_total),
                    'memory_utilization': (float(mem_used) / float(mem_total)) * 100
                }
        except:
            pass
        return {'gpu_utilization': 0, 'memory_utilization': 0}
        
    def benchmark_gpu_vs_cpu(self, test_image_path, iterations=5):
        """GPU vs CPU性能对比测试"""
        self.logger.info("🏁 开始GPU vs CPU性能对比测试...")
        
        # 加载测试图像
        test_img = cv2.imread(test_image_path)
        if test_img is None:
            raise ValueError(f"无法加载测试图像: {test_image_path}")
            
        self.logger.info(f"测试图像大小: {test_img.shape}")
        
        results = {
            'test_image_size': test_img.shape,
            'iterations': iterations,
            'gpu_available': self.gpu_available,
            'operations': {}
        }
        
        # 测试操作
        operations = {
            'upscale_2x': {
                'gpu': lambda img: self.gpu_upscale_opencv(img, 2),
                'cpu': lambda img: cv2.resize(img, (img.shape[1]*2, img.shape[0]*2), interpolation=cv2.INTER_LANCZOS4)
            },
            'denoise': {
                'gpu': lambda img: self.gpu_denoise_opencv(img),
                'cpu': lambda img: cv2.bilateralFilter(img, 9, 75, 75)
            }
        }
        
        for op_name, op_funcs in operations.items():
            self.logger.info(f"测试操作: {op_name}")
            
            # GPU测试
            gpu_times = []
            if self.gpu_available:
                for i in range(iterations):
                    start_time = time.time()
                    result = op_funcs['gpu'](test_img.copy())
                    end_time = time.time()
                    gpu_times.append(end_time - start_time)
                    
            # CPU测试
            cpu_times = []
            for i in range(iterations):
                start_time = time.time()
                result = op_funcs['cpu'](test_img.copy())
                end_time = time.time()
                cpu_times.append(end_time - start_time)
            
            # 计算统计
            results['operations'][op_name] = {
                'gpu': {
                    'avg_time': np.mean(gpu_times) if gpu_times else 0,
                    'min_time': np.min(gpu_times) if gpu_times else 0,
                    'max_time': np.max(gpu_times) if gpu_times else 0
                } if self.gpu_available else None,
                'cpu': {
                    'avg_time': np.mean(cpu_times),
                    'min_time': np.min(cpu_times),
                    'max_time': np.max(cpu_times)
                },
                'speedup': np.mean(cpu_times) / np.mean(gpu_times) if gpu_times else 0
            }
            
            if self.gpu_available and gpu_times:
                speedup = np.mean(cpu_times) / np.mean(gpu_times)
                self.logger.info(f"  {op_name} - GPU: {np.mean(gpu_times):.3f}s, CPU: {np.mean(cpu_times):.3f}s, 加速比: {speedup:.1f}x")
            else:
                self.logger.info(f"  {op_name} - CPU: {np.mean(cpu_times):.3f}s")
        
        # 保存基准测试结果
        with open("gpu_vs_cpu_benchmark.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info("✅ GPU vs CPU性能对比测试完成")
        return results

def main():
    if len(sys.argv) < 3:
        print("使用方法: python3 opencv_gpu_accelerator.py <operation> <input_path> [output_path]")
        print("操作选项:")
        print("  process - 处理视频帧目录")
        print("  benchmark - GPU vs CPU性能对比")
        print("  test - 测试单张图像")
        sys.exit(1)
    
    operation = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    accelerator = OpenCVGPUAccelerator()
    
    if operation == "process":
        if not output_path:
            output_path = "gpu_enhanced_frames"
        operations = ['upscale']  # 可以修改为 ['denoise', 'upscale', 'sharpen']
        accelerator.process_video_frames_directory(input_path, output_path, operations)
        
    elif operation == "benchmark":
        accelerator.benchmark_gpu_vs_cpu(input_path)
        
    elif operation == "test":
        # 测试单张图像
        test_img = cv2.imread(input_path)
        if test_img is not None:
            print("测试GPU图像增强...")
            enhanced = accelerator.gpu_enhance_image(test_img, ['upscale'])
            if output_path:
                cv2.imwrite(output_path, enhanced)
                print(f"✅ 增强图像已保存: {output_path}")
            else:
                print("✅ GPU图像增强测试完成")
        else:
            print("❌ 无法加载测试图像")

if __name__ == "__main__":
    main()