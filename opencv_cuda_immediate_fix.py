#!/usr/bin/env python3
"""
OpenCV CUDA立即修复方案
在CUDA编译完成前提供优化的CPU处理方案
"""

import cv2
import numpy as np
import json
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('opencv_cuda_immediate_fix.log'),
        logging.StreamHandler()
    ]
)

class OpenCVCUDAImmediateFix:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.diagnose_system()
        self.setup_optimization()
        
    def diagnose_system(self):
        """系统诊断"""
        self.logger.info("🔍 系统诊断...")
        
        # OpenCV状态
        self.opencv_version = cv2.__version__
        self.cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        
        self.logger.info(f"OpenCV版本: {self.opencv_version}")
        self.logger.info(f"CUDA设备数: {self.cuda_devices}")
        
        # 系统资源
        self.cpu_cores = mp.cpu_count()
        
        # CUDA编译状态检查
        self.check_cuda_compilation_status()
        
    def check_cuda_compilation_status(self):
        """检查CUDA编译状态"""
        if self.cuda_devices > 0:
            self.logger.info("✅ OpenCV CUDA支持已可用")
            self.cuda_available = True
        else:
            self.logger.info("❌ OpenCV CUDA支持不可用")
            self.logger.info("🔨 检查是否有CUDA编译进程运行...")
            
            # 检查编译进程
            import subprocess
            try:
                result = subprocess.run(['pgrep', '-f', 'compile_opencv'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    self.logger.info("✅ 检测到OpenCV CUDA编译进程正在运行")
                    self.logger.info("⏳ 编译完成后将自动获得CUDA支持")
                else:
                    self.logger.info("📝 未检测到编译进程，可手动执行: ./compile_opencv_cuda_final.sh")
            except:
                pass
                
            self.cuda_available = False
            
    def setup_optimization(self):
        """设置优化配置"""
        if self.cuda_available:
            self.logger.info("🚀 使用CUDA加速")
            self.processing_mode = "CUDA"
        else:
            self.logger.info("⚡ 使用CPU多进程优化")
            self.processing_mode = "CPU_OPTIMIZED"
            
        # CPU优化配置
        self.max_workers = min(self.cpu_cores, 8)  # 限制最大进程数
        self.batch_size = 4  # 批处理大小
        
        self.logger.info(f"CPU核心: {self.cpu_cores}")
        self.logger.info(f"最大工作进程: {self.max_workers}")
        self.logger.info(f"批处理大小: {self.batch_size}")
        
    def process_frame_cpu_optimized(self, args):
        """CPU优化的帧处理"""
        input_path, output_path, operations = args
        
        try:
            # 读取图像
            image = cv2.imread(str(input_path))
            if image is None:
                return False, f"无法读取: {input_path}"
                
            result = image.copy()
            
            # 执行操作
            for op in operations:
                if op == 'upscale_2x':
                    h, w = result.shape[:2]
                    result = cv2.resize(result, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
                elif op == 'upscale_4x':
                    h, w = result.shape[:2]
                    result = cv2.resize(result, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
                elif op == 'denoise':
                    result = cv2.bilateralFilter(result, 9, 75, 75)
                elif op == 'sharpen':
                    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    result = cv2.filter2D(result, -1, kernel)
                elif op == 'enhance':
                    # 综合增强
                    result = cv2.bilateralFilter(result, 5, 50, 50)  # 降噪
                    kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
                    result = cv2.filter2D(result, -1, kernel)  # 锐化
                    
            # 保存结果
            success = cv2.imwrite(str(output_path), result)
            return success, None
            
        except Exception as e:
            return False, str(e)
            
    def process_frame_cuda(self, args):
        """CUDA加速的帧处理"""
        input_path, output_path, operations = args
        
        try:
            # 读取图像到GPU
            image = cv2.imread(str(input_path))
            if image is None:
                return False, f"无法读取: {input_path}"
                
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)
            
            # GPU操作
            for op in operations:
                if op == 'upscale_2x':
                    h, w = gpu_image.size()
                    gpu_image = cv2.cuda.resize(gpu_image, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
                elif op == 'upscale_4x':
                    h, w = gpu_image.size()
                    gpu_image = cv2.cuda.resize(gpu_image, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
                elif op == 'denoise':
                    gpu_image = cv2.cuda.bilateralFilter(gpu_image, -1, 50, 50)
                    
            # 下载结果
            result = gpu_image.download()
            success = cv2.imwrite(str(output_path), result)
            return success, None
            
        except Exception as e:
            return False, str(e)
            
    def process_video_frames(self, input_dir, output_dir, operations=['upscale_2x']):
        """处理视频帧"""
        start_time = time.time()
        
        self.logger.info(f"🚀 开始处理视频帧...")
        self.logger.info(f"输入目录: {input_dir}")
        self.logger.info(f"输出目录: {output_dir}")
        self.logger.info(f"处理模式: {self.processing_mode}")
        self.logger.info(f"操作: {operations}")
        
        # 获取帧文件
        input_path = Path(input_dir)
        frame_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
        
        if not frame_files:
            self.logger.error(f"未找到图像文件: {input_dir}")
            return False
            
        total_frames = len(frame_files)
        self.logger.info(f"找到 {total_frames} 个帧文件")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 准备任务
        tasks = []
        for frame_file in frame_files:
            output_file = output_path / frame_file.name
            tasks.append((frame_file, output_file, operations))
            
        # 选择处理函数
        if self.cuda_available:
            process_func = self.process_frame_cuda
        else:
            process_func = self.process_frame_cpu_optimized
            
        # 并行处理
        processed = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(process_func, task): task for task in tasks}
            
            for i, future in enumerate(as_completed(future_to_task)):
                try:
                    success, error = future.result()
                    if success:
                        processed += 1
                    else:
                        failed += 1
                        if error:
                            self.logger.error(f"处理失败: {error}")
                except Exception as e:
                    failed += 1
                    self.logger.error(f"任务异常: {e}")
                    
                # 进度报告
                if (i + 1) % 10 == 0 or i == len(tasks) - 1:
                    progress = ((i + 1) / len(tasks)) * 100
                    elapsed = time.time() - start_time
                    fps = processed / elapsed if elapsed > 0 else 0
                    self.logger.info(f"📊 进度: {progress:.1f}% 成功: {processed} 失败: {failed} 速度: {fps:.1f} fps")
                    
        # 最终统计
        total_time = time.time() - start_time
        success_rate = (processed / total_frames) * 100 if total_frames > 0 else 0
        avg_fps = processed / total_time if total_time > 0 else 0
        
        self.logger.info(f"✅ 处理完成!")
        self.logger.info(f"成功: {processed}/{total_frames}")
        self.logger.info(f"成功率: {success_rate:.1f}%")
        self.logger.info(f"总用时: {total_time:.1f}秒")
        self.logger.info(f"平均速度: {avg_fps:.1f} fps")
        
        # 生成报告
        report = {
            "task": "OpenCV CUDA Immediate Fix",
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "opencv_version": self.opencv_version,
                "cuda_devices": self.cuda_devices,
                "processing_mode": self.processing_mode,
                "cpu_cores": self.cpu_cores
            },
            "processing_results": {
                "total_frames": total_frames,
                "processed_frames": processed,
                "failed_frames": failed,
                "success_rate_percent": success_rate,
                "processing_time_seconds": total_time,
                "average_fps": avg_fps
            },
            "configuration": {
                "max_workers": self.max_workers,
                "batch_size": self.batch_size,
                "operations": operations
            }
        }
        
        report_path = f"opencv_cuda_immediate_fix_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"📋 报告已保存: {report_path}")
        return True
        
    def test_processing(self):
        """测试处理功能"""
        self.logger.info("🧪 测试处理功能...")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_path = "test_frame.png"
        cv2.imwrite(test_path, test_image)
        
        # 测试处理
        if self.cuda_available:
            success, error = self.process_frame_cuda((test_path, "test_output_cuda.png", ['upscale_2x']))
        else:
            success, error = self.process_frame_cpu_optimized((test_path, "test_output_cpu.png", ['upscale_2x']))
            
        if success:
            self.logger.info("✅ 处理功能测试成功")
        else:
            self.logger.error(f"❌ 处理功能测试失败: {error}")
            
        # 清理测试文件
        try:
            Path(test_path).unlink()
            if success:
                if self.cuda_available:
                    Path("test_output_cuda.png").unlink()
                else:
                    Path("test_output_cpu.png").unlink()
        except:
            pass
            
        return success

def main():
    print("🚀 OpenCV CUDA立即修复方案")
    print("=" * 50)
    
    # 初始化修复方案
    fix = OpenCVCUDAImmediateFix()
    
    # 测试处理功能
    test_success = fix.test_processing()
    
    if not test_success:
        print("❌ 处理功能测试失败，请检查系统配置")
        return False
        
    print("\n📋 当前状态:")
    print(f"   OpenCV版本: {fix.opencv_version}")
    print(f"   CUDA设备: {fix.cuda_devices}")
    print(f"   处理模式: {fix.processing_mode}")
    print(f"   CPU核心: {fix.cpu_cores}")
    print(f"   最大工作进程: {fix.max_workers}")
    
    print("\n🛠️ 使用方法:")
    print("   fix.process_video_frames('input_frames', 'output_frames', ['upscale_2x'])")
    
    if fix.cuda_available:
        print("\n✅ CUDA加速可用 - 处理速度将显著提升")
    else:
        print("\n⚡ 使用CPU优化处理 - 多进程并行加速")
        print("   如需CUDA支持，请等待编译完成或手动执行:")
        print("   ./compile_opencv_cuda_final.sh")
        
    return fix

if __name__ == "__main__":
    fix = main()