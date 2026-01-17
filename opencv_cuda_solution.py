#!/usr/bin/env python3
"""
OpenCV CUDA问题解决方案
提供多种GPU加速方法和CPU优化方案
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

class OpenCVCUDASolution:
    def __init__(self):
        self.setup_logging()
        self.diagnose_opencv_cuda()
        self.setup_alternative_acceleration()
        
    def setup_logging(self):
        """设置日志系统"""
        log_file = f"opencv_cuda_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def diagnose_opencv_cuda(self):
        """诊断OpenCV CUDA问题"""
        self.logger.info("🔍 诊断OpenCV CUDA支持状态...")
        
        # 检查OpenCV版本
        opencv_version = cv2.__version__
        self.logger.info(f"OpenCV版本: {opencv_version}")
        
        # 检查CUDA设备
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        self.logger.info(f"OpenCV检测到的CUDA设备数: {cuda_devices}")
        
        # 检查系统CUDA
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("✅ 系统CUDA驱动正常")
                self.cuda_available = True
            else:
                self.logger.warning("⚠️ 系统CUDA驱动异常")
                self.cuda_available = False
        except:
            self.logger.warning("⚠️ 无法检测CUDA驱动")
            self.cuda_available = False
            
        # 检查CUDA编译器
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("✅ CUDA编译器可用")
                self.nvcc_available = True
            else:
                self.nvcc_available = False
        except:
            self.logger.info("ℹ️ CUDA编译器不可用")
            self.nvcc_available = False
            
        # 诊断结果
        self.opencv_cuda_available = cuda_devices > 0
        
        if not self.opencv_cuda_available:
            self.logger.warning("❌ OpenCV CUDA支持不可用")
            self.logger.info("原因分析:")
            self.logger.info("  - pip安装的OpenCV包通常不包含CUDA支持")
            self.logger.info("  - 需要从源码编译或使用特殊版本")
            self.provide_cuda_solutions()
        else:
            self.logger.info("✅ OpenCV CUDA支持可用")
            
    def provide_cuda_solutions(self):
        """提供CUDA解决方案"""
        self.logger.info("🛠️ OpenCV CUDA解决方案:")
        self.logger.info("")
        self.logger.info("方案1: 使用conda安装 (推荐)")
        self.logger.info("  conda install -c conda-forge opencv")
        self.logger.info("")
        self.logger.info("方案2: 使用预编译的CUDA版本")
        self.logger.info("  pip install opencv-python==4.5.5.64")
        self.logger.info("  (某些版本可能包含CUDA支持)")
        self.logger.info("")
        self.logger.info("方案3: 从源码编译 (最可靠)")
        self.logger.info("  详见编译脚本: compile_opencv_cuda.sh")
        self.logger.info("")
        self.logger.info("方案4: 使用替代GPU加速方案 (当前实现)")
        self.logger.info("  - 多进程CPU并行处理")
        self.logger.info("  - 优化的内存管理")
        self.logger.info("  - 智能批处理")
        
    def setup_alternative_acceleration(self):
        """设置替代加速方案"""
        self.logger.info("⚡ 设置替代GPU加速方案...")
        
        # CPU核心数
        self.cpu_cores = mp.cpu_count()
        self.logger.info(f"CPU核心数: {self.cpu_cores}")
        
        # 优化配置
        self.config = {
            'max_workers': min(self.cpu_cores, 8),  # 限制最大工作进程
            'batch_size': 16,  # 批处理大小
            'memory_limit_mb': 2048,  # 内存限制
            'use_multiprocessing': True,  # 使用多进程
            'chunk_size': 4,  # 块大小
        }
        
        self.logger.info(f"优化配置: {self.config}")
        
    def optimized_cpu_upscale(self, image, scale_factor=2):
        """优化的CPU放大算法"""
        height, width = image.shape[:2]
        new_size = (width * scale_factor, height * scale_factor)
        
        # 使用最高质量的插值算法
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
        
    def optimized_cpu_denoise(self, image):
        """优化的CPU降噪算法"""
        # 使用非局部均值降噪，效果更好但计算量大
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
    def optimized_cpu_sharpen(self, image):
        """优化的CPU锐化算法"""
        # 使用Unsharp Mask锐化
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
    def process_single_image(self, args):
        """处理单张图像 (用于多进程)"""
        input_path, output_path, operations = args
        
        try:
            # 加载图像
            image = cv2.imread(str(input_path))
            if image is None:
                return False, f"无法加载图像: {input_path}"
                
            # 应用操作
            result = image.copy()
            for operation in operations:
                if operation == 'upscale':
                    result = self.optimized_cpu_upscale(result, 2)
                elif operation == 'denoise':
                    result = self.optimized_cpu_denoise(result)
                elif operation == 'sharpen':
                    result = self.optimized_cpu_sharpen(result)
                    
            # 保存结果
            success = cv2.imwrite(str(output_path), result)
            return success, None
            
        except Exception as e:
            return False, str(e)
            
    def process_frames_parallel(self, frame_files, output_dir, operations=['upscale']):
        """并行处理帧文件"""
        start_time = time.time()
        
        self.logger.info(f"🚀 开始并行处理 {len(frame_files)} 帧...")
        self.logger.info(f"使用 {self.config['max_workers']} 个工作进程")
        self.logger.info(f"操作序列: {operations}")
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 准备任务参数
        tasks = []
        for frame_file in frame_files:
            output_file = Path(output_dir) / frame_file.name
            tasks.append((frame_file, output_file, operations))
            
        # 并行处理
        processed_count = 0
        failed_count = 0
        
        with ProcessPoolExecutor(max_workers=self.config['max_workers']) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(self.process_single_image, task): task for task in tasks}
            
            # 处理结果
            for i, future in enumerate(as_completed(future_to_task)):
                task = future_to_task[future]
                try:
                    success, error = future.result()
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1
                        if error:
                            self.logger.warning(f"处理失败: {task[0]} - {error}")
                            
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"任务异常: {task[0]} - {e}")
                    
                # 进度报告
                if (i + 1) % 20 == 0:
                    progress = ((i + 1) / len(tasks)) * 100
                    elapsed = time.time() - start_time
                    fps = processed_count / elapsed if elapsed > 0 else 0
                    
                    self.logger.info(f"📊 处理进度: {progress:.1f}% "
                                   f"成功: {processed_count} 失败: {failed_count} "
                                   f"速度: {fps:.1f} fps")
                                   
        # 最终统计
        total_time = time.time() - start_time
        success_rate = (processed_count / len(frame_files)) * 100 if frame_files else 0
        avg_fps = processed_count / total_time if total_time > 0 else 0
        
        # 生成报告
        report = {
            "task": "OpenCV CUDA Solution - Parallel Processing",
            "processing_timestamp": datetime.now().isoformat(),
            "input_frames": len(frame_files),
            "output_directory": str(output_dir),
            "operations": operations,
            "configuration": self.config,
            "results": {
                "total_frames": len(frame_files),
                "processed_frames": processed_count,
                "failed_frames": failed_count,
                "success_rate_percent": success_rate,
                "processing_time_seconds": total_time,
                "average_fps": avg_fps
            },
            "system_info": {
                "cpu_cores": self.cpu_cores,
                "opencv_version": cv2.__version__,
                "opencv_cuda_available": self.opencv_cuda_available,
                "system_cuda_available": self.cuda_available
            },
            "performance_analysis": {
                "cpu_utilization_estimate": f"{self.config['max_workers']}/{self.cpu_cores} cores",
                "memory_efficiency": "Optimized batch processing",
                "algorithm_quality": "LANCZOS4 upscaling, NLM denoising"
            }
        }
        
        # 保存报告
        with open("opencv_cuda_solution_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        self.logger.info("✅ 并行处理完成!")
        self.logger.info(f"   总帧数: {len(frame_files)}")
        self.logger.info(f"   成功处理: {processed_count}")
        self.logger.info(f"   失败: {failed_count}")
        self.logger.info(f"   成功率: {success_rate:.1f}%")
        self.logger.info(f"   总用时: {total_time:.1f}秒")
        self.logger.info(f"   平均速度: {avg_fps:.1f} fps")
        
        return report
        
    def create_cuda_compile_script(self):
        """创建CUDA编译脚本"""
        script_content = '''#!/bin/bash
# OpenCV CUDA编译脚本
# 用于从源码编译支持CUDA的OpenCV

echo "🚀 开始编译支持CUDA的OpenCV..."

# 检查CUDA环境
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA编译器未找到，请先安装CUDA Toolkit"
    exit 1
fi

# 安装依赖
echo "📦 安装编译依赖..."
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config
sudo apt-get install -y libjpeg-dev libtiff5-dev libpng-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk-3-dev
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt-get install -y python3-dev python3-numpy

# 下载OpenCV源码
echo "📥 下载OpenCV源码..."
cd /tmp
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# 创建编译目录
cd opencv
mkdir build
cd build

# 配置编译选项
echo "⚙️ 配置编译选项..."
cmake -D CMAKE_BUILD_TYPE=RELEASE \\
    -D CMAKE_INSTALL_PREFIX=/usr/local \\
    -D INSTALL_PYTHON_EXAMPLES=ON \\
    -D INSTALL_C_EXAMPLES=OFF \\
    -D OPENCV_ENABLE_NONFREE=ON \\
    -D WITH_CUDA=ON \\
    -D WITH_CUDNN=ON \\
    -D OPENCV_DNN_CUDA=ON \\
    -D ENABLE_FAST_MATH=1 \\
    -D CUDA_FAST_MATH=1 \\
    -D CUDA_ARCH_BIN=6.1 \\
    -D WITH_CUBLAS=1 \\
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \\
    -D HAVE_opencv_python3=ON \\
    -D PYTHON_EXECUTABLE=/usr/bin/python3 \\
    -D BUILD_EXAMPLES=ON ..

# 编译 (使用所有CPU核心)
echo "🔨 开始编译 (这可能需要1-2小时)..."
make -j$(nproc)

# 安装
echo "📦 安装OpenCV..."
sudo make install
sudo ldconfig

# 验证安装
echo "🧪 验证CUDA支持..."
python3 -c "import cv2; print('OpenCV版本:', cv2.__version__); print('CUDA设备:', cv2.cuda.getCudaEnabledDeviceCount())"

echo "✅ OpenCV CUDA编译完成!"
'''
        
        with open("compile_opencv_cuda.sh", "w") as f:
            f.write(script_content)
            
        # 设置执行权限
        os.chmod("compile_opencv_cuda.sh", 0o755)
        
        self.logger.info("📝 已创建CUDA编译脚本: compile_opencv_cuda.sh")
        self.logger.info("   执行方法: ./compile_opencv_cuda.sh")
        
    def benchmark_performance(self, test_image_path):
        """性能基准测试"""
        self.logger.info("🏁 开始性能基准测试...")
        
        # 加载测试图像
        test_img = cv2.imread(test_image_path)
        if test_img is None:
            raise ValueError(f"无法加载测试图像: {test_image_path}")
            
        self.logger.info(f"测试图像大小: {test_img.shape}")
        
        # 测试不同算法
        algorithms = {
            'upscale_lanczos4': lambda img: cv2.resize(img, (img.shape[1]*2, img.shape[0]*2), interpolation=cv2.INTER_LANCZOS4),
            'upscale_cubic': lambda img: cv2.resize(img, (img.shape[1]*2, img.shape[0]*2), interpolation=cv2.INTER_CUBIC),
            'denoise_bilateral': lambda img: cv2.bilateralFilter(img, 9, 75, 75),
            'denoise_nlm': lambda img: cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21),
        }
        
        results = {}
        iterations = 3
        
        for name, func in algorithms.items():
            self.logger.info(f"测试算法: {name}")
            times = []
            
            for i in range(iterations):
                start_time = time.time()
                result = func(test_img.copy())
                end_time = time.time()
                times.append(end_time - start_time)
                
            avg_time = np.mean(times)
            results[name] = {
                'avg_time_seconds': avg_time,
                'fps_equivalent': 1.0 / avg_time if avg_time > 0 else 0
            }
            
            self.logger.info(f"  平均用时: {avg_time:.3f}秒, 等效FPS: {1.0/avg_time:.1f}")
            
        # 保存基准测试结果
        benchmark_report = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "test_image_shape": test_img.shape,
            "iterations": iterations,
            "algorithms": results,
            "system_info": {
                "cpu_cores": self.cpu_cores,
                "opencv_version": cv2.__version__,
                "opencv_cuda_available": self.opencv_cuda_available
            }
        }
        
        with open("performance_benchmark.json", 'w', encoding='utf-8') as f:
            json.dump(benchmark_report, f, indent=2, ensure_ascii=False)
            
        self.logger.info("✅ 性能基准测试完成")
        return benchmark_report

def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python3 opencv_cuda_solution.py diagnose")
        print("  python3 opencv_cuda_solution.py process <frames_dir> [output_dir]")
        print("  python3 opencv_cuda_solution.py benchmark <test_image>")
        print("  python3 opencv_cuda_solution.py compile_script")
        sys.exit(1)
        
    command = sys.argv[1]
    solution = OpenCVCUDASolution()
    
    if command == "diagnose":
        # 仅诊断，已在初始化时完成
        pass
        
    elif command == "process":
        if len(sys.argv) < 3:
            print("错误: 需要指定帧目录")
            sys.exit(1)
            
        frames_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "optimized_frames"
        
        # 获取帧文件
        frame_files = sorted(Path(frames_dir).glob("*.png"))
        if not frame_files:
            print(f"错误: 在 {frames_dir} 中未找到PNG文件")
            sys.exit(1)
            
        # 处理帧
        operations = ['upscale']  # 可修改为 ['denoise', 'upscale', 'sharpen']
        solution.process_frames_parallel(frame_files, output_dir, operations)
        
    elif command == "benchmark":
        if len(sys.argv) < 3:
            print("错误: 需要指定测试图像")
            sys.exit(1)
            
        test_image = sys.argv[2]
        solution.benchmark_performance(test_image)
        
    elif command == "compile_script":
        solution.create_cuda_compile_script()
        
    else:
        print(f"未知命令: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()