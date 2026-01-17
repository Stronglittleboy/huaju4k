#!/usr/bin/env python3
"""
快速OpenCV CUDA修复方案
直接安装和测试支持CUDA的OpenCV版本
"""

import subprocess
import sys
import os
import cv2
import numpy as np
from datetime import datetime

def run_command(cmd):
    """执行命令"""
    print(f"执行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(f"输出: {result.stdout.strip()}")
        if result.stderr and result.returncode != 0:
            print(f"错误: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"命令失败: {e}")
        return False

def check_current_opencv():
    """检查当前OpenCV状态"""
    print("🔍 检查当前OpenCV状态...")
    print(f"OpenCV版本: {cv2.__version__}")
    
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"CUDA设备数: {cuda_devices}")
    
    return cuda_devices > 0

def test_cuda_operations():
    """测试CUDA操作"""
    print("🧪 测试CUDA操作...")
    
    try:
        # 创建测试图像
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"测试图像大小: {test_img.shape}")
        
        # 尝试CUDA操作
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(test_img)
        
        # 转换为灰度
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        
        # 下载结果
        result = gpu_gray.download()
        
        print("✅ CUDA操作测试成功")
        print(f"结果图像大小: {result.shape}")
        return True
        
    except Exception as e:
        print(f"❌ CUDA操作失败: {e}")
        return False

def install_opencv_versions():
    """尝试安装不同版本的OpenCV"""
    print("📦 尝试安装支持CUDA的OpenCV版本...")
    
    # 卸载现有版本
    uninstall_commands = [
        "pip3 uninstall -y opencv-python",
        "pip3 uninstall -y opencv-contrib-python", 
        "pip3 uninstall -y opencv-python-headless"
    ]
    
    for cmd in uninstall_commands:
        run_command(cmd)
    
    # 尝试不同版本
    versions = [
        "opencv-contrib-python==4.5.5.64",
        "opencv-python==4.6.0.66", 
        "opencv-contrib-python==4.8.1.78"
    ]
    
    for version in versions:
        print(f"\n尝试安装: {version}")
        if run_command(f"pip3 install {version}"):
            print("安装成功，测试CUDA支持...")
            
            # 重新导入OpenCV
            import importlib
            importlib.reload(cv2)
            
            if check_current_opencv():
                print(f"✅ {version} 支持CUDA!")
                return True
            else:
                print(f"⚠️ {version} 不支持CUDA")
                
    return False

def create_optimized_cpu_solution():
    """创建优化的CPU解决方案"""
    print("⚡ 创建优化的CPU处理方案...")
    
    cpu_solution = '''#!/usr/bin/env python3
"""
优化的CPU图像处理方案 (OpenCV CUDA替代)
使用多进程和优化算法实现高性能处理
"""

import cv2
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from pathlib import Path

class OptimizedCPUProcessor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        print(f"初始化CPU处理器，使用 {self.max_workers} 个进程")
        
    def upscale_image(self, image, scale=2):
        """优化的图像放大"""
        height, width = image.shape[:2]
        new_size = (width * scale, height * scale)
        # 使用INTER_CUBIC，比LANCZOS4快但质量仍然很好
        return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        
    def denoise_image(self, image):
        """优化的降噪"""
        # 使用双边滤波，比NLM快很多
        return cv2.bilateralFilter(image, 9, 75, 75)
        
    def sharpen_image(self, image):
        """图像锐化"""
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
        
    def process_single_image(self, args):
        """处理单张图像"""
        input_path, output_path, operations = args
        
        try:
            image = cv2.imread(str(input_path))
            if image is None:
                return False, f"无法加载: {input_path}"
                
            result = image.copy()
            
            for op in operations:
                if op == 'upscale':
                    result = self.upscale_image(result)
                elif op == 'denoise':
                    result = self.denoise_image(result)
                elif op == 'sharpen':
                    result = self.sharpen_image(result)
                    
            success = cv2.imwrite(str(output_path), result)
            return success, None
            
        except Exception as e:
            return False, str(e)
            
    def process_batch(self, input_files, output_dir, operations=['upscale']):
        """批量处理图像"""
        start_time = time.time()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        tasks = []
        for input_file in input_files:
            output_file = Path(output_dir) / input_file.name
            tasks.append((input_file, output_file, operations))
            
        processed = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(self.process_single_image, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                try:
                    success, error = future.result()
                    if success:
                        processed += 1
                    else:
                        failed += 1
                        if error:
                            print(f"处理失败: {error}")
                except Exception as e:
                    failed += 1
                    print(f"任务异常: {e}")
                    
        total_time = time.time() - start_time
        fps = processed / total_time if total_time > 0 else 0
        
        print(f"\\n✅ 批量处理完成:")
        print(f"   处理成功: {processed}")
        print(f"   处理失败: {failed}")
        print(f"   总用时: {total_time:.1f}秒")
        print(f"   平均速度: {fps:.1f} fps")
        
        return processed, failed, total_time

# 使用示例
if __name__ == "__main__":
    processor = OptimizedCPUProcessor()
    
    # 处理示例
    input_dir = Path("frames")  # 输入帧目录
    output_dir = Path("enhanced_frames")  # 输出目录
    
    if input_dir.exists():
        frame_files = sorted(input_dir.glob("*.png"))
        if frame_files:
            processor.process_batch(frame_files, output_dir, ['upscale'])
        else:
            print("未找到PNG文件")
    else:
        print(f"输入目录不存在: {input_dir}")
'''
    
    with open("optimized_cpu_processor.py", "w", encoding='utf-8') as f:
        f.write(cpu_solution)
        
    print("✅ 已创建优化CPU处理方案: optimized_cpu_processor.py")

def main():
    print("🚀 OpenCV CUDA快速修复方案")
    print("=" * 50)
    
    # 检查当前状态
    cuda_available = check_current_opencv()
    
    if cuda_available:
        print("✅ OpenCV CUDA已经可用!")
        if test_cuda_operations():
            print("✅ CUDA操作测试成功")
            return True
        else:
            print("⚠️ CUDA操作测试失败")
    
    print("\n❌ OpenCV CUDA不可用，尝试修复...")
    
    # 尝试安装支持CUDA的版本
    if install_opencv_versions():
        print("✅ 成功安装支持CUDA的OpenCV!")
        if test_cuda_operations():
            print("✅ CUDA操作测试成功")
            return True
    
    print("\n⚠️ 无法安装支持CUDA的OpenCV")
    print("创建优化的CPU处理方案作为替代...")
    
    create_optimized_cpu_solution()
    
    print("\n📋 解决方案总结:")
    print("1. 当前OpenCV不支持CUDA")
    print("2. 已创建优化的CPU处理方案")
    print("3. 使用多进程并行处理提高性能")
    print("4. 文件: optimized_cpu_processor.py")
    
    print("\n🛠️ 手动修复CUDA支持的方法:")
    print("方法1: 使用conda安装")
    print("  conda install -c conda-forge opencv")
    print("\n方法2: 从源码编译")
    print("  ./compile_opencv_cuda.sh")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)