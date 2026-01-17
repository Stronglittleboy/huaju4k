#!/usr/bin/env python3
"""
修复OpenCV CUDA支持问题
安装支持CUDA的OpenCV版本
"""

import subprocess
import sys
import os
import logging
from datetime import datetime

def setup_logging():
    """设置日志系统"""
    log_file = f"opencv_cuda_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_command(cmd, logger):
    """执行命令并记录输出"""
    logger.info(f"执行命令: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"输出: {result.stdout}")
        if result.stderr:
            logger.warning(f"错误: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        logger.error(f"命令执行失败: {e}")
        return False

def check_cuda_availability(logger):
    """检查CUDA可用性"""
    logger.info("🔍 检查CUDA环境...")
    
    # 检查nvidia-smi
    if run_command("nvidia-smi", logger):
        logger.info("✅ NVIDIA驱动正常")
    else:
        logger.error("❌ NVIDIA驱动未找到")
        return False
    
    # 检查CUDA版本
    if run_command("nvcc --version", logger):
        logger.info("✅ CUDA编译器可用")
    else:
        logger.warning("⚠️ CUDA编译器未找到，但驱动可用")
    
    return True

def uninstall_opencv(logger):
    """卸载现有OpenCV"""
    logger.info("🗑️ 卸载现有OpenCV包...")
    
    packages_to_remove = [
        "opencv-python",
        "opencv-contrib-python",
        "opencv-python-headless"
    ]
    
    for package in packages_to_remove:
        logger.info(f"卸载 {package}...")
        run_command(f"pip3 uninstall -y {package}", logger)

def install_opencv_cuda(logger):
    """安装支持CUDA的OpenCV"""
    logger.info("📦 安装支持CUDA的OpenCV...")
    
    # 方法1: 尝试安装预编译的CUDA版本
    logger.info("尝试方法1: 安装opencv-contrib-python (GPU版本)")
    
    # 首先更新pip
    run_command("pip3 install --upgrade pip", logger)
    
    # 安装支持CUDA的OpenCV版本
    # 注意：这可能需要从源码编译或使用特定的预编译版本
    success = False
    
    # 尝试安装opencv-contrib-python的最新版本
    if run_command("pip3 install opencv-contrib-python==4.12.0.88", logger):
        logger.info("✅ OpenCV安装成功")
        success = True
    
    if not success:
        logger.warning("⚠️ 标准安装失败，尝试其他方法...")
        
        # 方法2: 尝试conda安装（如果可用）
        if run_command("which conda", logger):
            logger.info("尝试使用conda安装...")
            if run_command("conda install -c conda-forge opencv", logger):
                success = True
        
        # 方法3: 从源码编译（最后手段）
        if not success:
            logger.warning("需要从源码编译OpenCV以支持CUDA")
            logger.info("这需要较长时间，建议手动执行以下步骤：")
            logger.info("1. 安装编译依赖: sudo apt-get install build-essential cmake git")
            logger.info("2. 下载OpenCV源码")
            logger.info("3. 使用CMAKE配置CUDA支持")
            logger.info("4. 编译安装")
    
    return success

def test_opencv_cuda(logger):
    """测试OpenCV CUDA功能"""
    logger.info("🧪 测试OpenCV CUDA功能...")
    
    test_code = '''
import cv2
import numpy as np

print("OpenCV版本:", cv2.__version__)
print("CUDA设备数量:", cv2.cuda.getCudaEnabledDeviceCount())

if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("✅ CUDA支持可用")
    
    # 测试基本CUDA操作
    try:
        # 创建测试图像
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 上传到GPU
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(test_img)
        
        # GPU操作测试
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        
        # 下载结果
        result = gpu_gray.download()
        
        print("✅ CUDA基本操作测试成功")
        print(f"   输入图像大小: {test_img.shape}")
        print(f"   输出图像大小: {result.shape}")
        
    except Exception as e:
        print(f"❌ CUDA操作测试失败: {e}")
else:
    print("❌ CUDA支持不可用")
'''
    
    # 将测试代码写入临时文件
    with open("test_opencv_cuda.py", "w") as f:
        f.write(test_code)
    
    # 执行测试
    success = run_command("python3 test_opencv_cuda.py", logger)
    
    # 清理临时文件
    if os.path.exists("test_opencv_cuda.py"):
        os.remove("test_opencv_cuda.py")
    
    return success

def provide_manual_solution(logger):
    """提供手动解决方案"""
    logger.info("📋 手动解决方案:")
    logger.info("由于预编译的OpenCV包通常不包含CUDA支持，需要以下步骤：")
    logger.info("")
    logger.info("方案1: 使用conda安装 (推荐)")
    logger.info("1. 安装miniconda或anaconda")
    logger.info("2. conda install -c conda-forge opencv")
    logger.info("")
    logger.info("方案2: 从源码编译OpenCV")
    logger.info("1. sudo apt-get update")
    logger.info("2. sudo apt-get install build-essential cmake git pkg-config")
    logger.info("3. sudo apt-get install libjpeg-dev libtiff5-dev libpng-dev")
    logger.info("4. sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev")
    logger.info("5. sudo apt-get install libgtk2.0-dev libcanberra-gtk-module")
    logger.info("6. sudo apt-get install python3-dev python3-numpy")
    logger.info("7. git clone https://github.com/opencv/opencv.git")
    logger.info("8. git clone https://github.com/opencv/opencv_contrib.git")
    logger.info("9. cd opencv && mkdir build && cd build")
    logger.info("10. cmake -D CMAKE_BUILD_TYPE=RELEASE \\")
    logger.info("    -D CMAKE_INSTALL_PREFIX=/usr/local \\")
    logger.info("    -D WITH_CUDA=ON \\")
    logger.info("    -D ENABLE_FAST_MATH=1 \\")
    logger.info("    -D CUDA_FAST_MATH=1 \\")
    logger.info("    -D WITH_CUBLAS=1 \\")
    logger.info("    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \\")
    logger.info("    -D BUILD_EXAMPLES=OFF ..")
    logger.info("11. make -j$(nproc)")
    logger.info("12. sudo make install")
    logger.info("")
    logger.info("方案3: 使用Docker (临时解决)")
    logger.info("1. 使用包含CUDA支持的OpenCV Docker镜像")
    logger.info("2. docker run --gpus all -it opencv/opencv:cuda")

def main():
    logger = setup_logging()
    logger.info("🚀 开始修复OpenCV CUDA支持问题...")
    
    # 检查CUDA环境
    if not check_cuda_availability(logger):
        logger.error("❌ CUDA环境不可用，无法继续")
        return False
    
    # 卸载现有OpenCV
    uninstall_opencv(logger)
    
    # 安装支持CUDA的OpenCV
    if install_opencv_cuda(logger):
        # 测试CUDA功能
        if test_opencv_cuda(logger):
            logger.info("✅ OpenCV CUDA修复成功!")
            return True
        else:
            logger.warning("⚠️ OpenCV安装成功但CUDA功能不可用")
    
    # 如果自动修复失败，提供手动解决方案
    logger.warning("❌ 自动修复失败")
    provide_manual_solution(logger)
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)