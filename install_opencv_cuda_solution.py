#!/usr/bin/env python3
"""
OpenCV CUDA修复解决方案 - 自动化安装脚本
针对WSL Ubuntu环境优化
"""

import os
import sys
import subprocess
import logging
import json
from datetime import datetime
from pathlib import Path

class OpenCVCUDAInstaller:
    def __init__(self):
        self.setup_logging()
        self.check_environment()
        
    def setup_logging(self):
        """设置日志系统"""
        log_file = f"opencv_cuda_install_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def run_command(self, cmd, check=True):
        """执行命令并记录输出"""
        self.logger.info(f"执行命令: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                self.logger.info(f"输出: {result.stdout.strip()}")
            if result.stderr and result.returncode != 0:
                self.logger.warning(f"错误: {result.stderr.strip()}")
            
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)
                
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            self.logger.error(f"命令执行失败: {e}")
            return False, "", str(e)
            
    def check_environment(self):
        """检查环境"""
        self.logger.info("🔍 检查系统环境...")
        
        # 检查CUDA驱动
        success, stdout, stderr = self.run_command("nvidia-smi", check=False)
        self.cuda_driver_available = success
        
        if self.cuda_driver_available:
            self.logger.info("✅ NVIDIA驱动正常")
        else:
            self.logger.error("❌ NVIDIA驱动不可用")
            
        # 检查conda
        success, stdout, stderr = self.run_command("which conda", check=False)
        self.conda_available = success
        
        if self.conda_available:
            self.logger.info("✅ Conda可用")
        else:
            self.logger.info("ℹ️ Conda不可用，将安装Miniconda")
            
    def install_miniconda(self):
        """安装Miniconda"""
        if self.conda_available:
            self.logger.info("Conda已安装，跳过Miniconda安装")
            return True
            
        self.logger.info("📦 安装Miniconda...")
        
        # 下载Miniconda
        miniconda_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        success, _, _ = self.run_command(f"wget -O miniconda.sh {miniconda_url}")
        
        if not success:
            self.logger.error("❌ 下载Miniconda失败")
            return False
            
        # 安装Miniconda
        success, _, _ = self.run_command("bash miniconda.sh -b -p $HOME/miniconda3")
        
        if not success:
            self.logger.error("❌ 安装Miniconda失败")
            return False
            
        # 初始化conda
        success, _, _ = self.run_command("$HOME/miniconda3/bin/conda init bash")
        
        if success:
            self.logger.info("✅ Miniconda安装成功")
            self.logger.info("请重启终端或运行: source ~/.bashrc")
            return True
        else:
            self.logger.error("❌ Conda初始化失败")
            return False
            
    def install_opencv_cuda_conda(self):
        """使用conda安装支持CUDA的OpenCV"""
        self.logger.info("📦 使用conda安装支持CUDA的OpenCV...")
        
        # 确保conda可用
        if not self.conda_available:
            conda_path = "$HOME/miniconda3/bin/conda"
        else:
            conda_path = "conda"
            
        # 创建专用环境
        self.logger.info("创建opencv-cuda环境...")
        success, _, _ = self.run_command(f"{conda_path} create -n opencv-cuda python=3.10 -y", check=False)
        
        # 激活环境并安装OpenCV
        install_commands = [
            f"{conda_path} install -n opencv-cuda -c conda-forge opencv -y",
            f"{conda_path} install -n opencv-cuda -c conda-forge numpy matplotlib -y",
            f"{conda_path} install -n opencv-cuda -c conda-forge pillow -y"
        ]
        
        for cmd in install_commands:
            success, _, _ = self.run_command(cmd)
            if not success:
                self.logger.warning(f"命令执行可能有问题: {cmd}")
                
        return True
        
    def install_opencv_cuda_pip(self):
        """使用pip安装可能支持CUDA的OpenCV版本"""
        self.logger.info("📦 尝试pip安装支持CUDA的OpenCV...")
        
        # 卸载现有版本
        uninstall_commands = [
            "pip3 uninstall -y opencv-python",
            "pip3 uninstall -y opencv-contrib-python",
            "pip3 uninstall -y opencv-python-headless"
        ]
        
        for cmd in uninstall_commands:
            self.run_command(cmd, check=False)
            
        # 尝试安装特定版本
        versions_to_try = [
            "opencv-contrib-python==4.5.5.64",
            "opencv-python==4.6.0.66",
            "opencv-contrib-python==4.8.1.78"
        ]
        
        for version in versions_to_try:
            self.logger.info(f"尝试安装: {version}")
            success, _, _ = self.run_command(f"pip3 install {version}", check=False)
            
            if success:
                # 测试CUDA支持
                test_success = self.test_opencv_cuda()
                if test_success:
                    self.logger.info(f"✅ {version} 安装成功且支持CUDA")
                    return True
                else:
                    self.logger.info(f"⚠️ {version} 安装成功但不支持CUDA")
                    
        return False
        
    def test_opencv_cuda(self):
        """测试OpenCV CUDA支持"""
        self.logger.info("🧪 测试OpenCV CUDA支持...")
        
        test_code = '''
import cv2
import sys

try:
    print(f"OpenCV版本: {cv2.__version__}")
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"CUDA设备数: {cuda_devices}")
    
    if cuda_devices > 0:
        print("✅ OpenCV CUDA支持可用")
        
        # 测试基本CUDA操作
        import numpy as np
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(test_img)
        
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        result = gpu_gray.download()
        
        print("✅ CUDA基本操作测试成功")
        sys.exit(0)
    else:
        print("❌ OpenCV CUDA支持不可用")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ 测试失败: {e}")
    sys.exit(1)
'''
        
        with open("test_cuda.py", "w") as f:
            f.write(test_code)
            
        success, stdout, stderr = self.run_command("python3 test_cuda.py", check=False)
        
        # 清理测试文件
        if os.path.exists("test_cuda.py"):
            os.remove("test_cuda.py")
            
        return success
        
    def create_activation_script(self):
        """创建环境激活脚本"""
        script_content = '''#!/bin/bash
# OpenCV CUDA环境激活脚本

echo "🚀 激活OpenCV CUDA环境..."

# 检查conda是否可用
if command -v conda &> /dev/null; then
    echo "📦 激活opencv-cuda环境..."
    conda activate opencv-cuda
    
    # 验证安装
    python3 -c "import cv2; print('OpenCV版本:', cv2.__version__); print('CUDA设备:', cv2.cuda.getCudaEnabledDeviceCount())"
    
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    echo "📦 使用Miniconda激活环境..."
    source $HOME/miniconda3/bin/activate opencv-cuda
    
    # 验证安装
    python3 -c "import cv2; print('OpenCV版本:', cv2.__version__); print('CUDA设备:', cv2.cuda.getCudaEnabledDeviceCount())"
    
else
    echo "❌ Conda不可用"
    exit 1
fi

echo "✅ 环境激活完成"
echo "现在可以使用支持CUDA的OpenCV了"
'''
        
        with open("activate_opencv_cuda.sh", "w") as f:
            f.write(script_content)
        os.chmod("activate_opencv_cuda.sh", 0o755)
        
        self.logger.info("📝 已创建环境激活脚本: activate_opencv_cuda.sh")
        
    def run_installation(self):
        """运行完整安装流程"""
        self.logger.info("🚀 开始OpenCV CUDA安装流程...")
        
        if not self.cuda_driver_available:
            self.logger.error("❌ CUDA驱动不可用，无法继续")
            return False
            
        # 方法1: 尝试conda安装
        if self.conda_available or self.install_miniconda():
            self.logger.info("📦 方法1: 使用conda安装...")
            self.install_opencv_cuda_conda()
            
            # 测试conda安装的结果
            # 注意：需要在新的shell中测试，因为环境变量可能未更新
            self.logger.info("ℹ️ conda安装完成，需要手动测试")
            
        # 方法2: 尝试pip安装
        self.logger.info("📦 方法2: 尝试pip安装...")
        pip_success = self.install_opencv_cuda_pip()
        
        # 创建激活脚本
        self.create_activation_script()
        
        # 生成安装报告
        report = {
            "installation_timestamp": datetime.now().isoformat(),
            "system_info": {
                "cuda_driver_available": self.cuda_driver_available,
                "conda_available": self.conda_available
            },
            "installation_methods": {
                "conda_attempted": True,
                "pip_attempted": True,
                "pip_success": pip_success
            },
            "next_steps": [
                "重启终端或运行: source ~/.bashrc",
                "运行: ./activate_opencv_cuda.sh",
                "测试: python3 -c \"import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())\"",
                "如果仍不可用，考虑从源码编译"
            ],
            "files_created": [
                "activate_opencv_cuda.sh",
                "miniconda.sh (如果下载了)",
                "opencv_cuda_install_*.log"
            ]
        }
        
        with open("opencv_cuda_install_report.json", "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        self.logger.info("✅ 安装流程完成")
        self.logger.info("📋 后续步骤:")
        self.logger.info("   1. 重启终端或运行: source ~/.bashrc")
        self.logger.info("   2. 运行: ./activate_opencv_cuda.sh")
        self.logger.info("   3. 测试CUDA支持")
        
        return True

def main():
    installer = OpenCVCUDAInstaller()
    success = installer.run_installation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()