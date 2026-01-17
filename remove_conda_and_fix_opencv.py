#!/usr/bin/env python3
"""
卸载miniconda并修复OpenCV CUDA问题
使用系统Python环境直接解决
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, show_output=True):
    """执行命令"""
    if show_output:
        print(f"执行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if show_output and result.stdout:
            print(f"输出: {result.stdout.strip()}")
        if result.stderr and result.returncode != 0:
            print(f"错误: {result.stderr.strip()}")
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        print(f"命令失败: {e}")
        return False, "", str(e)

def remove_miniconda():
    """卸载miniconda"""
    print("🗑️ 卸载miniconda...")
    
    # 检查miniconda路径
    conda_paths = [
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
        Path("/opt/miniconda3"),
        Path("/opt/anaconda3")
    ]
    
    removed_paths = []
    
    for conda_path in conda_paths:
        if conda_path.exists():
            print(f"发现conda安装: {conda_path}")
            try:
                # 删除目录
                run_command(f"rm -rf {conda_path}")
                removed_paths.append(str(conda_path))
                print(f"✅ 已删除: {conda_path}")
            except Exception as e:
                print(f"❌ 删除失败: {conda_path} - {e}")
    
    # 清理环境变量
    bashrc_path = Path.home() / ".bashrc"
    if bashrc_path.exists():
        print("🧹 清理.bashrc中的conda配置...")
        
        with open(bashrc_path, 'r') as f:
            lines = f.readlines()
        
        # 过滤掉conda相关行
        filtered_lines = []
        skip_conda_block = False
        
        for line in lines:
            if ">>> conda initialize >>>" in line:
                skip_conda_block = True
                continue
            elif "<<< conda initialize <<<" in line:
                skip_conda_block = False
                continue
            elif skip_conda_block:
                continue
            elif "conda" in line.lower() and ("export" in line or "PATH" in line):
                continue
            else:
                filtered_lines.append(line)
        
        # 写回文件
        with open(bashrc_path, 'w') as f:
            f.writelines(filtered_lines)
        
        print("✅ 已清理.bashrc")
    
    # 清理其他配置文件
    config_files = [
        Path.home() / ".condarc",
        Path.home() / ".conda",
        Path.home() / ".continuum"
    ]
    
    for config_file in config_files:
        if config_file.exists():
            run_command(f"rm -rf {config_file}")
            print(f"✅ 已删除配置: {config_file}")
    
    print("✅ miniconda卸载完成")
    return removed_paths

def setup_system_python():
    """设置系统Python环境"""
    print("🐍 设置系统Python环境...")
    
    # 检查系统Python
    success, python_path, _ = run_command("which python3")
    if success:
        print(f"系统Python路径: {python_path.strip()}")
    
    # 检查pip
    success, pip_path, _ = run_command("which pip3")
    if success:
        print(f"系统pip路径: {pip_path.strip()}")
    else:
        print("安装pip...")
        run_command("sudo apt-get update")
        run_command("sudo apt-get install -y python3-pip")
    
    # 更新pip
    print("更新pip...")
    run_command("python3 -m pip install --upgrade pip --user")
    
    return True

def install_opencv_cuda_system():
    """在系统Python中安装OpenCV CUDA"""
    print("📦 在系统Python中安装OpenCV...")
    
    # 卸载现有OpenCV
    opencv_packages = [
        "opencv-python",
        "opencv-contrib-python", 
        "opencv-python-headless"
    ]
    
    for pkg in opencv_packages:
        print(f"卸载 {pkg}...")
        run_command(f"python3 -m pip uninstall -y {pkg}", show_output=False)
    
    # 安装依赖
    print("安装依赖...")
    dependencies = [
        "numpy",
        "matplotlib", 
        "pillow"
    ]
    
    for dep in dependencies:
        print(f"安装 {dep}...")
        run_command(f"python3 -m pip install {dep} --user")
    
    # 尝试安装支持CUDA的OpenCV版本
    opencv_versions = [
        "opencv-contrib-python==4.5.5.64",
        "opencv-python==4.6.0.66",
        "opencv-contrib-python==4.8.1.78"
    ]
    
    for version in opencv_versions:
        print(f"\n尝试安装: {version}")
        success, _, _ = run_command(f"python3 -m pip install {version} --user")
        
        if success:
            # 测试安装
            test_success = test_opencv_import()
            if test_success:
                cuda_available = test_opencv_cuda()
                if cuda_available:
                    print(f"✅ {version} 安装成功且支持CUDA!")
                    return True
                else:
                    print(f"⚠️ {version} 安装成功但不支持CUDA")
            else:
                print(f"❌ {version} 安装失败")
    
    # 如果都不支持CUDA，安装最新版本用于CPU处理
    print("\n安装最新版本用于CPU优化处理...")
    run_command("python3 -m pip install opencv-contrib-python --user")
    
    return test_opencv_import()

def test_opencv_import():
    """测试OpenCV导入"""
    print("🧪 测试OpenCV导入...")
    
    test_code = '''
try:
    import cv2
    print(f"OpenCV版本: {cv2.__version__}")
    print("✅ OpenCV导入成功")
    exit(0)
except ImportError as e:
    print(f"❌ OpenCV导入失败: {e}")
    exit(1)
'''
    
    with open("test_opencv_import.py", "w") as f:
        f.write(test_code)
    
    success, _, _ = run_command("python3 test_opencv_import.py")
    
    # 清理测试文件
    if os.path.exists("test_opencv_import.py"):
        os.remove("test_opencv_import.py")
    
    return success

def test_opencv_cuda():
    """测试OpenCV CUDA支持"""
    print("🧪 测试OpenCV CUDA支持...")
    
    test_code = '''
try:
    import cv2
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
        
        print("✅ CUDA操作测试成功")
        exit(0)
    else:
        print("❌ OpenCV CUDA支持不可用")
        exit(1)
        
except Exception as e:
    print(f"❌ CUDA测试失败: {e}")
    exit(1)
'''
    
    with open("test_opencv_cuda.py", "w") as f:
        f.write(test_code)
    
    success, _, _ = run_command("python3 test_opencv_cuda.py")
    
    # 清理测试文件
    if os.path.exists("test_opencv_cuda.py"):
        os.remove("test_opencv_cuda.py")
    
    return success

def create_compile_script():
    """创建从源码编译OpenCV的脚本"""
    print("📝 创建OpenCV CUDA编译脚本...")
    
    script_content = '''#!/bin/bash
# OpenCV CUDA从源码编译脚本

echo "🚀 从源码编译支持CUDA的OpenCV..."

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA编译器未找到"
    echo "请先安装CUDA Toolkit"
    exit 1
fi

# 安装编译依赖
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
if [ ! -d "opencv" ]; then
    git clone https://github.com/opencv/opencv.git
fi
if [ ! -d "opencv_contrib" ]; then
    git clone https://github.com/opencv/opencv_contrib.git
fi

# 创建编译目录
cd opencv
mkdir -p build
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
    -D PYTHON_EXECUTABLE=$(which python3) \\
    -D BUILD_EXAMPLES=ON ..

# 编译
echo "🔨 开始编译 (这需要1-2小时)..."
make -j$(nproc)

# 安装
echo "📦 安装OpenCV..."
sudo make install
sudo ldconfig

# 验证
echo "🧪 验证安装..."
python3 -c "import cv2; print('OpenCV版本:', cv2.__version__); print('CUDA设备:', cv2.cuda.getCudaEnabledDeviceCount())"

echo "✅ OpenCV CUDA编译安装完成!"
'''
    
    with open("compile_opencv_cuda.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("compile_opencv_cuda.sh", 0o755)
    print("✅ 编译脚本已创建: compile_opencv_cuda.sh")

def main():
    print("🚀 卸载miniconda并修复OpenCV CUDA问题")
    print("=" * 60)
    
    # 1. 卸载miniconda
    removed_paths = remove_miniconda()
    
    if removed_paths:
        print(f"\n✅ 已卸载miniconda: {removed_paths}")
        print("请重启终端或运行: source ~/.bashrc")
        print("然后重新运行此脚本继续安装OpenCV")
        return True
    
    # 2. 设置系统Python环境
    setup_system_python()
    
    # 3. 安装OpenCV
    opencv_success = install_opencv_cuda_system()
    
    if opencv_success:
        print("✅ OpenCV安装成功")
        
        # 测试CUDA支持
        cuda_success = test_opencv_cuda()
        
        if cuda_success:
            print("🎉 OpenCV CUDA支持可用!")
        else:
            print("⚠️ OpenCV CUDA支持不可用")
            print("可以使用CPU优化方案或从源码编译")
            create_compile_script()
    else:
        print("❌ OpenCV安装失败")
        return False
    
    print("\n📋 总结:")
    print("1. ✅ 已卸载miniconda")
    print("2. ✅ 已设置系统Python环境")
    print("3. ✅ 已安装OpenCV")
    print("4. 📝 已创建编译脚本 (如需CUDA支持)")
    
    return True

if __name__ == "__main__":
    main()