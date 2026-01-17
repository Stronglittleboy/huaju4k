#!/usr/bin/env python3
"""
Google Colab 环境快速设置脚本
适用于华剧4K视频增强项目
"""

import os
import subprocess
import sys

def run_command(cmd, check=True):
    """执行命令并显示输出"""
    print(f"执行: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and check:
        print(f"错误: {result.stderr}")
    return result.returncode == 0

def setup_colab_environment():
    """设置Colab环境"""
    print("🚀 开始设置华剧4K视频增强环境...")
    
    # 1. 检查GPU
    print("\n📊 检查GPU状态...")
    run_command("nvidia-smi")
    
    # 2. 克隆项目
    print("\n📥 克隆项目...")
    if not os.path.exists("huaju4k"):
        run_command("git clone https://github.com/Stronglittleboy/huaju4k.git")
    
    os.chdir("huaju4k")
    
    # 3. 安装依赖
    print("\n📦 安装Python依赖...")
    run_command("pip install -r requirements-gpu.txt")
    
    # 4. 安装额外的系统依赖
    print("\n🔧 安装系统依赖...")
    run_command("apt-get update -qq")
    run_command("apt-get install -y ffmpeg")
    
    # 5. 验证OpenCV GPU支持
    print("\n🔍 验证OpenCV GPU支持...")
    run_command("python -c 'import cv2; print(f\"OpenCV版本: {cv2.__version__}\"); print(f\"CUDA设备数: {cv2.cuda.getCudaEnabledDeviceCount()}'\"")
    
    # 6. 创建工作目录
    print("\n📁 创建工作目录...")
    os.makedirs("colab_workspace", exist_ok=True)
    os.makedirs("colab_output", exist_ok=True)
    
    print("\n✅ 环境设置完成！")
    print("📝 使用说明:")
    print("1. 上传视频文件到 colab_workspace/ 目录")
    print("2. 运行处理脚本")
    print("3. 处理结果保存在 colab_output/ 目录")

def create_colab_demo():
    """创建Colab演示脚本"""
    demo_script = '''
# 华剧4K视频增强 - Colab演示
import sys
sys.path.append('/content/huaju4k')

from huaju4k.core.video_enhancement_processor import VideoEnhancementProcessor
from huaju4k.configs.config_manager import ConfigManager

# 配置处理参数
config = {
    "video": {
        "ai_model": "real-esrgan",
        "target_resolution": [3840, 2160],
        "quality": "medium",
        "tile_size": 512,
        "batch_size": 4
    },
    "performance": {
        "use_gpu": True,
        "cpu_threads": 2,
        "memory_limit": 8192
    }
}

# 处理视频
processor = VideoEnhancementProcessor(config)

# 示例：处理上传的视频
input_video = "/content/huaju4k/colab_workspace/input_video.mp4"
output_video = "/content/huaju4k/colab_output/enhanced_video.mp4"

if os.path.exists(input_video):
    processor.process_video(input_video, output_video)
    print(f"✅ 处理完成: {output_video}")
else:
    print("❌ 请先上传视频文件到 colab_workspace/ 目录")
'''
    
    with open("colab_demo.py", "w", encoding="utf-8") as f:
        f.write(demo_script)
    
    print("📝 已创建 colab_demo.py 演示脚本")

if __name__ == "__main__":
    setup_colab_environment()
    create_colab_demo()