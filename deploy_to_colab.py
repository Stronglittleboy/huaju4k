#!/usr/bin/env python3
"""
华剧4K视频增强项目 - Google Colab 部署脚本
自动化设置和运行环境
"""

import os
import subprocess
import json
from pathlib import Path

class ColabDeployer:
    def __init__(self):
        self.project_name = "huaju4k"
        self.github_repo = "https://github.com/Stronglittleboy/huaju4k.git"
        
    def check_gpu(self):
        """检查GPU可用性"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ GPU检测成功")
                print(result.stdout)
                return True
            else:
                print("❌ 未检测到GPU")
                return False
        except FileNotFoundError:
            print("❌ nvidia-smi 未找到")
            return False
    
    def setup_environment(self):
        """设置Colab环境"""
        print("🚀 开始设置华剧4K视频增强环境...")
        
        # 1. 更新系统包
        os.system("apt-get update -qq")
        os.system("apt-get install -y ffmpeg mediainfo")
        
        # 2. 克隆项目
        if not os.path.exists(self.project_name):
            print(f"📥 克隆项目: {self.github_repo}")
            os.system(f"git clone {self.github_repo}")
        
        os.chdir(self.project_name)
        
        # 3. 安装Python依赖
        print("📦 安装Python依赖...")
        
        # 创建轻量级requirements用于Colab
        colab_requirements = """
opencv-python==4.8.1.78
numpy>=1.21.0
torch>=1.12.0
torchvision>=0.13.0
librosa>=0.9.0
scipy>=1.7.0
pillow>=8.3.0
tqdm>=4.62.0
psutil>=5.8.0
pyyaml>=5.4.0
"""
        
        with open("requirements-colab.txt", "w") as f:
            f.write(colab_requirements)
        
        os.system("pip install -r requirements-colab.txt")
        
        # 4. 创建工作目录
        os.makedirs("colab_input", exist_ok=True)
        os.makedirs("colab_output", exist_ok=True)
        os.makedirs("colab_temp", exist_ok=True)
        
        print("✅ 环境设置完成!")
    
    def create_simple_processor(self):
        """创建简化的处理器用于Colab"""
        processor_code = '''
import cv2
import numpy as np
import os
from pathlib import Path
import subprocess
import json
from tqdm import tqdm

class ColabVideoProcessor:
    def __init__(self, config=None):
        self.config = config or {
            "target_resolution": [1920, 1080],  # 降低分辨率以适应免费GPU
            "quality": "medium",
            "use_gpu": True,
            "tile_size": 256,  # 较小的tile size
            "batch_size": 2    # 较小的batch size
        }
        
        # 检查GPU
        self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        print(f"GPU可用: {self.gpu_available}")
    
    def extract_frames(self, video_path, output_dir):
        """提取视频帧"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用FFmpeg提取帧
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', 'fps=30',  # 限制帧率
            f'{output_dir}/frame_%06d.png',
            '-y'
        ]
        
        subprocess.run(cmd, check=True)
        
        frames = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
        return frames
    
    def enhance_frame(self, frame_path):
        """增强单帧"""
        # 读取图像
        img = cv2.imread(frame_path)
        if img is None:
            return None
        
        # 简单的增强处理（适合免费GPU）
        if self.gpu_available:
            try:
                # GPU处理
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)
                
                # 双边滤波去噪
                gpu_filtered = cv2.cuda.bilateralFilter(gpu_img, -1, 50, 50)
                
                # 下载回CPU
                enhanced = gpu_filtered.download()
            except:
                # GPU失败时回退到CPU
                enhanced = cv2.bilateralFilter(img, 9, 75, 75)
        else:
            # CPU处理
            enhanced = cv2.bilateralFilter(img, 9, 75, 75)
        
        # 锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def process_video(self, input_path, output_path):
        """处理视频"""
        print(f"开始处理视频: {input_path}")
        
        # 1. 提取帧
        temp_dir = "colab_temp/frames"
        frames = self.extract_frames(input_path, temp_dir)
        print(f"提取了 {len(frames)} 帧")
        
        # 2. 处理帧
        enhanced_dir = "colab_temp/enhanced"
        os.makedirs(enhanced_dir, exist_ok=True)
        
        for i, frame in enumerate(tqdm(frames, desc="处理帧")):
            frame_path = os.path.join(temp_dir, frame)
            enhanced_frame = self.enhance_frame(frame_path)
            
            if enhanced_frame is not None:
                output_frame_path = os.path.join(enhanced_dir, frame)
                cv2.imwrite(output_frame_path, enhanced_frame)
        
        # 3. 重新组装视频
        self.reassemble_video(enhanced_dir, input_path, output_path)
        
        # 4. 清理临时文件
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(enhanced_dir, ignore_errors=True)
        
        print(f"处理完成: {output_path}")
    
    def reassemble_video(self, frames_dir, original_video, output_path):
        """重新组装视频"""
        # 获取原视频信息
        probe_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', original_video
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_info = json.loads(result.stdout)
        
        # 找到视频流
        video_stream = None
        for stream in video_info['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
                break
        
        fps = eval(video_stream['r_frame_rate']) if video_stream else 30
        
        # 使用FFmpeg重新组装
        cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', f'{frames_dir}/frame_%06d.png',
            '-i', original_video,  # 原视频用于音频
            '-c:v', 'libx264',
            '-c:a', 'copy',  # 复制音频
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # 高质量
            output_path,
            '-y'
        ]
        
        subprocess.run(cmd, check=True)

# 使用示例
if __name__ == "__main__":
    processor = ColabVideoProcessor()
    
    # 处理colab_input目录中的所有视频
    input_dir = "colab_input"
    output_dir = "colab_output"
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"enhanced_{filename}")
            
            try:
                processor.process_video(input_path, output_path)
                print(f"✅ 成功处理: {filename}")
            except Exception as e:
                print(f"❌ 处理失败 {filename}: {e}")
'''
        
        with open("colab_processor.py", "w", encoding="utf-8") as f:
            f.write(processor_code)
        
        print("📝 已创建 colab_processor.py")
    
    def create_colab_notebook(self):
        """创建Colab notebook"""
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 0,
            "metadata": {
                "colab": {
                    "provenance": [],
                    "gpuType": "T4"
                },
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3"
                },
                "accelerator": "GPU"
            },
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": [
                        "# 华剧4K视频增强 - Google Colab版本\\n\\n",
                        "免费GPU视频处理解决方案\\n\\n",
                        "## 使用步骤：\\n",
                        "1. 运行时类型 → GPU (T4)\\n",
                        "2. 按顺序执行代码块\\n",
                        "3. 上传视频文件\\n",
                        "4. 等待处理完成\\n",
                        "5. 下载增强后的视频"
                    ],
                    "metadata": {"id": "header"}
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# 1. 环境检查和设置\\n",
                        "!nvidia-smi\\n",
                        "print('GPU检查完成')\\n\\n",
                        "# 克隆项目\\n",
                        "!git clone https://github.com/Stronglittleboy/huaju4k.git\\n",
                        "%cd huaju4k\\n\\n",
                        "# 运行部署脚本\\n",
                        "!python deploy_to_colab.py"
                    ],
                    "metadata": {"id": "setup"},
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# 2. 上传视频文件\\n",
                        "from google.colab import files\\n",
                        "import os\\n\\n",
                        "print('请选择要处理的视频文件:')\\n",
                        "uploaded = files.upload()\\n\\n",
                        "# 移动到输入目录\\n",
                        "for filename in uploaded.keys():\\n",
                        "    os.rename(filename, f'colab_input/{filename}')\\n",
                        "    print(f'文件已保存: {filename}')"
                    ],
                    "metadata": {"id": "upload"},
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# 3. 开始处理\\n",
                        "!python colab_processor.py"
                    ],
                    "metadata": {"id": "process"},
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# 4. 下载结果\\n",
                        "from google.colab import files\\n",
                        "import os\\n\\n",
                        "output_files = os.listdir('colab_output')\\n",
                        "for filename in output_files:\\n",
                        "    if filename.endswith(('.mp4', '.avi')):\\n",
                        "        files.download(f'colab_output/{filename}')\\n",
                        "        print(f'下载: {filename}')"
                    ],
                    "metadata": {"id": "download"},
                    "execution_count": None,
                    "outputs": []
                }
            ]
        }
        
        with open("HuaJu4K_Colab.ipynb", "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        print("📓 已创建 HuaJu4K_Colab.ipynb")
    
    def deploy(self):
        """执行完整部署"""
        print("🎬 华剧4K视频增强 - Colab部署器")
        print("=" * 50)
        
        # 检查GPU
        if not self.check_gpu():
            print("⚠️  建议在GPU环境中运行以获得最佳性能")
        
        # 设置环境
        self.setup_environment()
        
        # 创建处理器
        self.create_simple_processor()
        
        # 创建notebook
        self.create_colab_notebook()
        
        print("\\n🎉 部署完成!")
        print("📋 使用说明:")
        print("1. 打开 HuaJu4K_Colab.ipynb")
        print("2. 上传到Google Colab")
        print("3. 设置运行时为GPU")
        print("4. 按顺序执行代码块")

if __name__ == "__main__":
    deployer = ColabDeployer()
    deployer.deploy()