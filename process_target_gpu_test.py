#!/usr/bin/env python3
"""
测试处理目标视频的前 N 帧
"""

import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from huaju4k.gpu_stage import GPUVideoSuperResolver


def main():
    input_video = "/mnt/c/Users/Administrator/Downloads/target.mp4"
    test_frames = 100  # 只处理前 100 帧（约 4 秒）
    
    # 创建测试片段
    test_input = "./target_test_100frames.mp4"
    test_output = "./target_test_100frames_4k.mp4"
    
    print("="*60)
    print("GPU Stage 测试处理（前 100 帧）")
    print("="*60)
    
    # 提取前 100 帧
    print(f"\n提取前 {test_frames} 帧...")
    cmd = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-vframes', str(test_frames),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        test_input
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"❌ 提取失败")
        return 1
    
    print(f"✅ 测试片段创建: {test_input}")
    
    # GPU 处理
    print(f"\n开始 GPU 超分处理...")
    print(f"  输入: {test_input}")
    print(f"  输出: {test_output}")
    print(f"  帧数: {test_frames}")
    print(f"  预计时间: {test_frames * 2.5 / 60:.1f} 分钟")
    
    resolver = GPUVideoSuperResolver(
        model_name="RealESRGAN_x4plus",
        tile_size=384,
        device="cuda"
    )
    
    start_time = time.time()
    success = resolver.enhance_video(test_input, test_output)
    elapsed = time.time() - start_time
    
    if success:
        print(f"\n✅ 测试处理完成!")
        print(f"   耗时: {elapsed/60:.1f} 分钟")
        print(f"   速度: {test_frames/elapsed:.2f} fps")
        print(f"   输出: {test_output}")
        
        # 预估完整视频时间
        full_time = 57999 * (elapsed / test_frames)
        print(f"\n📊 完整视频预估:")
        print(f"   总帧数: 57999")
        print(f"   预计耗时: {full_time/3600:.1f} 小时")
        
        return 0
    else:
        print(f"\n❌ 处理失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
