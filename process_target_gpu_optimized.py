#!/usr/bin/env python3
"""
优化版 GPU 视频处理

优化策略:
1. 使用 x2 模型 (3840x2160) 而不是 x4 (7680x4320)
2. 使用 FFmpeg pipe 减少 I/O 开销
3. 批处理帧减少 GPU 初始化开销
4. 更大的 tile size (512 vs 384)
"""

import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from huaju4k.gpu_stage import GPUVideoSuperResolver


def main():
    input_video = "/mnt/c/Users/Administrator/Downloads/target.mp4"
    
    print("="*60)
    print("优化版 GPU 视频处理")
    print("="*60)
    
    print("\n请选择处理模式:")
    print("1. 快速模式 (x2, 3840x2160, 约 10 小时)")
    print("2. 高质量模式 (x4, 7680x4320, 约 40 小时)")
    print("3. 测试模式 (前 100 帧)")
    
    choice = input("\n选择 (1/2/3): ").strip()
    
    if choice == "1":
        model_name = "RealESRGAN_x2plus"
        output_video = "./target_gpu_2k_enhanced.mp4"
        scale = 2
        tile_size = 512
        estimated_hours = 10
    elif choice == "2":
        model_name = "RealESRGAN_x4plus"
        output_video = "./target_gpu_4k_enhanced.mp4"
        scale = 4
        tile_size = 384
        estimated_hours = 40
    elif choice == "3":
        # 测试模式
        print("\n创建测试片段...")
        test_input = "./target_test_100frames.mp4"
        cmd = [
            'ffmpeg', '-y', '-i', input_video,
            '-vframes', '100',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            test_input
        ]
        subprocess.run(cmd, capture_output=True)
        
        input_video = test_input
        model_name = "RealESRGAN_x2plus"
        output_video = "./target_test_100frames_enhanced.mp4"
        scale = 2
        tile_size = 512
        estimated_hours = 0.1
    else:
        print("无效选择")
        return 1
    
    # 显示配置
    print(f"\n处理配置:")
    print(f"  输入: {input_video}")
    print(f"  输出: {output_video}")
    print(f"  模型: {model_name}")
    print(f"  输出分辨率: {1920*scale}x{1080*scale}")
    print(f"  瓦片大小: {tile_size}")
    print(f"  预估时间: {estimated_hours:.1f} 小时")
    
    if choice != "3":
        response = input("\n是否继续? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("已取消")
            return 0
    
    # 创建处理器
    print("\n初始化 GPU 处理器...")
    resolver = GPUVideoSuperResolver(
        model_name=model_name,
        tile_size=tile_size,
        device="cuda"
    )
    
    # 使用 FFmpeg pipe 模式（更快）
    print("\n开始处理（FFmpeg Pipe 模式）...")
    start_time = time.time()
    
    success = resolver.enhance_video_ffmpeg_pipe(
        input_video=input_video,
        output_video=output_video
    )
    
    elapsed = time.time() - start_time
    
    if success:
        print(f"\n✅ 处理完成!")
        print(f"   总耗时: {elapsed/3600:.2f} 小时")
        print(f"   输出文件: {output_video}")
        
        if Path(output_video).exists():
            size_gb = Path(output_video).stat().st_size / (1024**3)
            print(f"   文件大小: {size_gb:.2f} GB")
        
        return 0
    else:
        print(f"\n❌ 处理失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
