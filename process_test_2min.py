#!/usr/bin/env python3
"""
处理 2 分钟测试视频
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from huaju4k.gpu_stage import GPUVideoSuperResolver


def main():
    input_video = "./target_test_2min.mp4"
    output_video = "./target_test_2min_enhanced_x2.mp4"
    
    print("="*60)
    print("处理 2 分钟测试视频")
    print("="*60)
    
    print(f"\n输入视频: {input_video}")
    print(f"  分辨率: 1920x1080")
    print(f"  时长: 30 秒")
    print(f"  帧数: 778")
    print(f"  文件大小: 12 MB")
    
    print(f"\n输出视频: {output_video}")
    print(f"  分辨率: 3840x2160 (4K)")
    print(f"  模型: RealESRGAN_x2plus")
    print(f"  瓦片大小: 384 (6GB 显存优化)")
    
    print(f"\n预估处理时间: 约 {778 * 2.5 / 60:.0f} 分钟")
    
    input("\n按 Enter 开始处理...")
    
    # 创建处理器
    print("\n初始化 GPU 处理器...")
    resolver = GPUVideoSuperResolver(
        model_name="RealESRGAN_x2plus",
        tile_size=384,  # 384 在 6GB 显存更稳定
        device="cuda"
    )
    
    # 处理视频
    print("\n开始处理...")
    print("💡 提示: 可以在另一个终端运行 'watch -n 1 nvidia-smi' 监控 GPU\n")
    
    start_time = time.time()
    
    success = resolver.enhance_video(
        input_video=input_video,
        output_video=output_video
    )
    
    elapsed = time.time() - start_time
    
    if success:
        print(f"\n✅ 处理完成!")
        print(f"   总耗时: {elapsed/60:.1f} 分钟")
        print(f"   处理速度: {778/elapsed:.2f} fps")
        print(f"   输出文件: {output_video}")
        
        if Path(output_video).exists():
            size_mb = Path(output_video).stat().st_size / (1024*1024)
            print(f"   文件大小: {size_mb:.1f} MB")
            
            # 验证输出分辨率
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
                 '-show_entries', 'stream=width,height', '-of', 'csv=p=0',
                 output_video],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                width, height = result.stdout.strip().split(',')
                print(f"   输出分辨率: {width}x{height}")
        
        # 预估完整视频时间
        full_frames = 57999
        full_time = full_frames * (elapsed / 778)
        print(f"\n📊 完整视频预估:")
        print(f"   总帧数: {full_frames}")
        print(f"   预计耗时: {full_time/3600:.1f} 小时")
        
        return 0
    else:
        print(f"\n❌ 处理失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
