#!/usr/bin/env python3
"""
分段处理长视频

优势:
1. 可以暂停/恢复
2. 失败后可以从断点继续
3. 可以并行处理多段（如果有多个 GPU）
"""

import sys
import json
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from huaju4k.gpu_stage import GPUVideoSuperResolver


def split_video(input_video, segment_duration=60):
    """将视频分割成多个片段"""
    output_pattern = "./segments/segment_%03d.mp4"
    Path("./segments").mkdir(exist_ok=True)
    
    cmd = [
        'ffmpeg', '-i', input_video,
        '-c', 'copy',
        '-map', '0',
        '-segment_time', str(segment_duration),
        '-f', 'segment',
        '-reset_timestamps', '1',
        output_pattern
    ]
    
    print(f"分割视频（每段 {segment_duration} 秒）...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 分割失败: {result.stderr}")
        return []
    
    segments = sorted(Path("./segments").glob("segment_*.mp4"))
    print(f"✅ 分割完成: {len(segments)} 个片段")
    return segments


def process_segment(resolver, segment_path, output_path, segment_idx, total_segments):
    """处理单个片段"""
    print(f"\n处理片段 {segment_idx+1}/{total_segments}: {segment_path.name}")
    
    start_time = time.time()
    success = resolver.enhance_video_ffmpeg_pipe(
        str(segment_path),
        str(output_path)
    )
    elapsed = time.time() - start_time
    
    return {
        "segment": segment_path.name,
        "success": success,
        "elapsed": elapsed,
        "output": str(output_path)
    }


def merge_segments(segment_outputs, final_output):
    """合并处理后的片段"""
    print(f"\n合并 {len(segment_outputs)} 个片段...")
    
    # 创建文件列表
    list_file = "./segments/concat_list.txt"
    with open(list_file, 'w') as f:
        for seg in segment_outputs:
            f.write(f"file '{Path(seg).name}'\n")
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-c', 'copy',
        final_output
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def main():
    input_video = "/mnt/c/Users/Administrator/Downloads/target.mp4"
    segment_duration = 120  # 每段 2 分钟
    
    print("="*60)
    print("分段处理模式")
    print("="*60)
    
    print(f"\n配置:")
    print(f"  输入视频: {input_video}")
    print(f"  分段时长: {segment_duration} 秒")
    print(f"  总时长: 2320 秒 (约 38.7 分钟)")
    print(f"  预计片段数: {2320 // segment_duration + 1}")
    
    print("\n选择模式:")
    print("1. x2 快速模式 (3840x2160)")
    print("2. x4 高质量模式 (7680x4320)")
    
    choice = input("\n选择 (1/2): ").strip()
    
    if choice == "1":
        model_name = "RealESRGAN_x2plus"
        output_dir = "./segments_enhanced_x2"
        final_output = "./target_gpu_2k_enhanced_merged.mp4"
    elif choice == "2":
        model_name = "RealESRGAN_x4plus"
        output_dir = "./segments_enhanced_x4"
        final_output = "./target_gpu_4k_enhanced_merged.mp4"
    else:
        print("无效选择")
        return 1
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # 分割视频
    segments = split_video(input_video, segment_duration)
    if not segments:
        return 1
    
    print(f"\n将处理 {len(segments)} 个片段")
    response = input("是否继续? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("已取消")
        return 0
    
    # 初始化处理器
    print("\n初始化 GPU 处理器...")
    resolver = GPUVideoSuperResolver(
        model_name=model_name,
        tile_size=512 if choice == "1" else 384,
        device="cuda"
    )
    
    # 处理每个片段
    results = []
    total_start = time.time()
    
    for idx, segment in enumerate(segments):
        output_path = Path(output_dir) / f"enhanced_{segment.name}"
        
        # 检查是否已处理
        if output_path.exists():
            print(f"\n⏭️  跳过已处理的片段: {segment.name}")
            results.append({
                "segment": segment.name,
                "success": True,
                "elapsed": 0,
                "output": str(output_path)
            })
            continue
        
        result = process_segment(resolver, segment, output_path, idx, len(segments))
        results.append(result)
        
        # 保存进度
        with open("./segments/progress.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        if not result["success"]:
            print(f"❌ 片段处理失败: {segment.name}")
            print("进度已保存，可以稍后继续")
            return 1
    
    total_elapsed = time.time() - total_start
    
    # 合并片段
    segment_outputs = [r["output"] for r in results if r["success"]]
    
    if merge_segments(segment_outputs, final_output):
        print(f"\n✅ 全部完成!")
        print(f"   总耗时: {total_elapsed/3600:.2f} 小时")
        print(f"   输出文件: {final_output}")
        
        if Path(final_output).exists():
            size_gb = Path(final_output).stat().st_size / (1024**3)
            print(f"   文件大小: {size_gb:.2f} GB")
        
        return 0
    else:
        print(f"\n❌ 合并失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
