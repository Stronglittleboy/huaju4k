#!/usr/bin/env python3
"""
检查视频处理状态和进度
"""

import os
import time
import psutil
import subprocess
from pathlib import Path

def check_process_status():
    """检查处理进程状态"""
    try:
        # 查找huaju4k进程
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                if 'python' in proc.info['name'] and any('huaju4k' in arg for arg in proc.info['cmdline']):
                    print(f"进程ID: {proc.info['pid']}")
                    print(f"CPU使用率: {proc.info['cpu_percent']:.1f}%")
                    print(f"内存使用率: {proc.info['memory_percent']:.1f}%")
                    
                    # 获取进程运行时间
                    create_time = proc.create_time()
                    elapsed = time.time() - create_time
                    print(f"运行时间: {elapsed/60:.1f} 分钟")
                    
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        print("未找到huaju4k处理进程")
        return None
        
    except Exception as e:
        print(f"检查进程状态失败: {e}")
        return None

def check_output_files():
    """检查输出文件状态"""
    output_dir = Path("/mnt/c/Users/Administrator/Downloads/enhanced")
    temp_dirs = [Path("/tmp"), Path("/var/tmp")]
    
    print("\n=== 输出文件检查 ===")
    
    # 检查输出目录
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        if files:
            print(f"输出目录文件: {len(files)} 个")
            for f in files:
                if f.is_file():
                    size_mb = f.stat().st_size / (1024*1024)
                    print(f"  {f.name}: {size_mb:.1f} MB")
        else:
            print("输出目录为空")
    else:
        print("输出目录不存在")
    
    # 检查临时文件
    print("\n=== 临时文件检查 ===")
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            temp_files = list(temp_dir.glob("*huaju4k*")) + list(temp_dir.glob("*enhanced*"))
            if temp_files:
                print(f"{temp_dir} 中的临时文件:")
                for f in temp_files:
                    if f.is_file():
                        size_mb = f.stat().st_size / (1024*1024)
                        print(f"  {f.name}: {size_mb:.1f} MB")

def estimate_progress():
    """估算处理进度"""
    print("\n=== 进度估算 ===")
    
    # 视频信息
    total_frames = 57999
    fps = 25
    duration_minutes = total_frames / fps / 60
    
    print(f"视频总帧数: {total_frames:,}")
    print(f"视频时长: {duration_minutes:.1f} 分钟")
    
    # 基于运行时间估算
    pid = check_process_status()
    if pid:
        try:
            proc = psutil.Process(pid)
            elapsed_minutes = (time.time() - proc.create_time()) / 60
            
            # 假设处理速度为每分钟处理X帧（需要根据实际情况调整）
            # 对于AI增强，通常每秒处理1-10帧
            estimated_fps_processing = 2  # 保守估计每秒处理2帧
            estimated_frames_processed = elapsed_minutes * 60 * estimated_fps_processing
            
            progress_percent = min(100, (estimated_frames_processed / total_frames) * 100)
            
            print(f"估算已处理帧数: {estimated_frames_processed:,.0f}")
            print(f"估算进度: {progress_percent:.1f}%")
            
            if progress_percent < 100:
                remaining_frames = total_frames - estimated_frames_processed
                remaining_minutes = remaining_frames / (estimated_fps_processing * 60)
                print(f"估算剩余时间: {remaining_minutes:.1f} 分钟")
            
        except Exception as e:
            print(f"进度估算失败: {e}")

def main():
    print("=== huaju4k 视频处理状态检查 ===")
    print(f"检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    check_process_status()
    check_output_files()
    estimate_progress()
    
    print("\n=== 建议 ===")
    print("1. 如果进程CPU使用率很低，可能在等待I/O操作")
    print("2. 如果内存使用率很高，可能需要优化内存管理")
    print("3. 对于大视频文件，建议使用更快的处理模式或分段处理")
    print("4. 可以考虑降低输出质量以加快处理速度")

if __name__ == "__main__":
    main()