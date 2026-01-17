#!/usr/bin/env python3
"""
终极优化版 GPU 视频处理

整合优化:
1. x2 模型 (快 4 倍) + 分段处理 (可暂停/恢复)
2. FFmpeg pipe 模式 (减少 I/O)
3. 更大的 tile size (512)
4. 断点续传支持
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from huaju4k.gpu_stage import GPUVideoSuperResolver


class SegmentedVideoProcessor:
    """分段视频处理器"""
    
    def __init__(self, input_video, model_name="RealESRGAN_x2plus", 
                 segment_duration=120, tile_size=512):
        self.input_video = input_video
        self.model_name = model_name
        self.segment_duration = segment_duration
        self.tile_size = tile_size
        
        # 目录设置
        self.work_dir = Path("./video_processing_workspace")
        self.segments_dir = self.work_dir / "segments"
        self.enhanced_dir = self.work_dir / "enhanced"
        self.progress_file = self.work_dir / "progress.json"
        
        # 创建目录
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.enhanced_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载进度
        self.progress = self._load_progress()
    
    def _load_progress(self):
        """加载处理进度"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "segments": [],
            "completed": [],
            "failed": [],
            "total_elapsed": 0
        }
    
    def _save_progress(self):
        """保存处理进度"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def split_video(self):
        """分割视频"""
        print(f"\n📹 分割视频...")
        print(f"   每段时长: {self.segment_duration} 秒")
        
        output_pattern = str(self.segments_dir / "segment_%03d.mp4")
        
        cmd = [
            'ffmpeg', '-i', self.input_video,
            '-c', 'copy',
            '-map', '0',
            '-segment_time', str(self.segment_duration),
            '-f', 'segment',
            '-reset_timestamps', '1',
            output_pattern
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 分割失败: {result.stderr}")
            return False
        
        segments = sorted(self.segments_dir.glob("segment_*.mp4"))
        self.progress["segments"] = [s.name for s in segments]
        self._save_progress()
        
        print(f"✅ 分割完成: {len(segments)} 个片段")
        return True
    
    def process_segments(self):
        """处理所有片段"""
        segments = [self.segments_dir / s for s in self.progress["segments"]]
        total_segments = len(segments)
        completed = len(self.progress["completed"])
        
        print(f"\n🚀 开始处理片段")
        print(f"   总片段数: {total_segments}")
        print(f"   已完成: {completed}")
        print(f"   待处理: {total_segments - completed}")
        
        if completed == total_segments:
            print("✅ 所有片段已处理完成")
            return True
        
        # 初始化 GPU 处理器
        print(f"\n初始化 GPU 处理器...")
        print(f"   模型: {self.model_name}")
        print(f"   瓦片大小: {self.tile_size}")
        
        resolver = GPUVideoSuperResolver(
            model_name=self.model_name,
            tile_size=self.tile_size,
            device="cuda"
        )
        
        # 处理每个片段
        for idx, segment in enumerate(segments):
            segment_name = segment.name
            
            # 跳过已完成的
            if segment_name in self.progress["completed"]:
                print(f"\n⏭️  [{idx+1}/{total_segments}] 跳过: {segment_name}")
                continue
            
            output_path = self.enhanced_dir / f"enhanced_{segment_name}"
            
            print(f"\n🎬 [{idx+1}/{total_segments}] 处理: {segment_name}")
            
            start_time = time.time()
            
            try:
                success = resolver.enhance_video_ffmpeg_pipe(
                    str(segment),
                    str(output_path)
                )
                
                elapsed = time.time() - start_time
                
                if success:
                    self.progress["completed"].append(segment_name)
                    self.progress["total_elapsed"] += elapsed
                    self._save_progress()
                    
                    print(f"✅ 完成: {segment_name} ({elapsed/60:.1f} 分钟)")
                    
                    # 显示总体进度
                    total_completed = len(self.progress["completed"])
                    progress_pct = (total_completed / total_segments) * 100
                    avg_time = self.progress["total_elapsed"] / total_completed
                    remaining = (total_segments - total_completed) * avg_time
                    
                    print(f"\n📊 总体进度: {total_completed}/{total_segments} ({progress_pct:.1f}%)")
                    print(f"   已耗时: {self.progress['total_elapsed']/3600:.2f} 小时")
                    print(f"   预计剩余: {remaining/3600:.2f} 小时")
                else:
                    print(f"❌ 失败: {segment_name}")
                    self.progress["failed"].append(segment_name)
                    self._save_progress()
                    return False
                    
            except Exception as e:
                print(f"❌ 异常: {segment_name} - {e}")
                self.progress["failed"].append(segment_name)
                self._save_progress()
                return False
        
        print(f"\n✅ 所有片段处理完成!")
        return True
    
    def merge_segments(self, output_file):
        """合并处理后的片段"""
        print(f"\n🔗 合并片段...")
        
        # 创建文件列表
        list_file = self.work_dir / "concat_list.txt"
        with open(list_file, 'w') as f:
            for segment_name in self.progress["segments"]:
                enhanced_path = self.enhanced_dir / f"enhanced_{segment_name}"
                if enhanced_path.exists():
                    f.write(f"file 'enhanced/{enhanced_path.name}'\n")
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(list_file),
            '-c', 'copy',
            output_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 合并完成: {output_file}")
            return True
        else:
            print(f"❌ 合并失败: {result.stderr}")
            return False
    
    def cleanup(self, keep_segments=False):
        """清理临时文件"""
        if not keep_segments:
            print(f"\n🧹 清理临时文件...")
            import shutil
            if self.segments_dir.exists():
                shutil.rmtree(self.segments_dir)
            if self.enhanced_dir.exists():
                shutil.rmtree(self.enhanced_dir)
            print("✅ 清理完成")


def main():
    input_video = "/mnt/c/Users/Administrator/Downloads/target.mp4"
    
    print("="*60)
    print("终极优化版 GPU 视频处理")
    print("="*60)
    
    print("\n视频信息:")
    print("  输入: target.mp4")
    print("  分辨率: 1920x1080")
    print("  时长: 约 38.7 分钟")
    print("  总帧数: 57999")
    
    print("\n处理模式:")
    print("1. 测试模式 (前 100 帧, 约 5 分钟)")
    print("2. 快速模式 (x2, 3840x2160, 约 10 小时)")
    print("3. 高质量模式 (x4, 7680x4320, 约 40 小时)")
    print("4. 继续上次处理")
    
    choice = input("\n选择 (1/2/3/4): ").strip()
    
    # 测试模式
    if choice == "1":
        print("\n创建测试片段...")
        test_input = "./target_test_100frames.mp4"
        cmd = [
            'ffmpeg', '-y', '-i', input_video,
            '-vframes', '100',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            test_input
        ]
        subprocess.run(cmd, capture_output=True)
        
        processor = SegmentedVideoProcessor(
            input_video=test_input,
            model_name="RealESRGAN_x2plus",
            segment_duration=10,  # 测试用小片段
            tile_size=512
        )
        output_file = "./target_test_enhanced.mp4"
        scale_name = "x2"
    
    # 快速模式
    elif choice == "2":
        processor = SegmentedVideoProcessor(
            input_video=input_video,
            model_name="RealESRGAN_x2plus",
            segment_duration=120,  # 2 分钟一段
            tile_size=384  # 6GB 显存优化
        )
        output_file = "./target_gpu_2k_enhanced.mp4"
        scale_name = "x2"
    
    # 高质量模式
    elif choice == "3":
        processor = SegmentedVideoProcessor(
            input_video=input_video,
            model_name="RealESRGAN_x4plus",
            segment_duration=120,
            tile_size=384
        )
        output_file = "./target_gpu_4k_enhanced.mp4"
        scale_name = "x4"
    
    # 继续模式
    elif choice == "4":
        # 检测上次使用的配置
        progress_file = Path("./video_processing_workspace/progress.json")
        if not progress_file.exists():
            print("❌ 未找到上次的处理进度")
            return 1
        
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        print(f"\n找到上次的处理进度:")
        print(f"  已完成: {len(progress['completed'])} 片段")
        print(f"  总片段: {len(progress['segments'])} 片段")
        
        # 根据进度推断配置
        processor = SegmentedVideoProcessor(
            input_video=input_video,
            model_name="RealESRGAN_x2plus",  # 默认 x2
            segment_duration=120,
            tile_size=384  # 6GB 显存优化
        )
        output_file = "./target_gpu_enhanced_resumed.mp4"
        scale_name = "resumed"
    
    else:
        print("无效选择")
        return 1
    
    # 显示配置
    print(f"\n处理配置:")
    print(f"  模型: {processor.model_name}")
    print(f"  输出: {output_file}")
    print(f"  分段时长: {processor.segment_duration} 秒")
    print(f"  瓦片大小: {processor.tile_size}")
    
    if choice != "1" and choice != "4":
        response = input("\n是否继续? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("已取消")
            return 0
    
    # 执行处理流程
    start_time = time.time()
    
    # 1. 分割视频（如果还没分割）
    if not processor.progress["segments"]:
        if not processor.split_video():
            return 1
    
    # 2. 处理片段
    if not processor.process_segments():
        print("\n⚠️  处理中断，进度已保存")
        print("   可以稍后运行并选择 '4. 继续上次处理'")
        return 1
    
    # 3. 合并片段
    if not processor.merge_segments(output_file):
        return 1
    
    # 4. 清理（可选）
    print("\n是否删除临时文件? (yes/no): ", end='')
    if input().strip().lower() in ['yes', 'y']:
        processor.cleanup(keep_segments=False)
    
    # 总结
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("✅ 处理完成!")
    print("="*60)
    print(f"总耗时: {total_elapsed/3600:.2f} 小时")
    print(f"输出文件: {output_file}")
    
    if Path(output_file).exists():
        size_gb = Path(output_file).stat().st_size / (1024**3)
        print(f"文件大小: {size_gb:.2f} GB")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
