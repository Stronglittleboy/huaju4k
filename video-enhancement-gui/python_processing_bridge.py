#!/usr/bin/env python3
"""
Python处理桥接脚本

这个脚本作为Rust和Python之间的桥梁，集成现有的video_enhancement_toolkit
提供详细的进度跟踪和状态报告功能。

使用方法:
    python python_processing_bridge.py --task-id <id> --stage <stage> --input-file <file> --output-path <path> --workspace <dir> --config <json>
"""

import sys
import os
import json
import argparse
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback

# 添加video_enhancement_toolkit到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'video_enhancement_toolkit'))

try:
    from video_enhancement_toolkit.container import container
    from video_enhancement_toolkit.module_config import configure_default_modules
    from video_enhancement_toolkit.core.interfaces import IVideoProcessor, IAudioOptimizer
    from video_enhancement_toolkit.infrastructure.interfaces import ILogger, IProgressTracker
    from video_enhancement_toolkit.core.models import VideoConfig, AudioConfig
    from video_enhancement_toolkit.infrastructure.models import LoggingConfig, LogLevel
except ImportError as e:
    print(f"ERROR: 无法导入video_enhancement_toolkit: {e}", file=sys.stderr)
    sys.exit(1)


class ProgressReporter:
    """进度报告器，将进度信息输出到stdout供Rust读取"""
    
    def __init__(self, task_id: str, stage: str):
        self.task_id = task_id
        self.stage = stage
        self.start_time = time.time()
        self.last_report_time = 0
        
    def report_progress(self, progress: float, processed_frames: int = 0, 
                       total_frames: int = 0, frame_rate: float = 0.0,
                       cpu_usage: float = 0.0, gpu_usage: float = 0.0,
                       memory_usage: float = 0.0, disk_usage: float = 0.0,
                       processed_data_size: int = 0):
        """报告处理进度"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 计算预估剩余时间
        if progress > 0:
            eta = (elapsed_time / progress) * (100 - progress)
        else:
            eta = 0
        
        progress_data = {
            "task_id": self.task_id,
            "stage": self.stage,
            "progress": progress,
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "frame_rate": frame_rate,
            "elapsed_time": int(elapsed_time),
            "eta": int(eta),
            "cpu_usage": cpu_usage,
            "gpu_usage": gpu_usage,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage,
            "processed_data_size": processed_data_size,
            "timestamp": datetime.now().isoformat()
        }
        
        # 输出JSON格式的进度数据到stdout
        print(json.dumps(progress_data), flush=True)
        self.last_report_time = current_time
    
    def report_log(self, level: str, message: str, details: Optional[Dict[str, Any]] = None):
        """报告日志信息到stderr"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level.upper(),
            "stage": self.stage,
            "message": message,
            "details": details or {}
        }
        print(f"{level.upper()}: {message}", file=sys.stderr, flush=True)


class CustomProgressTracker:
    """自定义进度跟踪器，集成到现有的toolkit中"""
    
    def __init__(self, reporter: ProgressReporter):
        self.reporter = reporter
        self.tasks = {}
        
    def start_task(self, task_id: str, total_steps: int, description: str):
        """开始跟踪任务"""
        self.tasks[task_id] = {
            "total_steps": total_steps,
            "completed_steps": 0,
            "description": description,
            "start_time": time.time()
        }
        self.reporter.report_log("INFO", f"开始任务: {description}")
        
    def update_progress(self, task_id: str, completed_steps: int, status_message: str = ""):
        """更新任务进度"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task["completed_steps"] = completed_steps
            
            progress = (completed_steps / task["total_steps"]) * 100 if task["total_steps"] > 0 else 0
            
            self.reporter.report_progress(
                progress=progress,
                processed_frames=completed_steps,
                total_frames=task["total_steps"]
            )
            
            if status_message:
                self.reporter.report_log("INFO", status_message)
    
    def complete_task(self, task_id: str, success: bool, summary: str):
        """完成任务"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            elapsed = time.time() - task["start_time"]
            
            level = "INFO" if success else "ERROR"
            self.reporter.report_log(level, f"任务完成: {summary} (耗时: {elapsed:.2f}秒)")
            
            if success:
                self.reporter.report_progress(progress=100.0)


async def process_frame_extraction(config: Dict[str, Any], reporter: ProgressReporter) -> str:
    """执行帧提取阶段"""
    reporter.report_log("INFO", "开始帧提取阶段")
    
    try:
        # 配置默认模块
        configure_default_modules(container)
        
        # 获取视频处理器
        video_processor = container.get(IVideoProcessor)
        progress_tracker = CustomProgressTracker(reporter)
        
        # 创建视频配置
        video_config = VideoConfig(
            input_path=config["input_file"],
            output_path=config["output_path"],
            target_resolution=tuple(config["video"]["target_resolution"]),
            ai_model=config["video"]["ai_model"],
            tile_size=config["video"]["tile_size"],
            batch_size=config["video"]["batch_size"],
            gpu_acceleration=config["performance"]["use_gpu"]
        )
        
        # 设置进度跟踪器
        if hasattr(video_processor, 'progress_tracker'):
            video_processor.progress_tracker = progress_tracker
        
        # 执行帧提取
        workspace_dir = config["workspace_dir"]
        frames_dir = os.path.join(workspace_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        reporter.report_log("INFO", f"提取帧到目录: {frames_dir}")
        
        # 调用视频处理器的帧提取方法
        frame_files = video_processor.extract_frames(config["input_file"], frames_dir)
        
        reporter.report_log("INFO", f"成功提取 {len(frame_files)} 帧")
        return frames_dir
        
    except Exception as e:
        reporter.report_log("ERROR", f"帧提取失败: {str(e)}")
        raise


async def process_ai_enhancement(config: Dict[str, Any], reporter: ProgressReporter, frames_dir: str) -> str:
    """执行AI增强阶段"""
    reporter.report_log("INFO", "开始AI增强阶段")
    
    try:
        # 获取视频处理器
        video_processor = container.get(IVideoProcessor)
        progress_tracker = CustomProgressTracker(reporter)
        
        # 设置进度跟踪器
        if hasattr(video_processor, 'progress_tracker'):
            video_processor.progress_tracker = progress_tracker
        
        # 获取帧文件列表
        frame_files = []
        if os.path.exists(frames_dir):
            frame_files = sorted([
                os.path.join(frames_dir, f) 
                for f in os.listdir(frames_dir) 
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
        
        reporter.report_log("INFO", f"找到 {len(frame_files)} 个帧文件进行增强")
        
        if not frame_files:
            raise ValueError("没有找到可处理的帧文件")
        
        # 执行AI增强
        enhanced_frames = video_processor.enhance_frames(frame_files)
        
        reporter.report_log("INFO", f"成功增强 {len(enhanced_frames)} 帧")
        
        # 返回增强帧目录
        enhanced_dir = os.path.join(config["workspace_dir"], "enhanced")
        return enhanced_dir
        
    except Exception as e:
        reporter.report_log("ERROR", f"AI增强失败: {str(e)}")
        raise


async def process_audio_optimization(config: Dict[str, Any], reporter: ProgressReporter) -> str:
    """执行音频处理阶段"""
    reporter.report_log("INFO", "开始音频处理阶段")
    
    try:
        # 获取音频优化器
        audio_optimizer = container.get(IAudioOptimizer)
        
        # 创建音频配置
        audio_config = AudioConfig(
            noise_reduction_strength=config["audio"]["noise_reduction"],
            dialogue_enhancement=config["audio"]["dialogue_enhancement"],
            dynamic_range_target=-23.0,
            preserve_naturalness=True,
            theater_preset=config["audio"].get("theater_preset", "medium")
        )
        
        # 提取音频
        workspace_dir = config["workspace_dir"]
        audio_file = os.path.join(workspace_dir, "original_audio.wav")
        
        # 使用video_processor提取音频
        video_processor = container.get(IVideoProcessor)
        extracted_audio = video_processor.extract_audio(config["input_file"], audio_file)
        
        if not extracted_audio:
            raise ValueError("音频提取失败")
        
        reporter.report_log("INFO", f"音频提取完成: {extracted_audio}")
        
        # 分析音频特征
        reporter.report_progress(25.0)
        audio_analysis = audio_optimizer.analyze_audio_characteristics(extracted_audio)
        
        # 加载音频数据进行处理
        import librosa
        audio_data, sr = librosa.load(extracted_audio, sr=48000)
        
        # 应用降噪
        reporter.report_progress(50.0)
        denoised_audio = audio_optimizer.apply_noise_reduction(audio_data)
        
        # 优化对话清晰度
        reporter.report_progress(75.0)
        enhanced_audio = audio_optimizer.optimize_dialogue_clarity(denoised_audio)
        
        # 保存处理后的音频
        import soundfile as sf
        processed_audio_file = os.path.join(workspace_dir, "processed_audio.wav")
        sf.write(processed_audio_file, enhanced_audio, sr)
        
        # 验证音频质量
        quality_metrics = audio_optimizer.validate_audio_quality(audio_data, enhanced_audio)
        
        reporter.report_log("INFO", f"音频处理完成，质量指标: SNR改善={quality_metrics.snr_improvement:.2f}dB")
        reporter.report_progress(100.0)
        
        return processed_audio_file
        
    except Exception as e:
        reporter.report_log("ERROR", f"音频处理失败: {str(e)}")
        raise


async def process_video_assembly(config: Dict[str, Any], reporter: ProgressReporter, 
                                enhanced_frames_dir: str, processed_audio_file: str) -> str:
    """执行视频重组阶段"""
    reporter.report_log("INFO", "开始视频重组阶段")
    
    try:
        # 获取视频处理器
        video_processor = container.get(IVideoProcessor)
        progress_tracker = CustomProgressTracker(reporter)
        
        # 设置进度跟踪器
        if hasattr(video_processor, 'progress_tracker'):
            video_processor.progress_tracker = progress_tracker
        
        # 获取增强后的帧文件
        enhanced_frames = []
        if os.path.exists(enhanced_frames_dir):
            enhanced_frames = sorted([
                os.path.join(enhanced_frames_dir, f) 
                for f in os.listdir(enhanced_frames_dir) 
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
        
        reporter.report_log("INFO", f"找到 {len(enhanced_frames)} 个增强帧进行重组")
        
        if not enhanced_frames:
            raise ValueError("没有找到增强后的帧文件")
        
        # 重组视频（不含音频）
        reporter.report_progress(25.0)
        temp_video_path = os.path.join(config["workspace_dir"], "temp_video.mp4")
        video_only_path = video_processor.reassemble_video(enhanced_frames, temp_video_path)
        
        if not video_only_path:
            raise ValueError("视频重组失败")
        
        reporter.report_log("INFO", f"视频重组完成: {video_only_path}")
        
        # 合并音频和视频
        reporter.report_progress(75.0)
        final_output_path = config["output_path"]
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        
        merged_video_path = video_processor.merge_audio_video(
            video_only_path, processed_audio_file, final_output_path
        )
        
        if not merged_video_path:
            raise ValueError("音视频合并失败")
        
        reporter.report_log("INFO", f"最终视频生成完成: {merged_video_path}")
        reporter.report_progress(100.0)
        
        # 验证最终输出
        validation_result = video_processor.validate_final_output(
            merged_video_path, config["input_file"]
        )
        
        if validation_result.get("success", False):
            reporter.report_log("INFO", "最终输出验证通过")
        else:
            reporter.report_log("WARNING", f"输出验证警告: {validation_result}")
        
        return merged_video_path
        
    except Exception as e:
        reporter.report_log("ERROR", f"视频重组失败: {str(e)}")
        raise


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Python视频处理桥接脚本")
    parser.add_argument("--task-id", required=True, help="任务ID")
    parser.add_argument("--stage", required=True, help="处理阶段")
    parser.add_argument("--input-file", required=True, help="输入文件路径")
    parser.add_argument("--output-path", required=True, help="输出文件路径")
    parser.add_argument("--workspace", required=True, help="工作目录")
    parser.add_argument("--config", required=True, help="配置JSON字符串")
    
    # 阶段特定参数
    parser.add_argument("--extract-frames", action="store_true", help="执行帧提取")
    parser.add_argument("--ai-model", help="AI模型名称")
    parser.add_argument("--target-resolution", help="目标分辨率")
    parser.add_argument("--audio-config", help="音频配置JSON")
    parser.add_argument("--assemble-video", action="store_true", help="执行视频重组")
    
    args = parser.parse_args()
    
    # 解析配置
    try:
        config = json.loads(args.config)
        config["input_file"] = args.input_file
        config["output_path"] = args.output_path
        config["workspace_dir"] = args.workspace
    except json.JSONDecodeError as e:
        print(f"ERROR: 配置JSON解析失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 创建进度报告器
    reporter = ProgressReporter(args.task_id, args.stage)
    
    try:
        # 根据阶段执行相应的处理
        if args.stage == "FrameExtraction":
            result = await process_frame_extraction(config, reporter)
            reporter.report_log("INFO", f"帧提取阶段完成: {result}")
            
        elif args.stage == "AiEnhancement":
            frames_dir = os.path.join(args.workspace, "frames")
            result = await process_ai_enhancement(config, reporter, frames_dir)
            reporter.report_log("INFO", f"AI增强阶段完成: {result}")
            
        elif args.stage == "AudioProcessing":
            result = await process_audio_optimization(config, reporter)
            reporter.report_log("INFO", f"音频处理阶段完成: {result}")
            
        elif args.stage == "VideoAssembly":
            enhanced_frames_dir = os.path.join(args.workspace, "enhanced")
            processed_audio_file = os.path.join(args.workspace, "processed_audio.wav")
            result = await process_video_assembly(config, reporter, enhanced_frames_dir, processed_audio_file)
            reporter.report_log("INFO", f"视频重组阶段完成: {result}")
            
        else:
            raise ValueError(f"未知的处理阶段: {args.stage}")
        
        # 成功完成
        reporter.report_log("INFO", f"阶段 {args.stage} 处理成功完成")
        sys.exit(0)
        
    except Exception as e:
        reporter.report_log("ERROR", f"处理失败: {str(e)}")
        reporter.report_log("ERROR", f"错误详情: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())