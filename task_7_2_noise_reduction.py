#!/usr/bin/env python3
"""
Task 7.2: Intelligent Noise Reduction Processing
智能降噪处理 - 针对话剧音频的专业降噪
"""

import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

def run_ffmpeg_command(command, description=""):
    """执行FFmpeg命令"""
    print(f"  - {description}")
    print(f"    命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"    输出: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ❌ 错误: {e}")
        if e.stderr:
            print(f"    错误信息: {e.stderr}")
        return False

def intelligent_noise_reduction():
    """执行智能降噪处理"""
    
    print("🔧 Task 7.2: 智能降噪处理")
    
    video_file = "videos/大学生原创话剧《自杀既遂》.mp4"
    workspace = Path("audio_workspace")
    workspace.mkdir(exist_ok=True)
    
    # 读取Task 7.1的分析结果
    analysis_file = workspace / "task_7_1_basic_audio_analysis.json"
    if not analysis_file.exists():
        print("❌ 未找到Task 7.1的分析结果，请先运行Task 7.1")
        return False
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    snr_db = analysis_data['basic_metrics']['estimated_snr_db']
    noise_reduction_level = analysis_data['processing_recommendations']['noise_reduction']
    
    print(f"📊 基于Task 7.1分析结果:")
    print(f"  - 当前SNR: {snr_db:.1f} dB")
    print(f"  - 推荐降噪强度: {noise_reduction_level}")
    
    try:
        print("\n步骤1: 从视频中提取音频...")
        
        # 提取原始音频
        original_audio = workspace / "original_audio.wav"
        extract_cmd = f'ffmpeg -i "{video_file}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{original_audio}" -y'
        
        if not run_ffmpeg_command(extract_cmd, "提取原始音频"):
            return False
        
        print(f"  ✅ 原始音频已提取: {original_audio}")
        
        print("\n步骤2: 应用智能降噪处理...")
        
        # 根据分析结果设置降噪参数
        if noise_reduction_level == "light":
            # 轻度降噪 - 保持音质，轻微降噪
            nr_strength = 0.5
            noise_floor = -45
            print("  - 应用轻度降噪设置 (保持高音质)")
        elif noise_reduction_level == "medium":
            # 中度降噪 - 平衡音质和降噪效果
            nr_strength = 1.0
            noise_floor = -40
            print("  - 应用中度降噪设置 (平衡处理)")
        else:  # heavy
            # 重度降噪 - 强力降噪，可能轻微影响音质
            nr_strength = 1.5
            noise_floor = -35
            print("  - 应用重度降噪设置 (强力降噪)")
        
        # 第一阶段：使用FFmpeg的afftdn滤镜进行频域降噪
        stage1_audio = workspace / "stage1_denoised.wav"
        denoise_cmd = f'ffmpeg -i "{original_audio}" -af "afftdn=nr={nr_strength}:nf={noise_floor}:tn=1" "{stage1_audio}" -y'
        
        if not run_ffmpeg_command(denoise_cmd, f"第一阶段降噪 (强度: {nr_strength}, 噪音底噪: {noise_floor}dB)"):
            return False
        
        print("  ✅ 第一阶段降噪完成")
        
        # 第二阶段：使用高通滤波器去除低频噪音
        stage2_audio = workspace / "stage2_highpass.wav"
        highpass_cmd = f'ffmpeg -i "{stage1_audio}" -af "highpass=f=80:p=1" "{stage2_audio}" -y'
        
        if not run_ffmpeg_command(highpass_cmd, "第二阶段高通滤波 (去除80Hz以下低频噪音)"):
            return False
        
        print("  ✅ 第二阶段高通滤波完成")
        
        # 第三阶段：使用门限降噪进一步处理
        final_denoised = workspace / "denoised_audio.wav"
        
        # 计算门限值 (基于分析结果)
        noise_floor_linear = analysis_data['noise_analysis']['noise_floor']
        gate_threshold = max(-60, 20 * np.log10(noise_floor_linear * 3))  # 设置为噪音底噪的3倍
        
        gate_cmd = f'ffmpeg -i "{stage2_audio}" -af "agate=threshold={gate_threshold}dB:ratio=2:attack=1:release=10" "{final_denoised}" -y'
        
        if not run_ffmpeg_command(gate_cmd, f"第三阶段门限降噪 (门限: {gate_threshold:.1f}dB)"):
            return False
        
        print("  ✅ 第三阶段门限降噪完成")
        
        print("\n步骤3: 降噪效果验证...")
        
        # 使用librosa分析降噪前后的效果
        try:
            import librosa
            
            # 分析原始音频
            print("  - 分析原始音频...")
            y_orig, sr = librosa.load(str(original_audio), sr=48000, duration=30)
            rms_orig = np.sqrt(np.mean(y_orig**2))
            
            # 估算原始噪音
            silence_threshold_orig = np.percentile(np.abs(y_orig), 20)
            silence_mask_orig = np.abs(y_orig) < silence_threshold_orig
            noise_floor_orig = np.mean(np.abs(y_orig[silence_mask_orig])) if np.any(silence_mask_orig) else silence_threshold_orig
            snr_orig = 10 * np.log10((rms_orig**2) / (noise_floor_orig**2 + 1e-10))
            
            # 分析降噪后音频
            print("  - 分析降噪后音频...")
            y_denoised, _ = librosa.load(str(final_denoised), sr=48000, duration=30)
            rms_denoised = np.sqrt(np.mean(y_denoised**2))
            
            # 估算降噪后噪音
            silence_threshold_denoised = np.percentile(np.abs(y_denoised), 20)
            silence_mask_denoised = np.abs(y_denoised) < silence_threshold_denoised
            noise_floor_denoised = np.mean(np.abs(y_denoised[silence_mask_denoised])) if np.any(silence_mask_denoised) else silence_threshold_denoised
            snr_denoised = 10 * np.log10((rms_denoised**2) / (noise_floor_denoised**2 + 1e-10))
            
            # 计算改善效果
            snr_improvement = snr_denoised - snr_orig
            noise_reduction_db = 20 * np.log10(noise_floor_orig / (noise_floor_denoised + 1e-10))
            
            print(f"  📊 降噪效果分析:")
            print(f"    - 原始SNR: {snr_orig:.1f} dB")
            print(f"    - 降噪后SNR: {snr_denoised:.1f} dB")
            print(f"    - SNR改善: {snr_improvement:.1f} dB")
            print(f"    - 噪音降低: {noise_reduction_db:.1f} dB")
            
            # 评估降噪质量
            if snr_improvement > 2:
                quality_assessment = "显著改善"
            elif snr_improvement > 0.5:
                quality_assessment = "适度改善"
            elif snr_improvement > -0.5:
                quality_assessment = "基本保持"
            else:
                quality_assessment = "可能过度处理"
            
            print(f"    - 质量评估: {quality_assessment}")
            
        except Exception as e:
            print(f"  ⚠️ 降噪效果分析失败: {e}")
            snr_improvement = 0
            noise_reduction_db = 0
            quality_assessment = "未知"
        
        print("\n步骤4: 生成处理报告...")
        
        # 生成降噪处理报告
        processing_report = {
            "task": "7.2 Intelligent noise reduction processing",
            "input_file": str(video_file),
            "output_file": str(final_denoised),
            "processing_timestamp": datetime.now().isoformat(),
            "original_analysis": {
                "snr_db": analysis_data['basic_metrics']['estimated_snr_db'],
                "noise_floor": analysis_data['noise_analysis']['noise_floor'],
                "recommended_level": noise_reduction_level
            },
            "processing_stages": [
                {
                    "stage": 1,
                    "method": "FFmpeg afftdn (频域降噪)",
                    "parameters": {
                        "nr_strength": nr_strength,
                        "noise_floor_db": noise_floor
                    }
                },
                {
                    "stage": 2,
                    "method": "高通滤波器",
                    "parameters": {
                        "cutoff_frequency_hz": 80
                    }
                },
                {
                    "stage": 3,
                    "method": "门限降噪",
                    "parameters": {
                        "threshold_db": float(gate_threshold),
                        "ratio": 2
                    }
                }
            ],
            "results": {
                "snr_improvement_db": float(snr_improvement) if 'snr_improvement' in locals() else 0,
                "noise_reduction_db": float(noise_reduction_db) if 'noise_reduction_db' in locals() else 0,
                "quality_assessment": quality_assessment if 'quality_assessment' in locals() else "未评估",
                "processing_successful": True
            }
        }
        
        # 保存处理报告
        report_file = workspace / "task_7_2_noise_reduction_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(processing_report, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ 处理报告已保存: {report_file}")
        
        # 生成Markdown报告
        markdown_content = f"""# Task 7.2: 智能降噪处理报告

## 处理概述
- **输入文件**: {video_file}
- **输出文件**: {final_denoised}
- **处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **降噪级别**: {noise_reduction_level}

## 原始音频分析
- **SNR**: {analysis_data['basic_metrics']['estimated_snr_db']:.1f} dB
- **噪音底噪**: {analysis_data['noise_analysis']['noise_floor']:.6f}
- **推荐处理**: {noise_reduction_level}

## 处理流程

### 第一阶段: 频域降噪 (FFmpeg afftdn)
- **降噪强度**: {nr_strength}
- **噪音底噪**: {noise_floor} dB
- **方法**: 自适应频域降噪

### 第二阶段: 高通滤波
- **截止频率**: 80 Hz
- **目的**: 去除低频环境噪音

### 第三阶段: 门限降噪
- **门限值**: {gate_threshold:.1f} dB
- **压缩比**: 2:1
- **目的**: 进一步抑制背景噪音

## 处理结果

| 指标 | 数值 |
|------|------|
| SNR改善 | {snr_improvement:.1f} dB |
| 噪音降低 | {noise_reduction_db:.1f} dB |
| 质量评估 | {quality_assessment} |

## 结论

智能降噪处理已完成，采用了{noise_reduction_level}强度的多阶段降噪方案。处理结果显示{quality_assessment}，适合话剧音频的特点。

---
*Task 7.2 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        markdown_file = workspace / "task_7_2_noise_reduction_report.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"  ✅ Markdown报告已保存: {markdown_file}")
        
        print(f"\n🎉 Task 7.2 智能降噪处理完成!")
        print(f"📁 输出文件: {final_denoised}")
        print(f"📊 处理效果: {quality_assessment}")
        if 'snr_improvement' in locals():
            print(f"📈 SNR改善: {snr_improvement:.1f} dB")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    intelligent_noise_reduction()