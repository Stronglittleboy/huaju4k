#!/usr/bin/env python3
"""
Task 7.1: Basic Audio Quality Analysis
Simplified version without heavy plotting
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

def basic_audio_analysis():
    """执行基础音频分析"""
    
    print("🎵 Task 7.1: 话剧音频质量分析 (基础版)")
    
    video_file = "videos/大学生原创话剧《自杀既遂》.mp4"
    
    if not Path(video_file).exists():
        print(f"❌ 视频文件不存在: {video_file}")
        return
    
    # 创建工作目录
    workspace = Path("audio_workspace")
    workspace.mkdir(exist_ok=True)
    
    try:
        print("步骤1: 加载音频数据...")
        
        # 使用librosa加载音频 (前30秒)
        import librosa
        
        print("  - 正在加载音频数据 (前30秒)...")
        y, sr = librosa.load(video_file, sr=48000, duration=30)
        
        duration = len(y) / sr
        print(f"  ✅ 音频加载成功: {duration:.1f}秒, 采样率: {sr} Hz")
        
        print("\n步骤2: 基本特征分析...")
        
        # 基本统计
        rms_energy = np.sqrt(np.mean(y**2))
        peak_amplitude = np.max(np.abs(y))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        dynamic_range_db = 20 * np.log10(peak_amplitude / (rms_energy + 1e-10))
        
        print(f"  - RMS能量: {rms_energy:.6f}")
        print(f"  - 峰值振幅: {peak_amplitude:.6f}")
        print(f"  - 动态范围: {dynamic_range_db:.1f} dB")
        
        print("\n步骤3: 频谱分析...")
        
        # 频谱特征
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        print(f"  - 频谱质心: {np.mean(spectral_centroids):.1f} Hz")
        print(f"  - 频谱带宽: {np.mean(spectral_bandwidth):.1f} Hz")
        
        print("\n步骤4: 噪音分析...")
        
        # 噪音估算
        silence_threshold = np.percentile(np.abs(y), 20)
        silence_mask = np.abs(y) < silence_threshold
        noise_floor = np.mean(np.abs(y[silence_mask])) if np.any(silence_mask) else silence_threshold
        
        signal_power = rms_energy**2
        noise_power = noise_floor**2
        snr_estimate = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        print(f"  - 噪音底噪: {noise_floor:.6f}")
        print(f"  - 估算SNR: {snr_estimate:.1f} dB")
        
        print("\n步骤5: 语音活动检测...")
        
        # 简单的语音活动检测
        frame_length = 2048
        hop_length = 512
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        frame_energy = np.sum(frames**2, axis=0)
        energy_threshold = np.percentile(frame_energy, 60)
        speech_frames = frame_energy > energy_threshold
        speech_ratio = np.sum(speech_frames) / len(speech_frames)
        
        print(f"  - 语音活动比例: {speech_ratio:.1%}")
        
        print("\n步骤6: 质量评估...")
        
        # 质量评估
        if snr_estimate > 25 and dynamic_range_db > 15:
            overall_quality = "excellent"
        elif snr_estimate > 15 and dynamic_range_db > 10:
            overall_quality = "good"
        elif snr_estimate > 10:
            overall_quality = "fair"
        else:
            overall_quality = "poor"
        
        print(f"  - 整体质量: {overall_quality}")
        
        # 处理建议
        noise_reduction = "heavy" if snr_estimate < 10 else "medium" if snr_estimate < 20 else "light"
        print(f"  - 建议降噪: {noise_reduction}")
        
        print("\n步骤7: 生成报告...")
        
        # 分析结果
        analysis_result = {
            "task": "7.1 Audio quality analysis and assessment",
            "file_info": {
                "file_path": str(video_file),
                "sample_duration_seconds": float(duration),
                "sample_rate": int(sr),
                "analysis_timestamp": datetime.now().isoformat()
            },
            "basic_metrics": {
                "rms_energy": float(rms_energy),
                "peak_amplitude": float(peak_amplitude),
                "dynamic_range_db": float(dynamic_range_db),
                "zero_crossing_rate": float(zero_crossing_rate),
                "estimated_snr_db": float(snr_estimate)
            },
            "spectral_features": {
                "spectral_centroid_mean": float(np.mean(spectral_centroids)),
                "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth))
            },
            "noise_analysis": {
                "noise_floor": float(noise_floor),
                "silence_ratio": float(np.sum(silence_mask) / len(silence_mask)),
                "snr_estimate_db": float(snr_estimate)
            },
            "theater_specific": {
                "speech_activity_ratio": float(speech_ratio)
            },
            "quality_assessment": {
                "overall_quality": overall_quality,
                "background_noise_level": "low" if snr_estimate > 20 else "moderate" if snr_estimate > 10 else "high",
                "dynamic_range_type": "wide" if dynamic_range_db > 15 else "moderate" if dynamic_range_db > 10 else "compressed"
            },
            "processing_recommendations": {
                "noise_reduction": noise_reduction,
                "compression_needed": bool(dynamic_range_db > 20),
                "limiting_needed": bool(peak_amplitude > 0.9)
            }
        }
        
        # 保存报告
        report_file = workspace / "task_7_1_basic_audio_analysis.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ 报告已保存: {report_file}")
        
        # 生成简单的Markdown报告
        markdown_content = f"""# Task 7.1: 话剧音频质量分析报告

## 基本信息
- **文件**: {video_file}
- **分析样本**: 前{duration:.1f}秒
- **采样率**: {sr} Hz
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 分析结果

### 整体质量: **{overall_quality.upper()}**

| 指标 | 数值 |
|------|------|
| RMS能量 | {rms_energy:.6f} |
| 峰值振幅 | {peak_amplitude:.6f} |
| 动态范围 | {dynamic_range_db:.1f} dB |
| 估算SNR | {snr_estimate:.1f} dB |
| 语音活动比例 | {speech_ratio:.1%} |

### 频谱特征
- **频谱质心**: {np.mean(spectral_centroids):.1f} Hz
- **频谱带宽**: {np.mean(spectral_bandwidth):.1f} Hz

### 质量评估
- **整体质量**: {overall_quality}
- **背景噪音**: {analysis_result['quality_assessment']['background_noise_level']}
- **动态范围**: {analysis_result['quality_assessment']['dynamic_range_type']}

### 处理建议
- **降噪强度**: {noise_reduction}
- **需要压缩**: {'是' if analysis_result['processing_recommendations']['compression_needed'] else '否'}
- **需要限幅**: {'是' if analysis_result['processing_recommendations']['limiting_needed'] else '否'}

## 结论

基于对话剧音频的分析，该文件的音频质量为 **{overall_quality}**，建议进行 **{noise_reduction}** 强度的降噪处理。

---
*Task 7.1 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        markdown_file = workspace / "task_7_1_audio_analysis_report.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"  ✅ Markdown报告已保存: {markdown_file}")
        
        print(f"\n🎉 Task 7.1 完成!")
        print(f"📊 关键结果:")
        print(f"  - 整体质量: {overall_quality}")
        print(f"  - SNR: {snr_estimate:.1f}dB")
        print(f"  - 动态范围: {dynamic_range_db:.1f}dB")
        print(f"  - 语音活动: {speech_ratio:.1%}")
        
        return analysis_result
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    basic_audio_analysis()