#!/usr/bin/env python3
"""
Task 7.1: Audio Quality Analysis and Assessment
Simple implementation for theater drama audio analysis
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def analyze_theater_audio_simple():
    """执行Task 7.1的音频质量分析"""
    
    print("🎵 Task 7.1: 话剧音频质量分析开始")
    
    video_file = "videos/大学生原创话剧《自杀既遂》.mp4"
    
    if not Path(video_file).exists():
        print(f"❌ 视频文件不存在: {video_file}")
        return
    
    # 创建工作目录
    workspace = Path("audio_workspace")
    workspace.mkdir(exist_ok=True)
    
    try:
        print("步骤1: 加载音频数据进行分析...")
        
        # 使用librosa直接从视频加载音频 (前60秒用于分析)
        import librosa
        import soundfile as sf
        
        print("  - 正在从视频文件加载音频数据...")
        y, sr = librosa.load(video_file, sr=48000, duration=60)  # 只加载前60秒进行分析
        
        duration = len(y) / sr
        print(f"  ✅ 音频加载成功: {duration:.1f}秒, 采样率: {sr} Hz")
        
        # 保存临时音频文件用于后续分析
        temp_audio_file = workspace / "sample_audio_60s.wav"
        sf.write(temp_audio_file, y, sr)
        print(f"  - 样本音频已保存: {temp_audio_file}")
        
        print("\n步骤2: 基本音频特征分析...")
        
        # 基本统计信息
        rms_energy = np.sqrt(np.mean(y**2))
        peak_amplitude = np.max(np.abs(y))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        
        print(f"  - RMS能量: {rms_energy:.6f}")
        print(f"  - 峰值振幅: {peak_amplitude:.6f}")
        print(f"  - 过零率: {zero_crossing_rate:.6f}")
        
        # 动态范围分析
        dynamic_range_db = 20 * np.log10(peak_amplitude / (rms_energy + 1e-10))
        print(f"  - 动态范围: {dynamic_range_db:.1f} dB")
        
        print("\n步骤3: 频谱特征分析...")
        
        # 频谱特征
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        print(f"  - 频谱质心均值: {np.mean(spectral_centroids):.1f} Hz")
        print(f"  - 频谱带宽均值: {np.mean(spectral_bandwidth):.1f} Hz")
        print(f"  - 频谱滚降均值: {np.mean(spectral_rolloff):.1f} Hz")
        
        print("\n步骤4: 噪音和信噪比分析...")
        
        # 噪音分析
        silence_threshold = np.percentile(np.abs(y), 20)
        silence_mask = np.abs(y) < silence_threshold
        noise_floor = np.mean(np.abs(y[silence_mask])) if np.any(silence_mask) else silence_threshold
        
        # 估算信噪比
        signal_power = rms_energy**2
        noise_power = noise_floor**2
        snr_estimate = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        print(f"  - 噪音底噪: {noise_floor:.6f}")
        print(f"  - 静音比例: {np.sum(silence_mask) / len(silence_mask):.1%}")
        print(f"  - 估算SNR: {snr_estimate:.1f} dB")
        
        print("\n步骤5: 话剧特有音频特征分析...")
        
        # 语音活动检测
        frame_length = 2048
        hop_length = 512
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        frame_energy = np.sum(frames**2, axis=0)
        energy_threshold = np.percentile(frame_energy, 60)
        speech_frames = frame_energy > energy_threshold
        speech_ratio = np.sum(speech_frames) / len(speech_frames)
        
        print(f"  - 语音活动比例: {speech_ratio:.1%}")
        
        # 频段能量分析
        stft = librosa.stft(y, hop_length=hop_length)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr)
        
        # 定义话剧关键频段
        freq_bands = {
            "低频": (20, 250),
            "中频": (250, 2000),
            "人声关键": (2000, 4000),
            "清晰度": (4000, 8000),
            "高频": (8000, 20000)
        }
        
        freq_energy = {}
        for band_name, (low_freq, high_freq) in freq_bands.items():
            band_indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
            if len(band_indices) > 0:
                band_energy = np.mean(magnitude[band_indices, :])
                freq_energy[band_name] = float(band_energy)
                print(f"  - {band_name}频段能量: {band_energy:.4f}")
        
        print("\n步骤6: 音频质量评估...")
        
        # 质量评估
        dialogue_prominence = freq_energy.get("人声关键", 0) / (freq_energy.get("低频", 1) + 1e-10)
        
        # 整体质量评估
        if snr_estimate > 25 and dynamic_range_db > 15:
            overall_quality = "excellent"
        elif snr_estimate > 15 and dynamic_range_db > 10:
            overall_quality = "good"
        elif snr_estimate > 10:
            overall_quality = "fair"
        else:
            overall_quality = "poor"
        
        print(f"  - 对话突出度: {dialogue_prominence:.2f}")
        print(f"  - 整体质量评估: {overall_quality}")
        
        # 处理建议
        noise_reduction_needed = "heavy" if snr_estimate < 10 else "medium" if snr_estimate < 20 else "light"
        eq_needed = freq_energy.get("低频", 0) > freq_energy.get("中频", 0)
        compression_needed = dynamic_range_db > 20
        
        print(f"  - 建议降噪强度: {noise_reduction_needed}")
        print(f"  - 需要EQ调整: {'是' if eq_needed else '否'}")
        print(f"  - 需要动态压缩: {'是' if compression_needed else '否'}")
        
        print("\n步骤7: 生成分析报告...")
        
        # 汇总分析结果
        analysis_result = {
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
                "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff))
            },
            "noise_analysis": {
                "noise_floor": float(noise_floor),
                "silence_ratio": float(np.sum(silence_mask) / len(silence_mask)),
                "snr_estimate_db": float(snr_estimate)
            },
            "frequency_energy_distribution": freq_energy,
            "theater_specific": {
                "speech_activity_ratio": float(speech_ratio),
                "dialogue_prominence": float(dialogue_prominence),
                "ambient_noise_level": float(noise_floor)
            },
            "quality_assessment": {
                "overall_quality": overall_quality,
                "dialogue_clarity": "good" if dialogue_prominence > 1.0 else "needs_improvement",
                "background_noise_level": "low" if snr_estimate > 20 else "moderate" if snr_estimate > 10 else "high",
                "dynamic_range_type": "wide" if dynamic_range_db > 15 else "moderate" if dynamic_range_db > 10 else "compressed"
            },
            "processing_recommendations": {
                "noise_reduction": noise_reduction_needed,
                "eq_adjustments": {
                    "bass_cut": freq_energy.get("低频", 0) > freq_energy.get("中频", 0),
                    "speech_boost": freq_energy.get("人声关键", 0) < freq_energy.get("中频", 0),
                    "presence_boost": freq_energy.get("清晰度", 0) < freq_energy.get("人声关键", 0)
                },
                "dynamics_processing": {
                    "compression_needed": compression_needed,
                    "limiting_needed": peak_amplitude > 0.9
                }
            }
        }
        
        # 保存JSON报告
        report_file = workspace / "task_7_1_audio_analysis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ JSON报告已保存: {report_file}")
        
        # 生成简单的可视化图表
        print("\n步骤8: 生成分析图表...")
        
        plt.figure(figsize=(12, 8))
        
        # 波形图
        plt.subplot(2, 2, 1)
        time = np.linspace(0, duration, len(y))
        plt.plot(time, y, alpha=0.7)
        plt.title('音频波形')
        plt.xlabel('时间 (秒)')
        plt.ylabel('振幅')
        plt.grid(True, alpha=0.3)
        
        # 频谱图
        plt.subplot(2, 2, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('频谱图')
        
        # 频段能量分布
        plt.subplot(2, 2, 3)
        bands = list(freq_energy.keys())
        energies = list(freq_energy.values())
        plt.bar(bands, energies, color='skyblue', alpha=0.7)
        plt.title('频段能量分布')
        plt.xlabel('频段')
        plt.ylabel('能量')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 质量指标雷达图
        plt.subplot(2, 2, 4)
        categories = ['SNR', '动态范围', '语音活动', '对话突出度']
        values = [
            min(snr_estimate / 30, 1.0),
            min(dynamic_range_db / 25, 1.0),
            speech_ratio,
            min(dialogue_prominence / 2, 1.0)
        ]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax = plt.subplot(2, 2, 4, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2, color='purple')
        ax.fill(angles, values, alpha=0.25, color='purple')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        plt.title('音频质量评估')
        
        plt.tight_layout()
        
        plot_file = workspace / "task_7_1_audio_analysis_plots.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 分析图表已保存: {plot_file}")
        
        # 生成Markdown报告
        print("\n步骤9: 生成Markdown报告...")
        
        markdown_content = f"""# Task 7.1: 话剧音频质量分析报告

## 分析概述
- **视频文件**: {video_file}
- **分析样本**: 前{duration:.1f}秒
- **采样率**: {sr} Hz
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 关键发现

### 整体质量评估: **{overall_quality.upper()}**

| 指标 | 数值 | 评估 |
|------|------|------|
| 估算信噪比 | {snr_estimate:.1f} dB | {analysis_result['quality_assessment']['background_noise_level']} |
| 动态范围 | {dynamic_range_db:.1f} dB | {analysis_result['quality_assessment']['dynamic_range_type']} |
| 语音活动比例 | {speech_ratio:.1%} | - |
| 对话清晰度 | - | {analysis_result['quality_assessment']['dialogue_clarity']} |

## 详细分析结果

### 基础音频特征
- **RMS能量**: {rms_energy:.6f}
- **峰值振幅**: {peak_amplitude:.6f}
- **过零率**: {zero_crossing_rate:.6f}

### 频谱特征
- **频谱质心**: {np.mean(spectral_centroids):.1f} Hz
- **频谱带宽**: {np.mean(spectral_bandwidth):.1f} Hz
- **频谱滚降**: {np.mean(spectral_rolloff):.1f} Hz

### 噪音分析
- **噪音底噪**: {noise_floor:.6f}
- **静音比例**: {np.sum(silence_mask) / len(silence_mask):.1%}
- **估算SNR**: {snr_estimate:.1f} dB

### 频段能量分布
"""
        
        for band, energy in freq_energy.items():
            markdown_content += f"- **{band}**: {energy:.4f}\n"
        
        markdown_content += f"""
### 话剧特有特征
- **语音活动比例**: {speech_ratio:.1%}
- **对话突出度**: {dialogue_prominence:.2f}
- **环境噪音水平**: {noise_floor:.6f}

## 处理建议

### 降噪处理
- **推荐强度**: {noise_reduction_needed}

### 均衡器调整
"""
        
        eq_adj = analysis_result['processing_recommendations']['eq_adjustments']
        if eq_adj['bass_cut']:
            markdown_content += "- ✅ 建议进行低频衰减\n"
        if eq_adj['speech_boost']:
            markdown_content += "- ✅ 建议增强人声频段\n"
        if eq_adj['presence_boost']:
            markdown_content += "- ✅ 建议提升存在感频段\n"
        
        dynamics = analysis_result['processing_recommendations']['dynamics_processing']
        markdown_content += f"""
### 动态处理
- **需要压缩**: {'是' if dynamics['compression_needed'] else '否'}
- **需要限幅**: {'是' if dynamics['limiting_needed'] else '否'}

## 结论

基于对话剧音频前{duration:.1f}秒的分析，该音频文件的整体质量为 **{overall_quality}**。

主要建议：
1. 应用 **{noise_reduction_needed}** 强度的降噪处理
2. {'进行EQ调整以优化人声清晰度' if eq_needed else '当前频率分布较为合理'}
3. {'应用动态压缩以平衡音量变化' if compression_needed else '动态范围适中'}

---
*本报告基于Task 7.1的音频质量分析要求生成*
"""
        
        markdown_file = workspace / "task_7_1_audio_quality_assessment_report.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"  ✅ Markdown报告已保存: {markdown_file}")
        
        print(f"\n🎉 Task 7.1 音频质量分析完成!")
        print(f"📊 主要结果:")
        print(f"  - 整体质量: {overall_quality}")
        print(f"  - 估算SNR: {snr_estimate:.1f}dB")
        print(f"  - 动态范围: {dynamic_range_db:.1f}dB")
        print(f"  - 语音活动: {speech_ratio:.1%}")
        print(f"📁 所有报告文件已保存到: {workspace}/")
        
        return analysis_result
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyze_theater_audio_simple()