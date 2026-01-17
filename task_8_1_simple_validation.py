#!/usr/bin/env python3
"""
任务8.1: 音频增强效果分析 - 简化版本
快速完成音频质量验证和对比分析
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import librosa
import librosa.display

def analyze_audio_simple(audio_path):
    """简化的音频分析"""
    print(f"🎵 分析音频: {audio_path}")
    
    # 加载音频
    y, sr = librosa.load(str(audio_path), sr=None)
    duration = len(y) / sr
    
    # 基本统计
    rms_energy = np.sqrt(np.mean(y**2))
    peak_amplitude = np.max(np.abs(y))
    dynamic_range = 20 * np.log10(peak_amplitude / (rms_energy + 1e-10))
    
    # 频谱分析
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    # 简单SNR估计
    stft = librosa.stft(y)
    magnitude = np.abs(stft)
    sorted_magnitude = np.sort(magnitude.flatten())
    noise_floor = np.mean(sorted_magnitude[:int(len(sorted_magnitude) * 0.1)])
    signal_power = np.mean(magnitude)
    snr_estimate = 20 * np.log10(signal_power / (noise_floor + 1e-10))
    
    return {
        "duration": duration,
        "rms_energy": rms_energy,
        "peak_amplitude": peak_amplitude,
        "dynamic_range_db": dynamic_range,
        "spectral_centroid_mean": np.mean(spectral_centroid),
        "estimated_snr_db": snr_estimate
    }, y, sr

def create_comparison_plot(original_data, enhanced_data, original_sr, enhanced_sr):
    """创建对比图"""
    print("📊 生成对比图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('音频增强效果对比', fontsize=14)
    
    # 频谱图对比
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original_data)), ref=np.max)
    D_enh = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_data)), ref=np.max)
    
    librosa.display.specshow(D_orig, sr=original_sr, x_axis='time', y_axis='hz', ax=axes[0,0])
    axes[0,0].set_title('原始音频频谱')
    
    librosa.display.specshow(D_enh, sr=enhanced_sr, x_axis='time', y_axis='hz', ax=axes[0,1])
    axes[0,1].set_title('增强音频频谱')
    
    # 频率响应对比
    freqs_orig = np.fft.rfftfreq(len(original_data), 1/original_sr)
    fft_orig = np.abs(np.fft.rfft(original_data))
    freqs_enh = np.fft.rfftfreq(len(enhanced_data), 1/enhanced_sr)
    fft_enh = np.abs(np.fft.rfft(enhanced_data))
    
    axes[1,0].semilogx(freqs_orig, 20*np.log10(fft_orig + 1e-10), label='原始', alpha=0.7)
    axes[1,0].semilogx(freqs_enh, 20*np.log10(fft_enh + 1e-10), label='增强', alpha=0.7)
    axes[1,0].set_xlabel('频率 (Hz)')
    axes[1,0].set_ylabel('幅度 (dB)')
    axes[1,0].set_title('频率响应对比')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 时域波形对比 (前5秒)
    max_samples_orig = min(len(original_data), int(5 * original_sr))
    max_samples_enh = min(len(enhanced_data), int(5 * enhanced_sr))
    
    time_orig = np.linspace(0, max_samples_orig/original_sr, max_samples_orig)
    time_enh = np.linspace(0, max_samples_enh/enhanced_sr, max_samples_enh)
    
    axes[1,1].plot(time_orig, original_data[:max_samples_orig], label='原始', alpha=0.7, linewidth=0.5)
    axes[1,1].plot(time_enh, enhanced_data[:max_samples_enh], label='增强', alpha=0.7, linewidth=0.5)
    axes[1,1].set_xlabel('时间 (秒)')
    axes[1,1].set_ylabel('幅度')
    axes[1,1].set_title('波形对比 (前5秒)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = "task_8_1_audio_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 对比图已保存: {plot_path}")
    return plot_path

def main():
    print("🚀 开始任务8.1: 音频增强效果分析")
    
    # 文件路径
    original_audio = Path("audio_workspace/original_audio.wav")
    enhanced_audio = Path("audio_workspace/acoustics_optimized_audio.wav")
    
    # 分析原始音频
    orig_analysis, orig_data, orig_sr = analyze_audio_simple(original_audio)
    
    # 分析增强音频
    enh_analysis, enh_data, enh_sr = analyze_audio_simple(enhanced_audio)
    
    # 计算改善指标
    snr_improvement = enh_analysis["estimated_snr_db"] - orig_analysis["estimated_snr_db"]
    dr_improvement = enh_analysis["dynamic_range_db"] - orig_analysis["dynamic_range_db"]
    centroid_change = enh_analysis["spectral_centroid_mean"] - orig_analysis["spectral_centroid_mean"]
    
    # 生成对比图
    plot_path = create_comparison_plot(orig_data, enh_data, orig_sr, enh_sr)
    
    # 生成报告
    report = {
        "task": "8.1 Audio enhancement effectiveness analysis",
        "timestamp": datetime.now().isoformat(),
        "original_audio": {
            "file": str(original_audio),
            "duration_seconds": orig_analysis["duration"],
            "dynamic_range_db": orig_analysis["dynamic_range_db"],
            "estimated_snr_db": orig_analysis["estimated_snr_db"],
            "spectral_centroid_hz": orig_analysis["spectral_centroid_mean"]
        },
        "enhanced_audio": {
            "file": str(enhanced_audio),
            "duration_seconds": enh_analysis["duration"],
            "dynamic_range_db": enh_analysis["dynamic_range_db"],
            "estimated_snr_db": enh_analysis["estimated_snr_db"],
            "spectral_centroid_hz": enh_analysis["spectral_centroid_mean"]
        },
        "improvements": {
            "snr_improvement_db": snr_improvement,
            "dynamic_range_improvement_db": dr_improvement,
            "spectral_centroid_change_hz": centroid_change
        },
        "assessment": {
            "snr_status": "改善" if snr_improvement > 1 else "轻微改善" if snr_improvement > 0 else "无明显改善",
            "dynamic_range_status": "改善" if dr_improvement > 1 else "轻微改善" if dr_improvement > 0 else "无明显改善",
            "overall_quality": "显著改善" if (snr_improvement > 3 and dr_improvement > 2) else "中等改善" if (snr_improvement > 1 or dr_improvement > 1) else "轻微改善"
        },
        "visualization": plot_path
    }
    
    # 保存报告
    report_path = "task_8_1_audio_enhancement_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print("\n📊 音频增强效果分析结果:")
    print(f"   SNR改善: {snr_improvement:.1f}dB ({report['assessment']['snr_status']})")
    print(f"   动态范围改善: {dr_improvement:.1f}dB ({report['assessment']['dynamic_range_status']})")
    print(f"   频谱重心变化: {centroid_change:.1f}Hz")
    print(f"   整体质量评估: {report['assessment']['overall_quality']}")
    print(f"   报告文件: {report_path}")
    print(f"   对比图: {plot_path}")
    
    print("\n✅ 任务8.1完成: 音频增强效果分析")
    return True

if __name__ == "__main__":
    main()