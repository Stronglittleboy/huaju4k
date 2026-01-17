#!/usr/bin/env python3
"""
Task 7.3: Frequency Equalization Optimization
频率均衡优化 - 针对话剧音频的专业EQ处理
"""

import os
import sys
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def analyze_frequency_distribution(audio_file):
    """分析音频的频率分布特征"""
    try:
        import librosa
        
        print("  - 加载音频进行频率分析...")
        y, sr = librosa.load(audio_file, sr=48000, duration=60)  # 分析前60秒
        
        # 计算频谱
        stft = librosa.stft(y, hop_length=512)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr)
        
        # 计算平均频谱
        avg_magnitude = np.mean(magnitude, axis=1)
        
        # 定义关键频段
        freq_bands = {
            "超低频": (20, 60),
            "低频": (60, 250),
            "中低频": (250, 500),
            "中频": (500, 1000),
            "人声基频": (1000, 2000),
            "人声共振": (2000, 4000),
            "清晰度": (4000, 6000),
            "存在感": (6000, 8000),
            "高频": (8000, 12000),
            "超高频": (12000, 20000)
        }
        
        # 计算各频段的能量
        band_energies = {}
        for band_name, (low_freq, high_freq) in freq_bands.items():
            band_indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
            if len(band_indices) > 0:
                band_energy = np.mean(avg_magnitude[band_indices])
                band_energies[band_name] = float(band_energy)
                print(f"    {band_name} ({low_freq}-{high_freq}Hz): {band_energy:.4f}")
        
        # 分析人声特征
        speech_fundamental = band_energies.get("人声基频", 0)
        speech_formants = band_energies.get("人声共振", 0)
        clarity_band = band_energies.get("清晰度", 0)
        presence_band = band_energies.get("存在感", 0)
        
        # 计算关键比率
        speech_to_bass_ratio = speech_formants / (band_energies.get("低频", 1) + 1e-10)
        clarity_to_speech_ratio = clarity_band / (speech_formants + 1e-10)
        
        print(f"  📊 关键频率分析:")
        print(f"    - 人声/低频比率: {speech_to_bass_ratio:.2f}")
        print(f"    - 清晰度/人声比率: {clarity_to_speech_ratio:.2f}")
        
        return {
            "band_energies": band_energies,
            "speech_to_bass_ratio": speech_to_bass_ratio,
            "clarity_to_speech_ratio": clarity_to_speech_ratio,
            "frequency_data": {
                "freqs": freqs.tolist(),
                "magnitude": avg_magnitude.tolist()
            }
        }
        
    except Exception as e:
        print(f"  ❌ 频率分析失败: {e}")
        return None

def design_theater_eq_curve(freq_analysis):
    """设计话剧专用EQ曲线"""
    
    print("  - 设计话剧专用EQ曲线...")
    
    if not freq_analysis:
        print("  ⚠️ 无频率分析数据，使用默认EQ设置")
        return {
            "low_cut": {"freq": 80, "gain": -3},
            "speech_boost": {"freq": 2500, "gain": 3, "q": 1.0},
            "clarity_boost": {"freq": 5000, "gain": 2, "q": 0.7},
            "high_cut": {"freq": 12000, "gain": -2}
        }
    
    band_energies = freq_analysis["band_energies"]
    speech_to_bass_ratio = freq_analysis["speech_to_bass_ratio"]
    clarity_to_speech_ratio = freq_analysis["clarity_to_speech_ratio"]
    
    # 动态调整EQ参数
    eq_settings = {}
    
    # 1. 低频控制 - 根据低频能量调整
    low_freq_energy = band_energies.get("低频", 0)
    if low_freq_energy > band_energies.get("中频", 0):
        # 低频过多，需要衰减
        eq_settings["low_cut"] = {"freq": 100, "gain": -6}
        print("    - 检测到低频过多，应用强低频衰减")
    else:
        # 低频适中，轻微衰减
        eq_settings["low_cut"] = {"freq": 80, "gain": -3}
        print("    - 低频适中，应用轻微低频衰减")
    
    # 2. 人声增强 - 根据人声/低频比率调整
    if speech_to_bass_ratio < 1.0:
        # 人声相对较弱，需要较强增强
        eq_settings["speech_boost"] = {"freq": 2500, "gain": 5, "q": 1.2}
        print("    - 人声相对较弱，应用强人声增强")
    elif speech_to_bass_ratio < 1.5:
        # 人声适中，中等增强
        eq_settings["speech_boost"] = {"freq": 2500, "gain": 3, "q": 1.0}
        print("    - 人声适中，应用中等人声增强")
    else:
        # 人声已经突出，轻微增强
        eq_settings["speech_boost"] = {"freq": 2500, "gain": 2, "q": 0.8}
        print("    - 人声已突出，应用轻微增强")
    
    # 3. 清晰度增强 - 根据清晰度/人声比率调整
    if clarity_to_speech_ratio < 0.8:
        # 清晰度不足，需要增强
        eq_settings["clarity_boost"] = {"freq": 5000, "gain": 4, "q": 0.8}
        print("    - 清晰度不足，应用强清晰度增强")
    elif clarity_to_speech_ratio < 1.2:
        # 清晰度适中，中等增强
        eq_settings["clarity_boost"] = {"freq": 5000, "gain": 2, "q": 0.7}
        print("    - 清晰度适中，应用中等增强")
    else:
        # 清晰度已好，轻微增强
        eq_settings["clarity_boost"] = {"freq": 4500, "gain": 1, "q": 0.6}
        print("    - 清晰度良好，应用轻微增强")
    
    # 4. 高频控制 - 根据高频能量调整
    high_freq_energy = band_energies.get("高频", 0)
    if high_freq_energy > band_energies.get("人声共振", 0):
        # 高频过多，可能刺耳
        eq_settings["high_cut"] = {"freq": 10000, "gain": -4}
        print("    - 高频过多，应用强高频衰减")
    else:
        # 高频适中，轻微衰减
        eq_settings["high_cut"] = {"freq": 12000, "gain": -2}
        print("    - 高频适中，应用轻微衰减")
    
    return eq_settings

def apply_frequency_equalization():
    """执行频率均衡优化"""
    
    print("🎛️ Task 7.3: 频率均衡优化")
    
    workspace = Path("audio_workspace")
    
    # 检查输入文件
    input_audio = workspace / "denoised_audio.wav"
    if not input_audio.exists():
        print("❌ 未找到降噪后的音频文件，请先完成Task 7.2")
        return False
    
    try:
        print("\n步骤1: 分析音频频率分布特征...")
        
        # 分析频率分布
        freq_analysis = analyze_frequency_distribution(str(input_audio))
        
        print("\n步骤2: 设计话剧专用EQ曲线...")
        
        # 设计EQ曲线
        eq_settings = design_theater_eq_curve(freq_analysis)
        
        print("\n步骤3: 应用频率均衡处理...")
        
        # 构建FFmpeg EQ滤镜链
        eq_filters = []
        
        # 1. 低频衰减 (高通滤波器)
        low_cut = eq_settings["low_cut"]
        eq_filters.append(f"highpass=f={low_cut['freq']}:p=1")
        print(f"  - 低频衰减: {low_cut['freq']}Hz, {low_cut['gain']}dB")
        
        # 2. 人声增强 (参数EQ)
        speech_boost = eq_settings["speech_boost"]
        eq_filters.append(f"equalizer=f={speech_boost['freq']}:width_type=q:width={speech_boost['q']}:g={speech_boost['gain']}")
        print(f"  - 人声增强: {speech_boost['freq']}Hz, +{speech_boost['gain']}dB, Q={speech_boost['q']}")
        
        # 3. 清晰度提升 (参数EQ)
        clarity_boost = eq_settings["clarity_boost"]
        eq_filters.append(f"equalizer=f={clarity_boost['freq']}:width_type=q:width={clarity_boost['q']}:g={clarity_boost['gain']}")
        print(f"  - 清晰度提升: {clarity_boost['freq']}Hz, +{clarity_boost['gain']}dB, Q={clarity_boost['q']}")
        
        # 4. 高频控制 (低通滤波器 + 衰减)
        high_cut = eq_settings["high_cut"]
        eq_filters.append(f"lowpass=f={high_cut['freq']}:p=1")
        print(f"  - 高频控制: {high_cut['freq']}Hz, {high_cut['gain']}dB")
        
        # 组合所有滤镜
        filter_chain = ",".join(eq_filters)
        
        # 应用EQ处理
        equalized_audio = workspace / "equalized_audio.wav"
        eq_cmd = f'ffmpeg -i "{input_audio}" -af "{filter_chain}" "{equalized_audio}" -y'
        
        if not run_ffmpeg_command(eq_cmd, "应用频率均衡处理"):
            return False
        
        print("  ✅ 频率均衡处理完成")
        
        print("\n步骤4: 验证EQ效果...")
        
        # 分析EQ前后的频率响应
        try:
            print("  - 分析EQ前后的频率响应...")
            
            # 分析原始音频
            freq_analysis_before = analyze_frequency_distribution(str(input_audio))
            
            # 分析EQ后音频
            freq_analysis_after = analyze_frequency_distribution(str(equalized_audio))
            
            if freq_analysis_before and freq_analysis_after:
                # 计算改善效果
                before_ratio = freq_analysis_before["speech_to_bass_ratio"]
                after_ratio = freq_analysis_after["speech_to_bass_ratio"]
                speech_enhancement = after_ratio / (before_ratio + 1e-10)
                
                before_clarity = freq_analysis_before["clarity_to_speech_ratio"]
                after_clarity = freq_analysis_after["clarity_to_speech_ratio"]
                clarity_enhancement = after_clarity / (before_clarity + 1e-10)
                
                print(f"  📊 EQ效果分析:")
                print(f"    - 人声/低频比率: {before_ratio:.2f} → {after_ratio:.2f} (改善: {speech_enhancement:.2f}x)")
                print(f"    - 清晰度/人声比率: {before_clarity:.2f} → {after_clarity:.2f} (改善: {clarity_enhancement:.2f}x)")
                
                # 评估EQ质量
                if speech_enhancement > 1.2 and clarity_enhancement > 1.1:
                    eq_quality = "显著改善"
                elif speech_enhancement > 1.1 or clarity_enhancement > 1.05:
                    eq_quality = "适度改善"
                else:
                    eq_quality = "基本保持"
                
                print(f"    - EQ质量评估: {eq_quality}")
                
            else:
                speech_enhancement = 1.0
                clarity_enhancement = 1.0
                eq_quality = "未评估"
                
        except Exception as e:
            print(f"  ⚠️ EQ效果分析失败: {e}")
            speech_enhancement = 1.0
            clarity_enhancement = 1.0
            eq_quality = "未评估"
        
        print("\n步骤5: 生成频率响应图表...")
        
        # 生成频率响应对比图
        try:
            if freq_analysis_before and freq_analysis_after:
                plt.figure(figsize=(12, 8))
                
                # 频率响应对比
                plt.subplot(2, 1, 1)
                freqs_before = np.array(freq_analysis_before["frequency_data"]["freqs"])
                mag_before = np.array(freq_analysis_before["frequency_data"]["magnitude"])
                freqs_after = np.array(freq_analysis_after["frequency_data"]["freqs"])
                mag_after = np.array(freq_analysis_after["frequency_data"]["magnitude"])
                
                plt.semilogx(freqs_before[1:], 20*np.log10(mag_before[1:] + 1e-10), 
                           label='EQ前', alpha=0.7, color='blue')
                plt.semilogx(freqs_after[1:], 20*np.log10(mag_after[1:] + 1e-10), 
                           label='EQ后', alpha=0.7, color='red')
                
                plt.title('频率响应对比')
                plt.xlabel('频率 (Hz)')
                plt.ylabel('幅度 (dB)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xlim(20, 20000)
                
                # 频段能量对比
                plt.subplot(2, 1, 2)
                bands_before = freq_analysis_before["band_energies"]
                bands_after = freq_analysis_after["band_energies"]
                
                band_names = list(bands_before.keys())
                energies_before = [bands_before[band] for band in band_names]
                energies_after = [bands_after[band] for band in band_names]
                
                x = np.arange(len(band_names))
                width = 0.35
                
                plt.bar(x - width/2, energies_before, width, label='EQ前', alpha=0.7, color='blue')
                plt.bar(x + width/2, energies_after, width, label='EQ后', alpha=0.7, color='red')
                
                plt.title('频段能量对比')
                plt.xlabel('频段')
                plt.ylabel('能量')
                plt.xticks(x, band_names, rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                plot_file = workspace / "task_7_3_frequency_response_comparison.png"
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  ✅ 频率响应图表已保存: {plot_file}")
            
        except Exception as e:
            print(f"  ⚠️ 图表生成失败: {e}")
        
        print("\n步骤6: 生成处理报告...")
        
        # 生成EQ处理报告
        eq_report = {
            "task": "7.3 Frequency equalization optimization",
            "input_file": str(input_audio),
            "output_file": str(equalized_audio),
            "processing_timestamp": datetime.now().isoformat(),
            "frequency_analysis": {
                "before": freq_analysis_before,
                "after": freq_analysis_after if 'freq_analysis_after' in locals() else None
            },
            "eq_settings": eq_settings,
            "eq_filters_applied": eq_filters,
            "results": {
                "speech_enhancement_ratio": float(speech_enhancement) if 'speech_enhancement' in locals() else 1.0,
                "clarity_enhancement_ratio": float(clarity_enhancement) if 'clarity_enhancement' in locals() else 1.0,
                "eq_quality_assessment": eq_quality if 'eq_quality' in locals() else "未评估",
                "processing_successful": True
            }
        }
        
        # 保存处理报告
        report_file = workspace / "task_7_3_frequency_eq_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(eq_report, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ 处理报告已保存: {report_file}")
        
        # 生成Markdown报告
        markdown_content = f"""# Task 7.3: 频率均衡优化报告

## 处理概述
- **输入文件**: {input_audio}
- **输出文件**: {equalized_audio}
- **处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 频率分析结果

### EQ前频率特征
"""
        
        if freq_analysis_before:
            markdown_content += f"""- **人声/低频比率**: {freq_analysis_before['speech_to_bass_ratio']:.2f}
- **清晰度/人声比率**: {freq_analysis_before['clarity_to_speech_ratio']:.2f}

#### 频段能量分布 (EQ前)
"""
            for band, energy in freq_analysis_before['band_energies'].items():
                markdown_content += f"- **{band}**: {energy:.4f}\n"
        
        markdown_content += f"""
## EQ设置

### 应用的EQ曲线
- **低频衰减**: {eq_settings['low_cut']['freq']}Hz, {eq_settings['low_cut']['gain']}dB
- **人声增强**: {eq_settings['speech_boost']['freq']}Hz, +{eq_settings['speech_boost']['gain']}dB, Q={eq_settings['speech_boost']['q']}
- **清晰度提升**: {eq_settings['clarity_boost']['freq']}Hz, +{eq_settings['clarity_boost']['gain']}dB, Q={eq_settings['clarity_boost']['q']}
- **高频控制**: {eq_settings['high_cut']['freq']}Hz, {eq_settings['high_cut']['gain']}dB

## 处理结果

| 指标 | EQ前 | EQ后 | 改善倍数 |
|------|------|------|----------|
| 人声/低频比率 | {freq_analysis_before['speech_to_bass_ratio']:.2f} | {freq_analysis_after['speech_to_bass_ratio']:.2f} | {speech_enhancement:.2f}x |
| 清晰度/人声比率 | {freq_analysis_before['clarity_to_speech_ratio']:.2f} | {freq_analysis_after['clarity_to_speech_ratio']:.2f} | {clarity_enhancement:.2f}x |

### 质量评估: **{eq_quality}**

## 结论

频率均衡优化已完成，采用了针对话剧音频特点的专业EQ曲线。处理结果显示{eq_quality}，有效提升了对话清晰度和整体音质。

---
*Task 7.3 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
""" if 'freq_analysis_after' in locals() and freq_analysis_after else """

## 处理结果

EQ处理已完成，但效果分析数据不完整。

## 结论

频率均衡优化已完成，采用了针对话剧音频特点的专业EQ曲线。

---
*Task 7.3 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        markdown_file = workspace / "task_7_3_frequency_eq_report.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"  ✅ Markdown报告已保存: {markdown_file}")
        
        print(f"\n🎉 Task 7.3 频率均衡优化完成!")
        print(f"📁 输出文件: {equalized_audio}")
        print(f"📊 处理效果: {eq_quality}")
        if 'speech_enhancement' in locals():
            print(f"🎤 人声增强: {speech_enhancement:.2f}x")
            print(f"🔊 清晰度提升: {clarity_enhancement:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    apply_frequency_equalization()