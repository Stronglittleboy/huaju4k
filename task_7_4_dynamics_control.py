#!/usr/bin/env python3
"""
Task 7.4: Dynamic Range Control and Compression
动态范围控制 - 针对话剧音频的智能压缩处理
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

def analyze_dynamic_range(audio_file):
    """分析音频的动态范围特征"""
    try:
        import librosa
        
        print("  - 加载音频进行动态范围分析...")
        y, sr = librosa.load(audio_file, sr=48000, duration=60)
        
        # 基本动态特征
        rms_energy = np.sqrt(np.mean(y**2))
        peak_amplitude = np.max(np.abs(y))
        crest_factor = 20 * np.log10(peak_amplitude / (rms_energy + 1e-10))
        
        print(f"    - RMS能量: {rms_energy:.6f}")
        print(f"    - 峰值振幅: {peak_amplitude:.6f}")
        print(f"    - 峰值因子: {crest_factor:.1f} dB")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 动态范围分析失败: {e}")
        return False

def apply_dynamics_control():
    """执行动态范围控制和压缩"""
    
    print("🎚️ Task 7.4: 动态范围控制和压缩")
    
    workspace = Path("audio_workspace")
    
    # 检查输入文件
    input_audio = workspace / "equalized_audio.wav"
    if not input_audio.exists():
        print("❌ 未找到EQ处理后的音频文件，请先完成Task 7.3")
        return False
    
    try:
        print("\n步骤1: 分析音频动态范围特征...")
        
        # 分析动态范围
        if not analyze_dynamic_range(str(input_audio)):
            return False
        
        print("\n步骤2: 设计压缩参数...")
        
        # 读取之前的分析结果
        analysis_file = workspace / "task_7_1_basic_audio_analysis.json"
        if analysis_file.exists():
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            dynamic_range_db = analysis_data['basic_metrics']['dynamic_range_db']
            compression_needed = analysis_data['processing_recommendations']['compression_needed']
            
            print(f"  - 当前动态范围: {dynamic_range_db:.1f} dB")
            print(f"  - 需要压缩: {compression_needed}")
        else:
            dynamic_range_db = 15.0
            compression_needed = False
        
        # 设计压缩参数
        if compression_needed or dynamic_range_db > 18:
            # 需要较强压缩
            comp_threshold = -18
            comp_ratio = 4.0
            comp_attack = 3
            comp_release = 100
            print("  - 应用中等强度压缩")
        else:
            # 轻度压缩
            comp_threshold = -20
            comp_ratio = 2.5
            comp_attack = 5
            comp_release = 150
            print("  - 应用轻度压缩")
        
        print(f"    压缩参数: 门限={comp_threshold}dB, 比率={comp_ratio}:1")
        
        print("\n步骤3: 应用动态压缩处理...")
        
        # 第一阶段：多段压缩器
        stage1_audio = workspace / "stage1_compressed.wav"
        comp_cmd = f'ffmpeg -i "{input_audio}" -af "acompressor=threshold={comp_threshold}dB:ratio={comp_ratio}:attack={comp_attack}:release={comp_release}:makeup=2" "{stage1_audio}" -y'
        
        if not run_ffmpeg_command(comp_cmd, f"应用压缩器 (门限: {comp_threshold}dB, 比率: {comp_ratio}:1)"):
            return False
        
        print("  ✅ 第一阶段压缩完成")
        
        # 第二阶段：限幅器
        final_compressed = workspace / "compressed_audio.wav"
        limiter_cmd = f'ffmpeg -i "{stage1_audio}" -af "alimiter=level_in=1:level_out=0.95:limit=-1:release=50" "{final_compressed}" -y'
        
        if not run_ffmpeg_command(limiter_cmd, "应用限幅器 (防止过载)"):
            return False
        
        print("  ✅ 第二阶段限幅完成")
        
        print("\n步骤4: 验证压缩效果...")
        
        # 分析压缩前后效果
        try:
            import librosa
            
            # 分析原始音频
            y_orig, sr = librosa.load(str(input_audio), sr=48000, duration=30)
            rms_orig = np.sqrt(np.mean(y_orig**2))
            peak_orig = np.max(np.abs(y_orig))
            crest_orig = 20 * np.log10(peak_orig / (rms_orig + 1e-10))
            
            # 分析压缩后音频
            y_comp, _ = librosa.load(str(final_compressed), sr=48000, duration=30)
            rms_comp = np.sqrt(np.mean(y_comp**2))
            peak_comp = np.max(np.abs(y_comp))
            crest_comp = 20 * np.log10(peak_comp / (rms_comp + 1e-10))
            
            # 计算改善效果
            dynamic_reduction = crest_orig - crest_comp
            loudness_increase = 20 * np.log10(rms_comp / (rms_orig + 1e-10))
            
            print(f"  📊 压缩效果分析:")
            print(f"    - 原始峰值因子: {crest_orig:.1f} dB")
            print(f"    - 压缩后峰值因子: {crest_comp:.1f} dB")
            print(f"    - 动态范围减少: {dynamic_reduction:.1f} dB")
            print(f"    - 响度提升: {loudness_increase:.1f} dB")
            
            if dynamic_reduction > 2:
                quality_assessment = "显著改善"
            elif dynamic_reduction > 1:
                quality_assessment = "适度改善"
            else:
                quality_assessment = "基本保持"
            
            print(f"    - 质量评估: {quality_assessment}")
            
        except Exception as e:
            print(f"  ⚠️ 压缩效果分析失败: {e}")
            dynamic_reduction = 0
            loudness_increase = 0
            quality_assessment = "未评估"
        
        print("\n步骤5: 生成处理报告...")
        
        # 生成报告
        dynamics_report = {
            "task": "7.4 Dynamic range control and compression",
            "input_file": str(input_audio),
            "output_file": str(final_compressed),
            "processing_timestamp": datetime.now().isoformat(),
            "compression_settings": {
                "threshold_db": comp_threshold,
                "ratio": comp_ratio,
                "attack_ms": comp_attack,
                "release_ms": comp_release
            },
            "results": {
                "dynamic_range_reduction_db": float(dynamic_reduction) if 'dynamic_reduction' in locals() else 0,
                "loudness_increase_db": float(loudness_increase) if 'loudness_increase' in locals() else 0,
                "quality_assessment": quality_assessment if 'quality_assessment' in locals() else "未评估",
                "processing_successful": True
            }
        }
        
        # 保存报告
        report_file = workspace / "task_7_4_dynamics_control_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(dynamics_report, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ 处理报告已保存: {report_file}")
        
        print(f"\n🎉 Task 7.4 动态范围控制完成!")
        print(f"📁 输出文件: {final_compressed}")
        print(f"📊 处理效果: {quality_assessment}")
        if 'dynamic_reduction' in locals():
            print(f"📉 动态范围减少: {dynamic_reduction:.1f} dB")
            print(f"📈 响度提升: {loudness_increase:.1f} dB")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    apply_dynamics_control()