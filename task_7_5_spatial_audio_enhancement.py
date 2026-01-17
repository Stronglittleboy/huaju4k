#!/usr/bin/env python3
"""
任务7.5: 空间音频增强和优化
Spatial audio enhancement and optimization for theater recordings
"""

import os
import json
import numpy as np
import librosa
import librosa.effects
import soundfile as sf
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SpatialAudioEnhancer:
    def __init__(self, workspace_dir="audio_workspace"):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(exist_ok=True)
        
        # 空间音频参数
        self.spatial_params = {
            "reverb_room_size": 0.7,      # 剧场空间大小
            "reverb_damping": 0.3,        # 阻尼系数
            "stereo_width": 1.2,          # 立体声宽度
            "stage_depth": 0.8,           # 舞台深度感
            "ambient_level": 0.15,        # 环境声级别
            "center_focus": 0.6           # 中心聚焦度
        }
        
    def analyze_spatial_characteristics(self, audio_path):
        """分析原始录音的空间特性"""
        print(f"🎭 分析空间特性: {audio_path}")
        
        # 加载立体声音频
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        
        if y.ndim == 1:
            print("⚠️ 音频是单声道，将创建伪立体声")
            y = np.array([y, y])
        
        left_channel = y[0]
        right_channel = y[1]
        
        # 分析立体声特性
        spatial_analysis = {}
        
        # 1. 立体声相关性分析
        correlation = np.corrcoef(left_channel, right_channel)[0, 1]
        spatial_analysis["stereo_correlation"] = correlation
        
        # 2. 左右声道能量差异
        left_energy = np.mean(left_channel**2)
        right_energy = np.mean(right_channel**2)
        energy_balance = (right_energy - left_energy) / (right_energy + left_energy + 1e-10)
        spatial_analysis["energy_balance"] = energy_balance
        
        # 3. 相位差分析
        # 使用短时傅里叶变换分析相位关系
        stft_left = librosa.stft(left_channel)
        stft_right = librosa.stft(right_channel)
        
        phase_left = np.angle(stft_left)
        phase_right = np.angle(stft_right)
        phase_diff = np.mean(np.abs(phase_left - phase_right))
        spatial_analysis["average_phase_difference"] = phase_diff
        
        # 4. 频谱宽度分析
        spectral_centroid_left = librosa.feature.spectral_centroid(y=left_channel, sr=sr)[0]
        spectral_centroid_right = librosa.feature.spectral_centroid(y=right_channel, sr=sr)[0]
        
        spatial_analysis["spectral_centroid_left"] = np.mean(spectral_centroid_left)
        spatial_analysis["spectral_centroid_right"] = np.mean(spectral_centroid_right)
        
        # 5. 动态范围分析
        left_rms = librosa.feature.rms(y=left_channel)[0]
        right_rms = librosa.feature.rms(y=right_channel)[0]
        
        spatial_analysis["left_dynamic_range"] = 20 * np.log10(np.max(left_rms) / (np.mean(left_rms) + 1e-10))
        spatial_analysis["right_dynamic_range"] = 20 * np.log10(np.max(right_rms) / (np.mean(right_rms) + 1e-10))
        
        print(f"  立体声相关性: {correlation:.3f}")
        print(f"  能量平衡: {energy_balance:.3f}")
        print(f"  平均相位差: {phase_diff:.3f}")
        
        return spatial_analysis, y, sr
    
    def enhance_stereo_width(self, audio_data, sr, width_factor=1.2):
        """增强立体声宽度"""
        print(f"🔊 增强立体声宽度 (因子: {width_factor})")
        
        if audio_data.ndim == 1:
            # 单声道转立体声
            enhanced = np.array([audio_data, audio_data])
        else:
            left = audio_data[0]
            right = audio_data[1]
            
            # 计算中间(Mid)和侧面(Side)信号
            mid = (left + right) / 2
            side = (left - right) / 2
            
            # 增强侧面信号以扩展立体声宽度
            enhanced_side = side * width_factor
            
            # 重构左右声道
            enhanced_left = mid + enhanced_side
            enhanced_right = mid - enhanced_side
            
            enhanced = np.array([enhanced_left, enhanced_right])
        
        return enhanced
    
    def add_stage_reverb(self, audio_data, sr):
        """添加舞台混响效果"""
        print("🎪 添加舞台混响效果")
        
        # 创建简单的混响效果
        # 使用多个延迟和衰减来模拟剧场空间
        
        reverb_delays = [0.02, 0.04, 0.08, 0.15, 0.25]  # 延迟时间(秒)
        reverb_gains = [0.3, 0.25, 0.2, 0.15, 0.1]      # 对应增益
        
        enhanced = audio_data.copy()
        
        for delay, gain in zip(reverb_delays, reverb_gains):
            delay_samples = int(delay * sr)
            
            if audio_data.ndim == 1:
                # 单声道处理
                delayed = np.zeros_like(audio_data)
                if delay_samples < len(audio_data):
                    delayed[delay_samples:] = audio_data[:-delay_samples] * gain
                enhanced += delayed
            else:
                # 立体声处理
                for channel in range(audio_data.shape[0]):
                    delayed = np.zeros_like(audio_data[channel])
                    if delay_samples < len(audio_data[channel]):
                        delayed[delay_samples:] = audio_data[channel][:-delay_samples] * gain
                    enhanced[channel] += delayed
        
        # 归一化防止削波
        max_val = np.max(np.abs(enhanced))
        if max_val > 0.95:
            enhanced = enhanced * 0.95 / max_val
        
        return enhanced
    
    def enhance_stage_presence(self, audio_data, sr):
        """增强舞台临场感"""
        print("🎭 增强舞台临场感")
        
        # 1. 增强中频范围 (人声主要频段)
        # 使用简单的频域滤波
        if audio_data.ndim == 1:
            channels = [audio_data]
        else:
            channels = [audio_data[i] for i in range(audio_data.shape[0])]
        
        enhanced_channels = []
        
        for channel in channels:
            # FFT处理
            fft = np.fft.rfft(channel)
            freqs = np.fft.rfftfreq(len(channel), 1/sr)
            
            # 创建增强滤波器
            # 增强300Hz-3kHz范围 (人声清晰度)
            enhancement = np.ones_like(freqs)
            
            # 人声增强
            voice_mask = (freqs >= 300) & (freqs <= 3000)
            enhancement[voice_mask] *= 1.15
            
            # 轻微衰减低频噪声
            low_freq_mask = freqs < 100
            enhancement[low_freq_mask] *= 0.9
            
            # 轻微衰减高频噪声
            high_freq_mask = freqs > 8000
            enhancement[high_freq_mask] *= 0.95
            
            # 应用滤波器
            enhanced_fft = fft * enhancement
            enhanced_channel = np.fft.irfft(enhanced_fft, len(channel))
            enhanced_channels.append(enhanced_channel)
        
        if audio_data.ndim == 1:
            return enhanced_channels[0]
        else:
            return np.array(enhanced_channels)
    
    def optimize_spatial_depth(self, audio_data, sr):
        """优化空间深度感"""
        print("🌊 优化空间深度感")
        
        if audio_data.ndim == 1:
            # 单声道：创建简单的深度效果
            # 使用轻微的延迟和滤波
            delayed = np.zeros_like(audio_data)
            delay_samples = int(0.01 * sr)  # 10ms延迟
            
            if delay_samples < len(audio_data):
                delayed[delay_samples:] = audio_data[:-delay_samples] * 0.2
            
            enhanced = audio_data + delayed
        else:
            # 立体声：创建交叉延迟效果
            left = audio_data[0]
            right = audio_data[1]
            
            # 左声道添加来自右声道的轻微延迟
            delay_samples = int(0.005 * sr)  # 5ms延迟
            
            left_enhanced = left.copy()
            right_enhanced = right.copy()
            
            if delay_samples < len(left):
                # 右到左的延迟反馈
                left_enhanced[delay_samples:] += right[:-delay_samples] * 0.1
                # 左到右的延迟反馈
                right_enhanced[delay_samples:] += left[:-delay_samples] * 0.1
            
            enhanced = np.array([left_enhanced, right_enhanced])
        
        return enhanced
    
    def process_spatial_enhancement(self, input_path, output_path):
        """执行完整的空间音频增强"""
        print(f"\n🚀 开始空间音频增强处理")
        print(f"输入: {input_path}")
        print(f"输出: {output_path}")
        
        # 1. 分析原始空间特性
        spatial_analysis, audio_data, sr = self.analyze_spatial_characteristics(input_path)
        
        # 2. 增强立体声宽度
        enhanced_audio = self.enhance_stereo_width(
            audio_data, sr, 
            width_factor=self.spatial_params["stereo_width"]
        )
        
        # 3. 添加舞台混响
        enhanced_audio = self.add_stage_reverb(enhanced_audio, sr)
        
        # 4. 增强舞台临场感
        enhanced_audio = self.enhance_stage_presence(enhanced_audio, sr)
        
        # 5. 优化空间深度
        enhanced_audio = self.optimize_spatial_depth(enhanced_audio, sr)
        
        # 6. 最终归一化
        max_val = np.max(np.abs(enhanced_audio))
        if max_val > 0.95:
            enhanced_audio = enhanced_audio * 0.95 / max_val
        
        # 7. 保存增强后的音频
        if enhanced_audio.ndim == 1:
            sf.write(output_path, enhanced_audio, sr)
        else:
            sf.write(output_path, enhanced_audio.T, sr)  # soundfile需要转置
        
        print(f"✅ 空间音频增强完成")
        
        # 8. 分析增强后的空间特性
        enhanced_analysis, _, _ = self.analyze_spatial_characteristics(output_path)
        
        return spatial_analysis, enhanced_analysis
    
    def create_spatial_comparison_plot(self, original_analysis, enhanced_analysis):
        """创建空间特性对比图"""
        print("📊 生成空间特性对比图")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('空间音频增强效果对比', fontsize=14)
        
        # 1. 立体声相关性对比
        correlations = [original_analysis["stereo_correlation"], enhanced_analysis["stereo_correlation"]]
        axes[0,0].bar(['原始', '增强'], correlations, color=['blue', 'orange'])
        axes[0,0].set_title('立体声相关性')
        axes[0,0].set_ylabel('相关系数')
        axes[0,0].set_ylim([0, 1])
        
        # 2. 能量平衡对比
        balances = [original_analysis["energy_balance"], enhanced_analysis["energy_balance"]]
        axes[0,1].bar(['原始', '增强'], balances, color=['blue', 'orange'])
        axes[0,1].set_title('左右声道能量平衡')
        axes[0,1].set_ylabel('平衡系数')
        
        # 3. 频谱重心对比
        left_centroids = [original_analysis["spectral_centroid_left"], enhanced_analysis["spectral_centroid_left"]]
        right_centroids = [original_analysis["spectral_centroid_right"], enhanced_analysis["spectral_centroid_right"]]
        
        x = np.arange(2)
        width = 0.35
        axes[1,0].bar(x - width/2, left_centroids, width, label='左声道', color='lightblue')
        axes[1,0].bar(x + width/2, right_centroids, width, label='右声道', color='lightcoral')
        axes[1,0].set_title('频谱重心对比')
        axes[1,0].set_ylabel('频率 (Hz)')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(['原始', '增强'])
        axes[1,0].legend()
        
        # 4. 动态范围对比
        left_dr = [original_analysis["left_dynamic_range"], enhanced_analysis["left_dynamic_range"]]
        right_dr = [original_analysis["right_dynamic_range"], enhanced_analysis["right_dynamic_range"]]
        
        axes[1,1].bar(x - width/2, left_dr, width, label='左声道', color='lightblue')
        axes[1,1].bar(x + width/2, right_dr, width, label='右声道', color='lightcoral')
        axes[1,1].set_title('动态范围对比')
        axes[1,1].set_ylabel('动态范围 (dB)')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(['原始', '增强'])
        axes[1,1].legend()
        
        plt.tight_layout()
        
        plot_path = self.workspace / "spatial_enhancement_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 对比图已保存: {plot_path}")
        return plot_path

def main():
    print("🎭 任务7.5: 空间音频增强和优化")
    print("=" * 50)
    
    # 初始化空间音频增强器
    enhancer = SpatialAudioEnhancer()
    
    # 输入输出文件路径
    input_audio = Path("audio_workspace/acoustics_optimized_audio.wav")
    output_audio = Path("audio_workspace/spatial_enhanced_audio.wav")
    
    if not input_audio.exists():
        print(f"❌ 输入音频文件不存在: {input_audio}")
        return False
    
    try:
        # 执行空间音频增强
        original_analysis, enhanced_analysis = enhancer.process_spatial_enhancement(
            input_audio, output_audio
        )
        
        # 生成对比图
        plot_path = enhancer.create_spatial_comparison_plot(original_analysis, enhanced_analysis)
        
        # 生成报告
        report = {
            "task": "7.5 Spatial audio enhancement and optimization",
            "timestamp": datetime.now().isoformat(),
            "input_file": str(input_audio),
            "output_file": str(output_audio),
            "spatial_parameters": enhancer.spatial_params,
            "original_analysis": original_analysis,
            "enhanced_analysis": enhanced_analysis,
            "improvements": {
                "stereo_correlation_change": enhanced_analysis["stereo_correlation"] - original_analysis["stereo_correlation"],
                "energy_balance_change": enhanced_analysis["energy_balance"] - original_analysis["energy_balance"],
                "phase_difference_change": enhanced_analysis["average_phase_difference"] - original_analysis["average_phase_difference"]
            },
            "visualization": str(plot_path)
        }
        
        # 保存报告
        report_path = Path("audio_workspace/task_7_5_spatial_enhancement_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 打印结果
        print(f"\n📊 空间音频增强结果:")
        print(f"   立体声相关性变化: {report['improvements']['stereo_correlation_change']:.3f}")
        print(f"   能量平衡变化: {report['improvements']['energy_balance_change']:.3f}")
        print(f"   相位差变化: {report['improvements']['phase_difference_change']:.3f}")
        print(f"   输出文件: {output_audio}")
        print(f"   对比图: {plot_path}")
        print(f"   报告文件: {report_path}")
        
        print(f"\n✅ 任务7.5完成: 空间音频增强和优化")
        return True
        
    except Exception as e:
        print(f"❌ 空间音频增强失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 空间音频增强任务成功完成!")
    else:
        print("\n❌ 空间音频增强任务失败")