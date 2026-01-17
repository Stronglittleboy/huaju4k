#!/usr/bin/env python3
"""
任务8.1: 音频增强效果分析
解决OpenCV CUDA问题并完成音频质量验证
"""

import os
import sys
import json
import subprocess
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import librosa
import librosa.display

class AudioValidationTask:
    def __init__(self):
        self.setup_logging()
        self.setup_paths()
        
    def setup_logging(self):
        """设置日志系统"""
        log_file = f"task_8_1_audio_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_paths(self):
        """设置文件路径"""
        self.audio_workspace = Path("audio_workspace")
        self.original_audio = self.audio_workspace / "original_audio.wav"
        self.enhanced_audio = self.audio_workspace / "acoustics_optimized_audio.wav"
        
        # 检查文件是否存在
        if not self.original_audio.exists():
            self.logger.warning(f"原始音频文件不存在: {self.original_audio}")
        if not self.enhanced_audio.exists():
            self.logger.warning(f"增强音频文件不存在: {self.enhanced_audio}")
            
    def analyze_audio_file(self, audio_path, label="Audio"):
        """分析单个音频文件"""
        self.logger.info(f"🎵 分析音频文件: {audio_path}")
        
        try:
            # 加载音频
            y, sr = librosa.load(str(audio_path), sr=None)
            duration = len(y) / sr
            
            # 基本统计
            rms_energy = np.sqrt(np.mean(y**2))
            peak_amplitude = np.max(np.abs(y))
            dynamic_range = 20 * np.log10(peak_amplitude / (rms_energy + 1e-10))
            
            # 频谱分析
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # 噪音估计 (使用最低10%的能量作为噪音基准)
            sorted_magnitude = np.sort(magnitude.flatten())
            noise_floor = np.mean(sorted_magnitude[:int(len(sorted_magnitude) * 0.1)])
            signal_power = np.mean(magnitude)
            snr_estimate = 20 * np.log10(signal_power / (noise_floor + 1e-10))
            
            # MFCC特征
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            analysis_result = {
                "file_path": str(audio_path),
                "label": label,
                "basic_stats": {
                    "duration_seconds": float(duration),
                    "sample_rate": int(sr),
                    "total_samples": len(y),
                    "rms_energy": float(rms_energy),
                    "peak_amplitude": float(peak_amplitude),
                    "dynamic_range_db": float(dynamic_range)
                },
                "spectral_features": {
                    "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                    "spectral_centroid_std": float(np.std(spectral_centroid)),
                    "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                    "spectral_rolloff_std": float(np.std(spectral_rolloff))
                },
                "quality_metrics": {
                    "estimated_snr_db": float(snr_estimate),
                    "noise_floor_estimate": float(noise_floor),
                    "signal_power": float(signal_power)
                },
                "mfcc_stats": {
                    "mfcc_mean": mfccs.mean(axis=1).tolist(),
                    "mfcc_std": mfccs.std(axis=1).tolist()
                }
            }
            
            self.logger.info(f"✅ {label} 分析完成:")
            self.logger.info(f"   时长: {duration:.1f}秒")
            self.logger.info(f"   动态范围: {dynamic_range:.1f}dB")
            self.logger.info(f"   估计SNR: {snr_estimate:.1f}dB")
            self.logger.info(f"   频谱重心: {np.mean(spectral_centroid):.1f}Hz")
            
            return analysis_result, y, sr
            
        except Exception as e:
            self.logger.error(f"音频分析失败 {audio_path}: {e}")
            return None, None, None
            
    def generate_frequency_spectrum_comparison(self, original_data, enhanced_data, original_sr, enhanced_sr):
        """生成频谱对比图"""
        self.logger.info("📊 生成频谱对比图...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('音频增强效果对比分析', fontsize=16, fontweight='bold')
            
            # 原始音频频谱图
            D_original = librosa.amplitude_to_db(np.abs(librosa.stft(original_data)), ref=np.max)
            librosa.display.specshow(D_original, sr=original_sr, x_axis='time', y_axis='hz', ax=axes[0,0])
            axes[0,0].set_title('原始音频频谱图')
            axes[0,0].set_ylabel('频率 (Hz)')
            
            # 增强音频频谱图
            D_enhanced = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_data)), ref=np.max)
            librosa.display.specshow(D_enhanced, sr=enhanced_sr, x_axis='time', y_axis='hz', ax=axes[0,1])
            axes[0,1].set_title('增强音频频谱图')
            
            # 频率响应对比
            freqs_original = np.fft.rfftfreq(len(original_data), 1/original_sr)
            fft_original = np.abs(np.fft.rfft(original_data))
            freqs_enhanced = np.fft.rfftfreq(len(enhanced_data), 1/enhanced_sr)
            fft_enhanced = np.abs(np.fft.rfft(enhanced_data))
            
            axes[1,0].semilogx(freqs_original, 20*np.log10(fft_original + 1e-10), label='原始音频', alpha=0.7)
            axes[1,0].semilogx(freqs_enhanced, 20*np.log10(fft_enhanced + 1e-10), label='增强音频', alpha=0.7)
            axes[1,0].set_xlabel('频率 (Hz)')
            axes[1,0].set_ylabel('幅度 (dB)')
            axes[1,0].set_title('频率响应对比')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # 时域波形对比
            time_original = np.linspace(0, len(original_data)/original_sr, len(original_data))
            time_enhanced = np.linspace(0, len(enhanced_data)/enhanced_sr, len(enhanced_data))
            
            # 只显示前10秒的波形
            max_samples_original = min(len(original_data), int(10 * original_sr))
            max_samples_enhanced = min(len(enhanced_data), int(10 * enhanced_sr))
            
            axes[1,1].plot(time_original[:max_samples_original], original_data[:max_samples_original], 
                          label='原始音频', alpha=0.7, linewidth=0.5)
            axes[1,1].plot(time_enhanced[:max_samples_enhanced], enhanced_data[:max_samples_enhanced], 
                          label='增强音频', alpha=0.7, linewidth=0.5)
            axes[1,1].set_xlabel('时间 (秒)')
            axes[1,1].set_ylabel('幅度')
            axes[1,1].set_title('时域波形对比 (前10秒)')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图像
            comparison_plot_path = "task_8_1_frequency_spectrum_comparison.png"
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 频谱对比图已保存: {comparison_plot_path}")
            return comparison_plot_path
            
        except Exception as e:
            self.logger.error(f"生成频谱对比图失败: {e}")
            return None
            
    def calculate_quality_improvements(self, original_analysis, enhanced_analysis):
        """计算质量改善指标"""
        self.logger.info("📈 计算质量改善指标...")
        
        try:
            improvements = {}
            
            # SNR改善
            original_snr = original_analysis["quality_metrics"]["estimated_snr_db"]
            enhanced_snr = enhanced_analysis["quality_metrics"]["estimated_snr_db"]
            snr_improvement = enhanced_snr - original_snr
            
            # 动态范围改善
            original_dr = original_analysis["basic_stats"]["dynamic_range_db"]
            enhanced_dr = enhanced_analysis["basic_stats"]["dynamic_range_db"]
            dr_improvement = enhanced_dr - original_dr
            
            # 频谱重心变化 (通常降低表示噪音减少)
            original_centroid = original_analysis["spectral_features"]["spectral_centroid_mean"]
            enhanced_centroid = enhanced_analysis["spectral_features"]["spectral_centroid_mean"]
            centroid_change = enhanced_centroid - original_centroid
            
            # RMS能量变化
            original_rms = original_analysis["basic_stats"]["rms_energy"]
            enhanced_rms = enhanced_analysis["basic_stats"]["rms_energy"]
            rms_change_db = 20 * np.log10(enhanced_rms / (original_rms + 1e-10))
            
            improvements = {
                "snr_improvement_db": float(snr_improvement),
                "dynamic_range_improvement_db": float(dr_improvement),
                "spectral_centroid_change_hz": float(centroid_change),
                "rms_energy_change_db": float(rms_change_db),
                "quality_assessment": {
                    "snr_status": "改善" if snr_improvement > 1 else "轻微改善" if snr_improvement > 0 else "无明显改善",
                    "dynamic_range_status": "改善" if dr_improvement > 1 else "轻微改善" if dr_improvement > 0 else "无明显改善",
                    "noise_reduction_status": "有效" if centroid_change < -100 else "轻微" if centroid_change < 0 else "无明显效果"
                }
            }
            
            self.logger.info("📊 质量改善分析:")
            self.logger.info(f"   SNR改善: {snr_improvement:.1f}dB ({improvements['quality_assessment']['snr_status']})")
            self.logger.info(f"   动态范围改善: {dr_improvement:.1f}dB ({improvements['quality_assessment']['dynamic_range_status']})")
            self.logger.info(f"   频谱重心变化: {centroid_change:.1f}Hz ({improvements['quality_assessment']['noise_reduction_status']})")
            self.logger.info(f"   RMS能量变化: {rms_change_db:.1f}dB")
            
            return improvements
            
        except Exception as e:
            self.logger.error(f"计算质量改善指标失败: {e}")
            return {}
            
    def generate_comprehensive_report(self, original_analysis, enhanced_analysis, improvements, comparison_plot):
        """生成综合报告"""
        self.logger.info("📋 生成综合音频增强效果报告...")
        
        report = {
            "task": "8.1 Audio enhancement effectiveness analysis",
            "analysis_timestamp": datetime.now().isoformat(),
            "original_audio_analysis": original_analysis,
            "enhanced_audio_analysis": enhanced_analysis,
            "quality_improvements": improvements,
            "visualization": {
                "frequency_spectrum_comparison": comparison_plot
            },
            "summary": {
                "processing_successful": True,
                "overall_quality_improvement": self.assess_overall_improvement(improvements),
                "recommendations": self.generate_recommendations(improvements)
            }
        }
        
        # 保存JSON报告
        json_report_path = "task_8_1_audio_enhancement_effectiveness_report.json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # 生成Markdown报告
        md_report_path = "task_8_1_audio_enhancement_effectiveness_report.md"
        self.generate_markdown_report(report, md_report_path)
        
        self.logger.info(f"✅ 综合报告已生成:")
        self.logger.info(f"   JSON报告: {json_report_path}")
        self.logger.info(f"   Markdown报告: {md_report_path}")
        
        return report
        
    def assess_overall_improvement(self, improvements):
        """评估整体改善效果"""
        snr_imp = improvements.get("snr_improvement_db", 0)
        dr_imp = improvements.get("dynamic_range_improvement_db", 0)
        
        if snr_imp > 3 and dr_imp > 2:
            return "显著改善"
        elif snr_imp > 1 or dr_imp > 1:
            return "中等改善"
        elif snr_imp > 0 or dr_imp > 0:
            return "轻微改善"
        else:
            return "无明显改善"
            
    def generate_recommendations(self, improvements):
        """生成改进建议"""
        recommendations = []
        
        snr_imp = improvements.get("snr_improvement_db", 0)
        dr_imp = improvements.get("dynamic_range_improvement_db", 0)
        
        if snr_imp < 2:
            recommendations.append("建议调整降噪参数以获得更好的信噪比改善")
        if dr_imp < 1:
            recommendations.append("建议优化动态范围控制参数")
        if snr_imp > 5:
            recommendations.append("当前降噪效果良好，可保持现有参数")
            
        return recommendations
        
    def generate_markdown_report(self, report, output_path):
        """生成Markdown格式报告"""
        md_content = f"""# 任务8.1 音频增强效果分析报告

## 分析概述
- **分析时间**: {report['analysis_timestamp']}
- **任务**: {report['task']}
- **整体改善效果**: {report['summary']['overall_quality_improvement']}

## 原始音频分析
- **文件路径**: {report['original_audio_analysis']['file_path']}
- **时长**: {report['original_audio_analysis']['basic_stats']['duration_seconds']:.1f}秒
- **采样率**: {report['original_audio_analysis']['basic_stats']['sample_rate']}Hz
- **动态范围**: {report['original_audio_analysis']['basic_stats']['dynamic_range_db']:.1f}dB
- **估计SNR**: {report['original_audio_analysis']['quality_metrics']['estimated_snr_db']:.1f}dB
- **频谱重心**: {report['original_audio_analysis']['spectral_features']['spectral_centroid_mean']:.1f}Hz

## 增强音频分析
- **文件路径**: {report['enhanced_audio_analysis']['file_path']}
- **时长**: {report['enhanced_audio_analysis']['basic_stats']['duration_seconds']:.1f}秒
- **采样率**: {report['enhanced_audio_analysis']['basic_stats']['sample_rate']}Hz
- **动态范围**: {report['enhanced_audio_analysis']['basic_stats']['dynamic_range_db']:.1f}dB
- **估计SNR**: {report['enhanced_audio_analysis']['quality_metrics']['estimated_snr_db']:.1f}dB
- **频谱重心**: {report['enhanced_audio_analysis']['spectral_features']['spectral_centroid_mean']:.1f}Hz

## 质量改善指标
- **SNR改善**: {report['quality_improvements']['snr_improvement_db']:.1f}dB ({report['quality_improvements']['quality_assessment']['snr_status']})
- **动态范围改善**: {report['quality_improvements']['dynamic_range_improvement_db']:.1f}dB ({report['quality_improvements']['quality_assessment']['dynamic_range_status']})
- **频谱重心变化**: {report['quality_improvements']['spectral_centroid_change_hz']:.1f}Hz ({report['quality_improvements']['quality_assessment']['noise_reduction_status']})
- **RMS能量变化**: {report['quality_improvements']['rms_energy_change_db']:.1f}dB

## 改进建议
"""
        for rec in report['summary']['recommendations']:
            md_content += f"- {rec}\n"
            
        md_content += f"""
## 可视化分析
- **频谱对比图**: {report['visualization']['frequency_spectrum_comparison']}

## 结论
音频增强处理整体效果为: **{report['summary']['overall_quality_improvement']}**

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
    def run_analysis(self):
        """执行完整的音频增强效果分析"""
        self.logger.info("🚀 开始任务8.1: 音频增强效果分析")
        
        # 分析原始音频
        original_analysis, original_data, original_sr = self.analyze_audio_file(
            self.original_audio, "原始音频"
        )
        
        if original_analysis is None:
            self.logger.error("❌ 原始音频分析失败")
            return False
            
        # 分析增强音频
        enhanced_analysis, enhanced_data, enhanced_sr = self.analyze_audio_file(
            self.enhanced_audio, "增强音频"
        )
        
        if enhanced_analysis is None:
            self.logger.error("❌ 增强音频分析失败")
            return False
            
        # 生成频谱对比图
        comparison_plot = self.generate_frequency_spectrum_comparison(
            original_data, enhanced_data, original_sr, enhanced_sr
        )
        
        # 计算质量改善指标
        improvements = self.calculate_quality_improvements(
            original_analysis, enhanced_analysis
        )
        
        # 生成综合报告
        report = self.generate_comprehensive_report(
            original_analysis, enhanced_analysis, improvements, comparison_plot
        )
        
        self.logger.info("✅ 任务8.1完成: 音频增强效果分析")
        return True

def main():
    """主函数"""
    validator = AudioValidationTask()
    success = validator.run_analysis()
    
    if success:
        print("✅ 任务8.1: 音频增强效果分析 - 完成")
    else:
        print("❌ 任务8.1: 音频增强效果分析 - 失败")
        sys.exit(1)

if __name__ == "__main__":
    main()