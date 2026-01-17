#!/usr/bin/env python3
"""
Theater Audio Enhancement System
专业话剧音频增强处理工具

功能包括:
- 智能降噪
- 频率均衡
- 动态范围控制
- 空间音效优化
- 音视频同步
"""

import os
import sys
import json
import subprocess
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import librosa
import soundfile as sf

class TheaterAudioEnhancer:
    def __init__(self, config_file="audio_config.json"):
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.workspace = Path("audio_workspace")
        self.workspace.mkdir(exist_ok=True)
        
    def load_config(self, config_file):
        """加载音频处理配置"""
        default_config = {
            "noise_reduction": {
                "method": "spectral_subtraction",
                "strength": "medium",
                "noise_floor": -40
            },
            "equalizer": {
                "low_cut": {"freq": 100, "gain": -6},
                "speech_boost": {"freq_range": [300, 3000], "gain": 3},
                "clarity_boost": {"freq_range": [4000, 8000], "gain": 2},
                "high_cut": {"freq": 10000, "gain": -3}
            },
            "dynamics": {
                "compressor": {
                    "ratio": 4.0,
                    "threshold": -18,
                    "attack": 5,
                    "release": 100
                },
                "limiter": {
                    "threshold": -1,
                    "release": 50
                }
            },
            "spatial": {
                "reverb_time": 1.2,
                "stereo_width": 120,
                "room_size": "medium"
            }
        }
        
        if Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"配置文件加载失败，使用默认配置: {e}")
        
        return default_config
    
    def setup_logging(self):
        """设置日志系统"""
        log_file = f"audio_enhancement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_ffmpeg(self, command, check=True):
        """执行FFmpeg命令"""
        self.logger.info(f"执行FFmpeg: {command}")
        try:
            result = subprocess.run(command, shell=True, check=check,
                                  capture_output=True, text=True)
            if result.stdout:
                self.logger.debug(f"输出: {result.stdout}")
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg命令失败: {e}")
            if e.stderr:
                self.logger.error(f"错误信息: {e.stderr}")
            raise
    
    def extract_audio(self, video_file):
        """从视频中提取音频"""
        self.logger.info("从视频中提取音频...")
        
        audio_file = self.workspace / "original_audio.wav"
        cmd = f'ffmpeg -i "{video_file}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{audio_file}" -y'
        
        self.run_ffmpeg(cmd)
        
        if audio_file.exists():
            self.logger.info(f"音频提取成功: {audio_file}")
            return str(audio_file)
        else:
            raise RuntimeError("音频提取失败")
    
    def analyze_audio(self, audio_file):
        """分析音频特征"""
        self.logger.info("分析音频特征...")
        
        # 使用librosa加载音频
        y, sr = librosa.load(audio_file, sr=48000)
        
        # 基本统计信息
        duration = len(y) / sr
        rms_energy = np.sqrt(np.mean(y**2))
        peak_amplitude = np.max(np.abs(y))
        
        # 频谱分析
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        
        # 计算频谱质心和带宽
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        # 动态范围分析
        dynamic_range = 20 * np.log10(peak_amplitude / (rms_energy + 1e-10))
        
        analysis_result = {
            "duration": duration,
            "sample_rate": sr,
            "rms_energy": float(rms_energy),
            "peak_amplitude": float(peak_amplitude),
            "dynamic_range_db": float(dynamic_range),
            "spectral_centroid_mean": float(np.mean(spectral_centroids)),
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth))
        }
        
        # 保存分析结果
        with open(self.workspace / "audio_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        # 生成频谱图
        self.generate_spectrum_plot(y, sr, "original_spectrum.png")
        
        self.logger.info(f"音频分析完成: 时长{duration:.1f}秒, 动态范围{dynamic_range:.1f}dB")
        return analysis_result
    
    def generate_spectrum_plot(self, y, sr, filename):
        """生成频谱分析图"""
        plt.figure(figsize=(12, 8))
        
        # 子图1: 波形图
        plt.subplot(3, 1, 1)
        time = np.linspace(0, len(y)/sr, len(y))
        plt.plot(time[:sr*10], y[:sr*10])  # 只显示前10秒
        plt.title('音频波形 (前10秒)')
        plt.xlabel('时间 (秒)')
        plt.ylabel('振幅')
        
        # 子图2: 频谱图
        plt.subplot(3, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('频谱图')
        
        # 子图3: 频率分布
        plt.subplot(3, 1, 3)
        fft = np.fft.fft(y[:sr*10])  # 前10秒的FFT
        freqs = np.fft.fftfreq(len(fft), 1/sr)
        magnitude = np.abs(fft)
        
        # 只显示正频率部分
        pos_freqs = freqs[:len(freqs)//2]
        pos_magnitude = magnitude[:len(magnitude)//2]
        
        plt.semilogx(pos_freqs[1:], 20*np.log10(pos_magnitude[1:] + 1e-10))
        plt.title('频率响应')
        plt.xlabel('频率 (Hz)')
        plt.ylabel('幅度 (dB)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.workspace / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"频谱图已保存: {filename}")
    
    def noise_reduction(self, audio_file):
        """智能降噪处理"""
        self.logger.info("开始降噪处理...")
        
        config = self.config["noise_reduction"]
        output_file = self.workspace / "denoised_audio.wav"
        
        # 使用FFmpeg的afftdn滤镜进行降噪
        noise_floor = config["noise_floor"]
        
        if config["strength"] == "light":
            nr_strength = 0.5
        elif config["strength"] == "medium":
            nr_strength = 1.0
        else:  # heavy
            nr_strength = 1.5
        
        # FFmpeg降噪命令
        cmd = f'''ffmpeg -i "{audio_file}" -af "afftdn=nr={nr_strength}:nf={noise_floor}:tn=1" "{output_file}" -y'''
        
        self.run_ffmpeg(cmd)
        
        if output_file.exists():
            self.logger.info("降噪处理完成")
            return str(output_file)
        else:
            raise RuntimeError("降噪处理失败")
    
    def apply_equalizer(self, audio_file):
        """应用频率均衡"""
        self.logger.info("应用频率均衡...")
        
        config = self.config["equalizer"]
        output_file = self.workspace / "equalized_audio.wav"
        
        # 构建EQ滤镜链
        eq_filters = []
        
        # 低频衰减
        low_cut = config["low_cut"]
        eq_filters.append(f"highpass=f={low_cut['freq']}:p=1")
        
        # 人声增强 (使用peaking EQ)
        speech = config["speech_boost"]
        center_freq = (speech["freq_range"][0] + speech["freq_range"][1]) // 2
        eq_filters.append(f"equalizer=f={center_freq}:width_type=h:width=1000:g={speech['gain']}")
        
        # 清晰度提升
        clarity = config["clarity_boost"]
        center_freq = (clarity["freq_range"][0] + clarity["freq_range"][1]) // 2
        eq_filters.append(f"equalizer=f={center_freq}:width_type=h:width=2000:g={clarity['gain']}")
        
        # 高频控制
        high_cut = config["high_cut"]
        eq_filters.append(f"lowpass=f={high_cut['freq']}:p=1")
        
        # 组合所有滤镜
        filter_chain = ",".join(eq_filters)
        cmd = f'ffmpeg -i "{audio_file}" -af "{filter_chain}" "{output_file}" -y'
        
        self.run_ffmpeg(cmd)
        
        if output_file.exists():
            self.logger.info("频率均衡完成")
            return str(output_file)
        else:
            raise RuntimeError("频率均衡失败")
    
    def apply_dynamics_processing(self, audio_file):
        """应用动态范围控制"""
        self.logger.info("应用动态范围控制...")
        
        config = self.config["dynamics"]
        output_file = self.workspace / "compressed_audio.wav"
        
        # 压缩器参数
        comp = config["compressor"]
        limiter = config["limiter"]
        
        # 构建动态处理滤镜链
        filters = []
        
        # 压缩器
        comp_filter = f"acompressor=threshold={comp['threshold']}dB:ratio={comp['ratio']}:attack={comp['attack']}:release={comp['release']}"
        filters.append(comp_filter)
        
        # 限幅器
        limiter_filter = f"alimiter=level_in=1:level_out=0.9:limit={limiter['threshold']}:release={limiter['release']}"
        filters.append(limiter_filter)
        
        filter_chain = ",".join(filters)
        cmd = f'ffmpeg -i "{audio_file}" -af "{filter_chain}" "{output_file}" -y'
        
        self.run_ffmpeg(cmd)
        
        if output_file.exists():
            self.logger.info("动态范围控制完成")
            return str(output_file)
        else:
            raise RuntimeError("动态范围控制失败")
    
    def apply_spatial_enhancement(self, audio_file):
        """应用空间音效优化"""
        self.logger.info("应用空间音效优化...")
        
        config = self.config["spatial"]
        output_file = self.workspace / "spatial_enhanced_audio.wav"
        
        filters = []
        
        # 立体声宽度调整
        stereo_width = config["stereo_width"] / 100.0
        filters.append(f"extrastereo=m={stereo_width}")
        
        # 简单混响 (使用aecho模拟)
        reverb_time = config["reverb_time"]
        delay_ms = int(reverb_time * 100)  # 转换为毫秒
        filters.append(f"aecho=0.8:0.88:{delay_ms}:0.4")
        
        filter_chain = ",".join(filters)
        cmd = f'ffmpeg -i "{audio_file}" -af "{filter_chain}" "{output_file}" -y'
        
        self.run_ffmpeg(cmd)
        
        if output_file.exists():
            self.logger.info("空间音效优化完成")
            return str(output_file)
        else:
            raise RuntimeError("空间音效优化失败")
    
    def merge_with_video(self, enhanced_audio, original_video, output_video):
        """将增强音频与原视频合并"""
        self.logger.info("合并增强音频与视频...")
        
        # 使用高质量编码参数
        cmd = f'''ffmpeg -i "{original_video}" -i "{enhanced_audio}" -c:v copy -c:a aac -b:a 192k -ar 48000 -ac 2 -map 0:v:0 -map 1:a:0 -shortest "{output_video}" -y'''
        
        self.run_ffmpeg(cmd)
        
        if Path(output_video).exists():
            file_size = Path(output_video).stat().st_size / (1024*1024)
            self.logger.info(f"音视频合并完成: {output_video} ({file_size:.1f} MB)")
            return True
        else:
            raise RuntimeError("音视频合并失败")
    
    def generate_comparison_report(self, original_audio, enhanced_audio):
        """生成处理前后对比报告"""
        self.logger.info("生成对比分析报告...")
        
        # 分析原始音频
        y_orig, sr = librosa.load(original_audio, sr=48000)
        
        # 分析增强音频
        y_enh, _ = librosa.load(enhanced_audio, sr=48000)
        
        # 确保长度一致
        min_len = min(len(y_orig), len(y_enh))
        y_orig = y_orig[:min_len]
        y_enh = y_enh[:min_len]
        
        # 计算对比指标
        rms_orig = np.sqrt(np.mean(y_orig**2))
        rms_enh = np.sqrt(np.mean(y_enh**2))
        
        peak_orig = np.max(np.abs(y_orig))
        peak_enh = np.max(np.abs(y_enh))
        
        # 信噪比估算 (简化计算)
        noise_floor_orig = np.percentile(np.abs(y_orig), 10)
        noise_floor_enh = np.percentile(np.abs(y_enh), 10)
        
        snr_orig = 20 * np.log10(rms_orig / (noise_floor_orig + 1e-10))
        snr_enh = 20 * np.log10(rms_enh / (noise_floor_enh + 1e-10))
        
        comparison_data = {
            "original": {
                "rms_level": float(rms_orig),
                "peak_level": float(peak_orig),
                "estimated_snr_db": float(snr_orig),
                "dynamic_range_db": float(20 * np.log10(peak_orig / (rms_orig + 1e-10)))
            },
            "enhanced": {
                "rms_level": float(rms_enh),
                "peak_level": float(peak_enh),
                "estimated_snr_db": float(snr_enh),
                "dynamic_range_db": float(20 * np.log10(peak_enh / (rms_enh + 1e-10)))
            },
            "improvements": {
                "snr_improvement_db": float(snr_enh - snr_orig),
                "noise_reduction_db": float(20 * np.log10(noise_floor_orig / (noise_floor_enh + 1e-10))),
                "rms_change_db": float(20 * np.log10(rms_enh / (rms_orig + 1e-10)))
            }
        }
        
        # 保存对比报告
        with open(self.workspace / "audio_comparison_report.json", 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        # 生成对比频谱图
        self.generate_comparison_plot(y_orig, y_enh, sr)
        
        self.logger.info(f"音频增强效果: SNR提升{comparison_data['improvements']['snr_improvement_db']:.1f}dB")
        return comparison_data
    
    def generate_detailed_analysis_report(self, audio_file, basic_analysis):
        """生成详细的音频质量分析报告 - Task 7.1"""
        self.logger.info("生成详细音频质量分析报告...")
        
        # 加载音频数据
        y, sr = librosa.load(audio_file, sr=48000)
        duration = len(y) / sr
        
        # 1. 基本音频特征分析
        rms_energy = np.sqrt(np.mean(y**2))
        peak_amplitude = np.max(np.abs(y))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        
        # 2. 频谱特征分析
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # 3. 噪音分析
        # 使用静音段估算噪音底噪
        silence_threshold = np.percentile(np.abs(y), 20)  # 20%分位数作为静音阈值
        silence_mask = np.abs(y) < silence_threshold
        noise_floor = np.mean(np.abs(y[silence_mask])) if np.any(silence_mask) else silence_threshold
        
        # 估算信噪比
        signal_power = rms_energy**2
        noise_power = noise_floor**2
        snr_estimate = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # 4. 动态范围分析
        # 计算不同百分位数的音量分布
        volume_percentiles = {
            "10th": np.percentile(np.abs(y), 10),
            "25th": np.percentile(np.abs(y), 25),
            "50th": np.percentile(np.abs(y), 50),
            "75th": np.percentile(np.abs(y), 75),
            "90th": np.percentile(np.abs(y), 90),
            "95th": np.percentile(np.abs(y), 95),
            "99th": np.percentile(np.abs(y), 99)
        }
        
        # 计算峰值因子 (Crest Factor)
        crest_factor = 20 * np.log10(peak_amplitude / (rms_energy + 1e-10))
        
        # 5. 话剧特有音频特征分析
        # 检测语音活动段
        # 使用能量和过零率检测语音段
        frame_length = 2048
        hop_length = 512
        
        # 计算短时能量
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        frame_energy = np.sum(frames**2, axis=0)
        
        # 语音活动检测 (简化版VAD)
        energy_threshold = np.percentile(frame_energy, 60)  # 60%分位数作为语音阈值
        speech_frames = frame_energy > energy_threshold
        speech_ratio = np.sum(speech_frames) / len(speech_frames)
        
        # 6. 频率分布分析
        # 计算不同频段的能量分布
        stft = librosa.stft(y, hop_length=hop_length)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr)
        
        # 定义频段
        freq_bands = {
            "sub_bass": (20, 60),      # 超低频
            "bass": (60, 250),         # 低频
            "low_mid": (250, 500),     # 中低频
            "mid": (500, 2000),        # 中频
            "high_mid": (2000, 4000),  # 中高频 (人声关键区域)
            "presence": (4000, 6000),  # 存在感频段
            "brilliance": (6000, 20000) # 高频
        }
        
        freq_energy = {}
        for band_name, (low_freq, high_freq) in freq_bands.items():
            band_indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
            if len(band_indices) > 0:
                band_energy = np.mean(magnitude[band_indices, :])
                freq_energy[band_name] = float(band_energy)
            else:
                freq_energy[band_name] = 0.0
        
        # 7. 立体声分析 (如果是立体声)
        stereo_analysis = {}
        if len(y.shape) > 1 or audio_file.endswith('.wav'):
            # 重新加载为立体声
            y_stereo, _ = librosa.load(audio_file, sr=48000, mono=False)
            if len(y_stereo.shape) > 1:
                left_channel = y_stereo[0]
                right_channel = y_stereo[1]
                
                # 计算立体声相关性
                correlation = np.corrcoef(left_channel, right_channel)[0, 1]
                
                # 计算左右声道能量差异
                left_rms = np.sqrt(np.mean(left_channel**2))
                right_rms = np.sqrt(np.mean(right_channel**2))
                channel_balance = 20 * np.log10((left_rms + 1e-10) / (right_rms + 1e-10))
                
                stereo_analysis = {
                    "correlation": float(correlation),
                    "channel_balance_db": float(channel_balance),
                    "left_rms": float(left_rms),
                    "right_rms": float(right_rms),
                    "stereo_width": float(1.0 - abs(correlation))  # 立体声宽度估算
                }
        
        # 8. 话剧音频质量评估
        theater_quality_assessment = {
            "dialogue_clarity": "good" if freq_energy.get("high_mid", 0) > freq_energy.get("bass", 0) else "needs_improvement",
            "background_noise_level": "low" if snr_estimate > 20 else "moderate" if snr_estimate > 10 else "high",
            "dynamic_range_type": "wide" if crest_factor > 15 else "moderate" if crest_factor > 10 else "compressed",
            "speech_activity_ratio": float(speech_ratio),
            "overall_quality": "excellent" if snr_estimate > 25 and crest_factor > 15 else 
                             "good" if snr_estimate > 15 and crest_factor > 10 else
                             "fair" if snr_estimate > 10 else "poor"
        }
        
        # 9. 处理建议
        processing_recommendations = {
            "noise_reduction": "heavy" if snr_estimate < 10 else "medium" if snr_estimate < 20 else "light",
            "eq_adjustments": {
                "bass_cut": freq_energy.get("bass", 0) > freq_energy.get("mid", 0),
                "speech_boost": freq_energy.get("high_mid", 0) < freq_energy.get("mid", 0),
                "presence_boost": freq_energy.get("presence", 0) < freq_energy.get("high_mid", 0)
            },
            "dynamics_processing": {
                "compression_needed": crest_factor > 20,
                "limiting_needed": peak_amplitude > 0.9
            },
            "spatial_enhancement": len(stereo_analysis) > 0 and stereo_analysis.get("correlation", 1.0) > 0.9
        }
        
        # 10. 汇总分析结果
        detailed_analysis = {
            "file_info": {
                "duration_seconds": float(duration),
                "sample_rate": int(sr),
                "channels": 2 if len(stereo_analysis) > 0 else 1,
                "file_path": audio_file
            },
            "basic_metrics": {
                "rms_energy": float(rms_energy),
                "peak_amplitude": float(peak_amplitude),
                "dynamic_range_db": float(crest_factor),
                "zero_crossing_rate": float(zero_crossing_rate),
                "estimated_snr_db": float(snr_estimate)
            },
            "spectral_features": {
                "spectral_centroid_mean": float(np.mean(spectral_centroids)),
                "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "spectral_contrast_mean": float(np.mean(spectral_contrast))
            },
            "noise_analysis": {
                "noise_floor": float(noise_floor),
                "silence_ratio": float(np.sum(silence_mask) / len(silence_mask)),
                "snr_estimate_db": float(snr_estimate)
            },
            "volume_distribution": {k: float(v) for k, v in volume_percentiles.items()},
            "frequency_energy_distribution": freq_energy,
            "stereo_analysis": stereo_analysis,
            "theater_specific": {
                "speech_activity_ratio": float(speech_ratio),
                "dialogue_prominence": float(freq_energy.get("high_mid", 0) / (freq_energy.get("bass", 1) + 1e-10)),
                "ambient_noise_level": float(noise_floor),
                "stage_acoustics_estimate": float(np.mean(spectral_rolloff) / sr * 100)  # 简化的声学特征
            },
            "quality_assessment": theater_quality_assessment,
            "processing_recommendations": processing_recommendations,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # 保存详细分析报告
        report_file = self.workspace / "detailed_audio_analysis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_analysis, f, indent=2, ensure_ascii=False)
        
        # 生成可视化分析图表
        self.generate_comprehensive_analysis_plots(y, sr, detailed_analysis)
        
        # 生成Markdown格式的分析报告
        self.generate_markdown_analysis_report(detailed_analysis)
        
        self.logger.info(f"详细音频分析完成:")
        self.logger.info(f"  - 时长: {duration:.1f}秒")
        self.logger.info(f"  - 估算SNR: {snr_estimate:.1f}dB")
        self.logger.info(f"  - 动态范围: {crest_factor:.1f}dB")
        self.logger.info(f"  - 语音活动比例: {speech_ratio:.1%}")
        self.logger.info(f"  - 整体质量评估: {theater_quality_assessment['overall_quality']}")
        self.logger.info(f"  - 报告已保存: {report_file}")
        
        return detailed_analysis

    def generate_comprehensive_analysis_plots(self, y, sr, analysis_data):
        """生成综合音频分析图表"""
        plt.style.use('default')
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 波形图
        plt.subplot(3, 3, 1)
        time = np.linspace(0, len(y)/sr, len(y))
        plt.plot(time[:sr*30], y[:sr*30])  # 显示前30秒
        plt.title('音频波形 (前30秒)')
        plt.xlabel('时间 (秒)')
        plt.ylabel('振幅')
        plt.grid(True, alpha=0.3)
        
        # 2. 频谱图
        plt.subplot(3, 3, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y[:sr*60])), ref=np.max)  # 前60秒
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('频谱图 (前60秒)')
        
        # 3. 频率响应
        plt.subplot(3, 3, 3)
        fft = np.fft.fft(y[:sr*10])  # 前10秒的FFT
        freqs = np.fft.fftfreq(len(fft), 1/sr)
        magnitude = np.abs(fft)
        pos_freqs = freqs[:len(freqs)//2]
        pos_magnitude = magnitude[:len(magnitude)//2]
        
        plt.semilogx(pos_freqs[1:], 20*np.log10(pos_magnitude[1:] + 1e-10))
        plt.title('频率响应')
        plt.xlabel('频率 (Hz)')
        plt.ylabel('幅度 (dB)')
        plt.grid(True, alpha=0.3)
        
        # 4. 频段能量分布
        plt.subplot(3, 3, 4)
        freq_bands = analysis_data['frequency_energy_distribution']
        bands = list(freq_bands.keys())
        energies = list(freq_bands.values())
        
        plt.bar(bands, energies, color='skyblue', alpha=0.7)
        plt.title('频段能量分布')
        plt.xlabel('频段')
        plt.ylabel('能量')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 5. 音量分布直方图
        plt.subplot(3, 3, 5)
        plt.hist(np.abs(y), bins=50, alpha=0.7, color='green', density=True)
        plt.title('音量分布')
        plt.xlabel('振幅')
        plt.ylabel('概率密度')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 6. 动态范围可视化
        plt.subplot(3, 3, 6)
        volume_percentiles = analysis_data['volume_distribution']
        percentiles = list(volume_percentiles.keys())
        values = list(volume_percentiles.values())
        
        plt.plot(percentiles, values, 'o-', color='red', linewidth=2, markersize=6)
        plt.title('音量百分位数分布')
        plt.xlabel('百分位数')
        plt.ylabel('振幅')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 7. 语音活动检测
        plt.subplot(3, 3, 7)
        frame_length = 2048
        hop_length = 512
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        frame_energy = np.sum(frames**2, axis=0)
        frame_times = librosa.frames_to_time(np.arange(len(frame_energy)), sr=sr, hop_length=hop_length)
        
        energy_threshold = np.percentile(frame_energy, 60)
        
        plt.plot(frame_times[:min(len(frame_times), sr//hop_length*60)], 
                frame_energy[:min(len(frame_energy), sr//hop_length*60)], 
                alpha=0.7, label='帧能量')
        plt.axhline(y=energy_threshold, color='red', linestyle='--', label='语音阈值')
        plt.title('语音活动检测 (前60秒)')
        plt.xlabel('时间 (秒)')
        plt.ylabel('能量')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. 立体声分析 (如果有)
        plt.subplot(3, 3, 8)
        if analysis_data['stereo_analysis']:
            stereo = analysis_data['stereo_analysis']
            metrics = ['相关性', '声道平衡', '立体声宽度']
            values = [stereo['correlation'], 
                     abs(stereo['channel_balance_db'])/20,  # 归一化到0-1
                     stereo['stereo_width']]
            
            plt.bar(metrics, values, color=['blue', 'orange', 'green'], alpha=0.7)
            plt.title('立体声特征')
            plt.ylabel('归一化值')
            plt.ylim(0, 1)
        else:
            plt.text(0.5, 0.5, '单声道音频', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('立体声分析')
        plt.grid(True, alpha=0.3)
        
        # 9. 质量评估雷达图
        plt.subplot(3, 3, 9, projection='polar')
        
        # 质量指标 (转换为0-1范围)
        snr_score = min(analysis_data['basic_metrics']['estimated_snr_db'] / 30, 1.0)
        dynamic_score = min(analysis_data['basic_metrics']['dynamic_range_db'] / 25, 1.0)
        speech_score = analysis_data['theater_specific']['speech_activity_ratio']
        dialogue_score = min(analysis_data['theater_specific']['dialogue_prominence'] / 2, 1.0)
        
        categories = ['SNR', '动态范围', '语音活动', '对话突出度']
        values = [snr_score, dynamic_score, speech_score, dialogue_score]
        
        # 闭合雷达图
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        plt.plot(angles, values, 'o-', linewidth=2, color='purple')
        plt.fill(angles, values, alpha=0.25, color='purple')
        plt.xticks(angles[:-1], categories)
        plt.ylim(0, 1)
        plt.title('音频质量评估')
        
        plt.tight_layout()
        plt.savefig(self.workspace / "comprehensive_audio_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info("综合音频分析图表已生成")

    def generate_markdown_analysis_report(self, analysis_data):
        """生成Markdown格式的分析报告"""
        report_content = f"""# 话剧音频质量分析报告

## 基本信息
- **文件路径**: {analysis_data['file_info']['file_path']}
- **时长**: {analysis_data['file_info']['duration_seconds']:.1f} 秒 ({analysis_data['file_info']['duration_seconds']/60:.1f} 分钟)
- **采样率**: {analysis_data['file_info']['sample_rate']} Hz
- **声道数**: {analysis_data['file_info']['channels']}
- **分析时间**: {analysis_data['analysis_timestamp']}

## 音频质量评估

### 整体质量: **{analysis_data['quality_assessment']['overall_quality'].upper()}**

### 关键指标
| 指标 | 数值 | 评估 |
|------|------|------|
| 估算信噪比 | {analysis_data['basic_metrics']['estimated_snr_db']:.1f} dB | {analysis_data['quality_assessment']['background_noise_level']} |
| 动态范围 | {analysis_data['basic_metrics']['dynamic_range_db']:.1f} dB | {analysis_data['quality_assessment']['dynamic_range_type']} |
| 语音活动比例 | {analysis_data['theater_specific']['speech_activity_ratio']:.1%} | - |
| 对话清晰度 | - | {analysis_data['quality_assessment']['dialogue_clarity']} |

## 详细分析结果

### 1. 基础音频特征
- **RMS能量**: {analysis_data['basic_metrics']['rms_energy']:.4f}
- **峰值振幅**: {analysis_data['basic_metrics']['peak_amplitude']:.4f}
- **过零率**: {analysis_data['basic_metrics']['zero_crossing_rate']:.4f}

### 2. 频谱特征
- **频谱质心均值**: {analysis_data['spectral_features']['spectral_centroid_mean']:.1f} Hz
- **频谱带宽均值**: {analysis_data['spectral_features']['spectral_bandwidth_mean']:.1f} Hz
- **频谱滚降均值**: {analysis_data['spectral_features']['spectral_rolloff_mean']:.1f} Hz

### 3. 噪音分析
- **噪音底噪**: {analysis_data['noise_analysis']['noise_floor']:.6f}
- **静音比例**: {analysis_data['noise_analysis']['silence_ratio']:.1%}
- **估算SNR**: {analysis_data['noise_analysis']['snr_estimate_db']:.1f} dB

### 4. 频段能量分布
"""
        
        for band, energy in analysis_data['frequency_energy_distribution'].items():
            report_content += f"- **{band}**: {energy:.4f}\n"
        
        if analysis_data['stereo_analysis']:
            report_content += f"""
### 5. 立体声分析
- **声道相关性**: {analysis_data['stereo_analysis']['correlation']:.3f}
- **声道平衡**: {analysis_data['stereo_analysis']['channel_balance_db']:.1f} dB
- **立体声宽度**: {analysis_data['stereo_analysis']['stereo_width']:.3f}
"""
        
        report_content += f"""
### 6. 话剧特有特征
- **语音活动比例**: {analysis_data['theater_specific']['speech_activity_ratio']:.1%}
- **对话突出度**: {analysis_data['theater_specific']['dialogue_prominence']:.2f}
- **环境噪音水平**: {analysis_data['theater_specific']['ambient_noise_level']:.6f}
- **舞台声学估算**: {analysis_data['theater_specific']['stage_acoustics_estimate']:.2f}

## 处理建议

### 降噪处理
- **推荐强度**: {analysis_data['processing_recommendations']['noise_reduction']}

### 均衡器调整
"""
        
        eq_adj = analysis_data['processing_recommendations']['eq_adjustments']
        if eq_adj['bass_cut']:
            report_content += "- ✅ 建议进行低频衰减\n"
        if eq_adj['speech_boost']:
            report_content += "- ✅ 建议增强人声频段\n"
        if eq_adj['presence_boost']:
            report_content += "- ✅ 建议提升存在感频段\n"
        
        dynamics = analysis_data['processing_recommendations']['dynamics_processing']
        report_content += f"""
### 动态处理
- **需要压缩**: {'是' if dynamics['compression_needed'] else '否'}
- **需要限幅**: {'是' if dynamics['limiting_needed'] else '否'}

### 空间增强
- **需要立体声增强**: {'是' if analysis_data['processing_recommendations']['spatial_enhancement'] else '否'}

## 音量分布统计
"""
        
        for percentile, value in analysis_data['volume_distribution'].items():
            report_content += f"- **{percentile}**: {value:.6f}\n"
        
        report_content += """
## 结论与建议

基于以上分析，该话剧音频文件的主要特点和建议处理方案如下：

1. **音频质量**: 整体质量评估为 **{overall_quality}**
2. **主要问题**: {main_issues}
3. **优先处理**: {priority_processing}
4. **预期改善**: 通过建议的处理流程，预计可以显著提升音频质量

---
*本报告由话剧音频增强系统自动生成*
""".format(
            overall_quality=analysis_data['quality_assessment']['overall_quality'],
            main_issues="背景噪音" if analysis_data['basic_metrics']['estimated_snr_db'] < 15 else "动态范围控制",
            priority_processing="降噪和均衡处理" if analysis_data['basic_metrics']['estimated_snr_db'] < 15 else "动态范围优化"
        )
        
        # 保存Markdown报告
        report_file = self.workspace / "audio_quality_assessment_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Markdown分析报告已生成: {report_file}")

    def generate_comparison_plot(self, y_orig, y_enh, sr):
        plt.figure(figsize=(15, 10))
        
        # 原始音频频谱
        plt.subplot(2, 2, 1)
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig[:sr*30])), ref=np.max)
        librosa.display.specshow(D_orig, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('原始音频频谱 (前30秒)')
        
        # 增强音频频谱
        plt.subplot(2, 2, 2)
        D_enh = librosa.amplitude_to_db(np.abs(librosa.stft(y_enh[:sr*30])), ref=np.max)
        librosa.display.specshow(D_enh, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('增强音频频谱 (前30秒)')
        
        # 频率响应对比
        plt.subplot(2, 1, 2)
        
        # 计算平均频谱
        fft_orig = np.fft.fft(y_orig[:sr*10])
        fft_enh = np.fft.fft(y_enh[:sr*10])
        
        freqs = np.fft.fftfreq(len(fft_orig), 1/sr)
        pos_freqs = freqs[:len(freqs)//2]
        
        mag_orig = np.abs(fft_orig[:len(fft_orig)//2])
        mag_enh = np.abs(fft_enh[:len(fft_enh)//2])
        
        plt.semilogx(pos_freqs[1:], 20*np.log10(mag_orig[1:] + 1e-10), 
                    label='原始音频', alpha=0.7)
        plt.semilogx(pos_freqs[1:], 20*np.log10(mag_enh[1:] + 1e-10), 
                    label='增强音频', alpha=0.7)
        
        plt.title('频率响应对比')
        plt.xlabel('频率 (Hz)')
        plt.ylabel('幅度 (dB)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.workspace / "audio_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info("对比图表已生成")
    
    def enhance_audio(self, video_file, output_video=None):
        """完整的音频增强流程"""
        start_time = datetime.now()
        
        if output_video is None:
            video_path = Path(video_file)
            output_video = f"enhanced_audio_{video_path.stem}.mp4"
        
        try:
            self.logger.info("🎵 开始话剧音频增强处理")
            self.logger.info(f"输入视频: {video_file}")
            self.logger.info(f"输出视频: {output_video}")
            
            # 步骤1: 提取音频
            original_audio = self.extract_audio(video_file)
            
            # 步骤2: 分析音频
            analysis = self.analyze_audio(original_audio)
            
            # 步骤3: 降噪处理
            denoised_audio = self.noise_reduction(original_audio)
            
            # 步骤4: 频率均衡
            equalized_audio = self.apply_equalizer(denoised_audio)
            
            # 步骤5: 动态范围控制
            compressed_audio = self.apply_dynamics_processing(equalized_audio)
            
            # 步骤6: 空间音效优化
            enhanced_audio = self.apply_spatial_enhancement(compressed_audio)
            
            # 步骤7: 与视频合并
            self.merge_with_video(enhanced_audio, video_file, output_video)
            
            # 步骤8: 生成对比报告
            comparison = self.generate_comparison_report(original_audio, enhanced_audio)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"🎉 音频增强完成! 处理时间: {processing_time:.1f}秒")
            self.logger.info(f"✅ 输出文件: {output_video}")
            self.logger.info(f"📊 SNR提升: {comparison['improvements']['snr_improvement_db']:.1f}dB")
            
            return output_video
            
        except Exception as e:
            self.logger.error(f"❌ 音频增强失败: {str(e)}")
            raise

def main():
    """主程序入口"""
    if len(sys.argv) < 2:
        print("使用方法: python theater_audio_enhancer.py input_video.mp4 [output_video.mp4]")
        print("或者: python theater_audio_enhancer.py --analyze-only input_video.mp4")
        sys.exit(1)
    
    # 检查是否只进行音频分析
    if sys.argv[1] == "--analyze-only" and len(sys.argv) >= 3:
        input_video = sys.argv[2]
        enhancer = TheaterAudioEnhancer()
        
        # 只执行音频分析部分
        print("🎵 开始话剧音频质量分析...")
        original_audio = enhancer.extract_audio(input_video)
        analysis = enhancer.analyze_audio(original_audio)
        
        # 生成详细的分析报告
        enhancer.generate_detailed_analysis_report(original_audio, analysis)
        print("✅ 音频质量分析完成!")
        return
    
    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else None
    
    enhancer = TheaterAudioEnhancer()
    enhancer.enhance_audio(input_video, output_video)

if __name__ == "__main__":
    main()
    main()