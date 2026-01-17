#!/usr/bin/env python3
"""
Task 8.1: 音频增强效果分析
生成前后频谱对比可视化，计算客观音频质量指标
"""

import os
import sys
import json
import subprocess
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from pathlib import Path
from datetime import datetime

# 音频处理库
try:
    import librosa
    import soundfile as sf
    from scipy import signal
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    print("警告: 音频分析库未安装")

class AudioEnhancementValidator:
    def __init__(self):
        self.setup_logging()
        self.workspace = Path("audio_workspace")
        self.workspace.mkdir(exist_ok=True)
        
        # 分析配置
        self.analysis_config = {
            'sample_duration': 60,  # 分析前60秒
            'fft_size': 2048,
            'hop_length': 512,
            'frequency_bands': {
                'sub_bass': (20, 60),
                'bass': (60, 250),
                'low_mid': (250, 500),
                'mid': (500, 2000),
                'high_mid': (2000, 4000),
                'presence': (4000, 6000),
                'brilliance': (6000, 20000)
            }
        }
        
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
        
    def load_audio_for_analysis(self, audio_file, duration=None):
        """加载音频用于分析"""
        if not AUDIO_LIBS_AVAILABLE:
            self.logger.error("音频分析库不可用")
            return None, None
            
        try:
            # 加载音频，限制时长以节省内存
            y, sr = librosa.load(audio_file, sr=48000, duration=duration)
            self.logger.info(f"加载音频: {audio_file}")
            self.logger.info(f"  时长: {len(y)/sr:.1f}s, 采样率: {sr}Hz")
            return y, sr
        except Exception as e:
            self.logger.error(f"加载音频失败: {e}")
            return None, None
            
    def calculate_audio_metrics(self, y, sr):
        """计算音频质量指标"""
        self.logger.info("📊 计算音频质量指标...")
        
        metrics = {}
        
        # 基本统计
        metrics['rms_level'] = float(np.sqrt(np.mean(y**2)))
        metrics['peak_level'] = float(np.max(np.abs(y)))
        metrics['dynamic_range_db'] = float(20 * np.log10(metrics['peak_level'] / (metrics['rms_level'] + 1e-10)))
        
        # 信噪比估算
        noise_floor = np.percentile(np.abs(y), 10)
        signal_level = np.percentile(np.abs(y), 90)
        metrics['estimated_snr_db'] = float(20 * np.log10(signal_level / (noise_floor + 1e-10)))
        
        # 频谱特征
        stft = librosa.stft(y, hop_length=self.analysis_config['hop_length'])
        magnitude = np.abs(stft)
        
        # 频谱质心和带宽
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        metrics['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        metrics['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        # 频段能量分析
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.analysis_config['fft_size'])
        freq_energy = {}
        
        for band_name, (low_freq, high_freq) in self.analysis_config['frequency_bands'].items():
            band_indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
            if len(band_indices) > 0:
                band_energy = np.mean(magnitude[band_indices, :])
                freq_energy[band_name] = float(band_energy)
            else:
                freq_energy[band_name] = 0.0
                
        metrics['frequency_energy'] = freq_energy
        
        # THD估算 (简化版)
        # 使用FFT分析谐波失真
        fft = np.fft.fft(y[:sr])  # 分析前1秒
        freqs_fft = np.fft.fftfreq(len(fft), 1/sr)
        magnitude_fft = np.abs(fft)
        
        # 找到主要频率成分
        fundamental_idx = np.argmax(magnitude_fft[:len(magnitude_fft)//2])
        fundamental_freq = freqs_fft[fundamental_idx]
        
        if fundamental_freq > 0:
            # 计算谐波
            harmonics_power = 0
            fundamental_power = magnitude_fft[fundamental_idx]**2
            
            for harmonic in range(2, 6):  # 2-5次谐波
                harmonic_freq = fundamental_freq * harmonic
                harmonic_idx = np.argmin(np.abs(freqs_fft - harmonic_freq))
                if harmonic_idx < len(magnitude_fft)//2:
                    harmonics_power += magnitude_fft[harmonic_idx]**2
                    
            thd = np.sqrt(harmonics_power / (fundamental_power + 1e-10))
            metrics['estimated_thd_percent'] = float(thd * 100)
        else:
            metrics['estimated_thd_percent'] = 0.0
            
        return metrics