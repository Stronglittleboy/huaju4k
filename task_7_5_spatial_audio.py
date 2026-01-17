#!/usr/bin/env python3
"""
Task 7.5: 空间音频增强和优化
为话剧音频添加空间感和舞台临场感
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

# 音频处理库
try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    print("警告: 音频处理库未安装")

class SpatialAudioEnhancer:
    def __init__(self):
        self.setup_logging()
        self.workspace = Path("audio_workspace")
        self.workspace.mkdir(exist_ok=True)
        
        # 空间音频配置
        self.spatial_config = {
            "reverb": {
                "room_size": "medium",  # small, medium, large
                "reverb_time": 1.2,     # 秒
                "early_reflections": 0.3,
                "diffusion": 0.7
            },
            "stereo_enhancement": {
                "width_factor": 1.2,    # 立体声宽度增强
                "center_extraction": 0.8,
                "side_enhancement": 1.1
            },
            "stage_acoustics": {
                "theater_type": "arena",  # arena, thrust, proscenium
                "audience_distance": "medium",
                "acoustic_treatment": "moderate"
            }
        }
        
    def setup_logging(self):
        """设置日志系统"""
        log_file = f"task_7_5_spatial_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def run_ffmpeg(self, command, description="FFmpeg操作"):
        """执行FFmpeg命令"""
        self.logger.info(f"{description}: {command}")
        try:
            result = subprocess.run(command, shell=True, check=True,
                                  capture_output=True, text=True)
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"{description}失败: {e}")
            if e.stderr:
                self.logger.error(f"错误信息: {e.stderr}")
            raise
            
    def analyze_spatial_characteristics(self, audio_file):
        """分析原始录音的空间特征"""
        self.logger.info("🎭 分析舞台空间特征...")
        
        if not AUDIO_LIBS_AVAILABLE:
            self.logger.warning("音频分析库不可用，跳过空间分析")
            return {}
            
        # 加载立体声音频
        y, sr = librosa.load(audio_file, sr=48000, mono=False)
        
        if len(y.shape) == 1:
            self.logger.warning("输入为单声道，空间增强效果有限")
            return {"mono_input": True}
            
        left_channel = y[0]
        right_channel = y[1]
        
        # 计算立体声特征
        correlation = np.corrcoef(left_channel, right_channel)[0, 1]
        
        # 计算左右声道能量差异
        left_rms = np.sqrt(np.mean(left_channel**2))
        right_rms = np.sqrt(np.mean(right_channel**2))
        channel_balance = 20 * np.log10((left_rms + 1e-10) / (right_rms + 1e-10))
        
        # 计算立体声宽度
        mid_signal = (left_channel + right_channel) / 2
        side_signal = (left_channel - right_channel) / 2
        
        mid_rms = np.sqrt(np.mean(mid_signal**2))
        side_rms = np.sqrt(np.mean(side_signal**2))
        stereo_width = side_rms / (mid_rms + 1e-10)
        
        # 分析混响特征
        # 使用自相关函数估算混响时间
        autocorr = np.correlate(left_channel, left_channel, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # 寻找混响衰减点
        decay_threshold = 0.1 * np.max(autocorr)
        decay_indices = np.where(autocorr < decay_threshold)[0]
        estimated_rt60 = decay_indices[0] / sr if len(decay_indices) > 0 else 0.5
        
        spatial_analysis = {
            "stereo_correlation": float(correlation),
            "channel_balance_db": float(channel_balance),
            "stereo_width": float(stereo_width),
            "estimated_rt60": float(estimated_rt60),
            "left_rms": float(left_rms),
            "right_rms": float(right_rms),
            "mid_rms": float(mid_rms),
            "side_rms": float(side_rms),
            "spatial_quality": "good" if stereo_width > 0.3 else "needs_enhancement"
        }
        
        # 保存分析结果
        with open(self.workspace / "spatial_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(spatial_analysis, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"空间分析完成:")
        self.logger.info(f"  - 立体声相关性: {correlation:.3f}")
        self.logger.info(f"  - 声道平衡: {channel_balance:.1f}dB")
        self.logger.info(f"  - 立体声宽度: {stereo_width:.3f}")
        self.logger.info(f"  - 估算RT60: {estimated_rt60:.2f}s")
        
        return spatial_analysis
        
    def enhance_stereo_width(self, audio_file):
        """增强立体声宽度"""
        self.logger.info("🎵 增强立体声宽度...")
        
        output_file = self.workspace / "stereo_enhanced_audio.wav"
        width_factor = self.spatial_config["stereo_enhancement"]["width_factor"]
        
        # 使用FFmpeg的extrastereo滤镜
        cmd = f'ffmpeg -i "{audio_file}" -af "extrastereo=m={width_factor}" "{output_file}" -y'
        self.run_ffmpeg(cmd, "立体声宽度增强")
        
        if output_file.exists():
            self.logger.info("✅ 立体声宽度增强完成")
            return str(output_file)
        else:
            raise RuntimeError("立体声宽度增强失败")
            
    def add_theater_reverb(self, audio_file):
        """添加话剧舞台混响效果"""
        self.logger.info("🎭 添加舞台混响效果...")
        
        output_file = self.workspace / "reverb_enhanced_audio.wav"
        reverb_config = self.spatial_config["reverb"]
        
        # 根据房间大小设置参数
        if reverb_config["room_size"] == "small":
            reverb_params = "0.8:0.88:60:0.4"
        elif reverb_config["room_size"] == "medium":
            reverb_params = "0.8:0.88:120:0.4"
        else:  # large
            reverb_params = "0.8:0.88:200:0.4"
            
        # 使用aecho滤镜模拟混响
        cmd = f'ffmpeg -i "{audio_file}" -af "aecho={reverb_params}" "{output_file}" -y'
        self.run_ffmpeg(cmd, "舞台混响添加")
        
        if output_file.exists():
            self.logger.info("✅ 舞台混响添加完成")
            return str(output_file)
        else:
            raise RuntimeError("舞台混响添加失败")
            
    def enhance_stage_presence(self, audio_file):
        """增强舞台临场感"""
        self.logger.info("🎪 增强舞台临场感...")
        
        output_file = self.workspace / "stage_presence_audio.wav"
        
        # 组合多种效果增强临场感
        filters = []
        
        # 1. 轻微的合唱效果（模拟空间反射）
        filters.append("chorus=0.5:0.9:50:0.4:0.25:2")
        
        # 2. 频率响应调整（增强空间感）
        filters.append("equalizer=f=250:width_type=h:width=100:g=-1")  # 轻微衰减低频
        filters.append("equalizer=f=8000:width_type=h:width=2000:g=1")  # 轻微提升高频
        
        # 3. 动态范围微调
        filters.append("acompressor=threshold=-25dB:ratio=2:attack=10:release=200")
        
        filter_chain = ",".join(filters)
        cmd = f'ffmpeg -i "{audio_file}" -af "{filter_chain}" "{output_file}" -y'
        self.run_ffmpeg(cmd, "舞台临场感增强")
        
        if output_file.exists():
            self.logger.info("✅ 舞台临场感增强完成")
            return str(output_file)
        else:
            raise RuntimeError("舞台临场感增强失败")
            
    def optimize_theater_acoustics(self, audio_file):
        """优化话剧声学环境"""
        self.logger.info("🏛️ 优化话剧声学环境...")
        
        output_file = self.workspace / "acoustics_optimized_audio.wav"
        stage_config = self.spatial_config["stage_acoustics"]
        
        filters = []
        
        # 根据剧场类型调整声学特性
        if stage_config["theater_type"] == "arena":
            # 环形剧场：增强环绕感
            filters.append("extrastereo=m=1.1")
            filters.append("aecho=0.8:0.88:80:0.3")
        elif stage_config["theater_type"] == "thrust":
            # 伸展式舞台：平衡前后声场
            filters.append("extrastereo=m=1.0")
            filters.append("aecho=0.8:0.88:100:0.35")
        else:  # proscenium
            # 镜框式舞台：传统剧场效果
            filters.append("extrastereo=m=0.9")
            filters.append("aecho=0.8:0.88:150:0.4")
            
        # 观众距离调整
        if stage_config["audience_distance"] == "close":
            filters.append("equalizer=f=2000:width_type=h:width=1000:g=1")
        elif stage_config["audience_distance"] == "far":
            filters.append("equalizer=f=4000:width_type=h:width=2000:g=-1")
            filters.append("aecho=0.7:0.85:200:0.3")
            
        filter_chain = ",".join(filters)
        cmd = f'ffmpeg -i "{audio_file}" -af "{filter_chain}" "{output_file}" -y'
        self.run_ffmpeg(cmd, "声学环境优化")
        
        if output_file.exists():
            self.logger.info("✅ 声学环境优化完成")
            return str(output_file)
        else:
            raise RuntimeError("声学环境优化失败")
            
    def generate_spatial_comparison(self, original_file, enhanced_file):
        """生成空间增强对比分析"""
        self.logger.info("📊 生成空间增强对比分析...")
        
        if not AUDIO_LIBS_AVAILABLE:
            self.logger.warning("音频分析库不可用，跳过对比分析")
            return {}
            
        # 分析原始音频
        y_orig, sr = librosa.load(original_file, sr=48000, mono=False)
        # 分析增强音频
        y_enh, _ = librosa.load(enhanced_file, sr=48000, mono=False)
        
        comparison_data = {}
        
        if len(y_orig.shape) > 1 and len(y_enh.shape) > 1:
            # 立体声分析
            orig_corr = np.corrcoef(y_orig[0], y_orig[1])[0, 1]
            enh_corr = np.corrcoef(y_enh[0], y_enh[1])[0, 1]
            
            # 立体声宽度计算
            orig_width = np.std(y_orig[0] - y_orig[1]) / (np.std(y_orig[0] + y_orig[1]) + 1e-10)
            enh_width = np.std(y_enh[0] - y_enh[1]) / (np.std(y_enh[0] + y_enh[1]) + 1e-10)
            
            comparison_data = {
                "original": {
                    "stereo_correlation": float(orig_corr),
                    "stereo_width": float(orig_width)
                },
                "enhanced": {
                    "stereo_correlation": float(enh_corr),
                    "stereo_width": float(enh_width)
                },
                "improvements": {
                    "correlation_change": float(enh_corr - orig_corr),
                    "width_improvement": float(enh_width - orig_width)
                }
            }
            
        # 保存对比结果
        with open(self.workspace / "spatial_comparison.json", 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
            
        self.logger.info("✅ 空间增强对比分析完成")
        return comparison_data
        
    def process_spatial_enhancement(self, input_audio):
        """完整的空间音频增强流程"""
        start_time = datetime.now()
        
        try:
            self.logger.info("🎭 开始空间音频增强处理")
            self.logger.info(f"输入音频: {input_audio}")
            
            # 步骤1: 分析空间特征
            spatial_analysis = self.analyze_spatial_characteristics(input_audio)
            
            # 步骤2: 增强立体声宽度
            stereo_enhanced = self.enhance_stereo_width(input_audio)
            
            # 步骤3: 添加舞台混响
            reverb_enhanced = self.add_theater_reverb(stereo_enhanced)
            
            # 步骤4: 增强舞台临场感
            presence_enhanced = self.enhance_stage_presence(reverb_enhanced)
            
            # 步骤5: 优化声学环境
            final_enhanced = self.optimize_theater_acoustics(presence_enhanced)
            
            # 步骤6: 生成对比分析
            comparison = self.generate_spatial_comparison(input_audio, final_enhanced)
            
            # 生成处理报告
            processing_time = (datetime.now() - start_time).total_seconds()
            
            report = {
                "task": "7.5 Spatial audio enhancement and optimization",
                "input_file": input_audio,
                "output_file": final_enhanced,
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "spatial_config": self.spatial_config,
                "spatial_analysis": spatial_analysis,
                "comparison_results": comparison,
                "processing_steps": [
                    "空间特征分析",
                    "立体声宽度增强",
                    "舞台混响添加",
                    "临场感增强",
                    "声学环境优化"
                ],
                "quality_assessment": "显著改善" if comparison.get("improvements", {}).get("width_improvement", 0) > 0.1 else "适度改善",
                "processing_successful": True
            }
            
            # 保存报告
            with open(self.workspace / "task_7_5_spatial_enhancement_report.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"🎉 空间音频增强完成! 处理时间: {processing_time:.1f}秒")
            self.logger.info(f"✅ 输出文件: {final_enhanced}")
            
            return final_enhanced, report
            
        except Exception as e:
            self.logger.error(f"❌ 空间音频增强失败: {str(e)}")
            raise

def main():
    if len(sys.argv) < 2:
        print("使用方法: python3 task_7_5_spatial_audio.py input_audio.wav")
        sys.exit(1)
    
    input_audio = sys.argv[1]
    
    enhancer = SpatialAudioEnhancer()
    enhanced_audio, report = enhancer.process_spatial_enhancement(input_audio)
    
    print(f"空间音频增强完成: {enhanced_audio}")

if __name__ == "__main__":
    main()