#!/usr/bin/env python3
"""
Simple script to run audio analysis for Task 7.1
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    from theater_audio_enhancer import TheaterAudioEnhancer
    
    def main():
        print("🎵 开始话剧音频质量分析 - Task 7.1")
        
        # 输入视频文件
        video_file = "videos/大学生原创话剧《自杀既遂》.mp4"
        
        if not Path(video_file).exists():
            print(f"❌ 视频文件不存在: {video_file}")
            return
        
        # 创建音频增强器实例
        enhancer = TheaterAudioEnhancer("audio_enhancement_config.json")
        
        print(f"📁 输入视频: {video_file}")
        
        # 步骤1: 直接分析视频中的音频 (使用librosa)
        print("步骤1: 直接从视频中加载音频进行分析...")
        try:
            # 使用librosa直接从视频文件加载音频
            import librosa
            y, sr = librosa.load(video_file, sr=48000)
            print(f"✅ 音频加载成功: 采样率 {sr} Hz, 时长 {len(y)/sr:.1f} 秒")
            
            # 创建临时音频文件路径用于分析
            import tempfile
            import soundfile as sf
            temp_audio = "temp_audio_for_analysis.wav"
            sf.write(temp_audio, y, sr)
            
            # 步骤2: 基本音频分析
            print("步骤2: 进行基本音频分析...")
            basic_analysis = enhancer.analyze_audio(temp_audio)
            
            original_audio = temp_audio
            
        except Exception as e:
            print(f"❌ 直接音频加载失败: {e}")
            print("尝试使用FFmpeg提取音频...")
            # 步骤1: 提取音频
            print("步骤1: 从视频中提取音频...")
            original_audio = enhancer.extract_audio(video_file)
            
            # 步骤2: 基本音频分析
            print("步骤2: 进行基本音频分析...")
            basic_analysis = enhancer.analyze_audio(original_audio)
        
        # 步骤3: 详细音频质量分析
        print("步骤3: 生成详细音频质量分析报告...")
        detailed_analysis = enhancer.generate_detailed_analysis_report(original_audio, basic_analysis)
        
        print("✅ Task 7.1 音频质量分析完成!")
        print(f"📊 分析结果:")
        print(f"  - 时长: {detailed_analysis['file_info']['duration_seconds']:.1f}秒")
        print(f"  - 估算SNR: {detailed_analysis['basic_metrics']['estimated_snr_db']:.1f}dB")
        print(f"  - 动态范围: {detailed_analysis['basic_metrics']['dynamic_range_db']:.1f}dB")
        print(f"  - 整体质量: {detailed_analysis['quality_assessment']['overall_quality']}")
        print(f"📁 报告文件已保存到 audio_workspace/ 目录")
        
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保所有依赖库已安装: librosa, matplotlib, numpy, soundfile")
except Exception as e:
    print(f"❌ 运行错误: {e}")
    import traceback
    traceback.print_exc()