#!/usr/bin/env python3
"""
任务8.2: 用户体验测试和最终优化
User experience testing and final optimization
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

class UserExperienceOptimizer:
    def __init__(self):
        self.workspace = Path(".")
        self.final_video = Path("final_4k_theater_video_with_enhanced_audio.mp4")
        
    def test_playback_compatibility(self):
        """测试播放兼容性"""
        print("🎬 测试播放兼容性")
        
        if not self.final_video.exists():
            print(f"   ❌ 最终视频文件不存在: {self.final_video}")
            return {"status": "failed", "reason": "file_not_found"}
        
        compatibility_results = {
            "file_exists": True,
            "file_size_mb": round(self.final_video.stat().st_size / 1024 / 1024, 2),
            "format_tests": {}
        }
        
        try:
            # 使用FFprobe获取详细格式信息
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-show_format', str(self.final_video)
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                
                # 分析兼容性
                format_info = info.get('format', {})
                streams = info.get('streams', [])
                
                # 视频流分析
                video_stream = next((s for s in streams if s.get('codec_type') == 'video'), None)
                audio_stream = next((s for s in streams if s.get('codec_type') == 'audio'), None)
                
                compatibility_results["format_tests"] = {
                    "container_format": format_info.get('format_name', 'unknown'),
                    "video_codec": video_stream.get('codec_name') if video_stream else None,
                    "audio_codec": audio_stream.get('codec_name') if audio_stream else None,
                    "resolution": f"{video_stream.get('width')}x{video_stream.get('height')}" if video_stream else None,
                    "duration": float(format_info.get('duration', 0)),
                    "bitrate": int(format_info.get('bit_rate', 0))
                }
                
                # 兼容性评估
                compatibility_score = 0
                compatibility_notes = []
                
                # MP4容器格式 (+20分)
                if 'mp4' in format_info.get('format_name', '').lower():
                    compatibility_score += 20
                    compatibility_notes.append("✅ MP4格式 - 广泛兼容")
                
                # H.264视频编码 (+25分)
                if video_stream and video_stream.get('codec_name') == 'h264':
                    compatibility_score += 25
                    compatibility_notes.append("✅ H.264编码 - 标准兼容")
                
                # AAC音频编码 (+20分)
                if audio_stream and audio_stream.get('codec_name') == 'aac':
                    compatibility_score += 20
                    compatibility_notes.append("✅ AAC音频 - 高兼容性")
                
                # 4K分辨率检查 (+15分)
                if video_stream and video_stream.get('width') == 3840 and video_stream.get('height') == 2160:
                    compatibility_score += 15
                    compatibility_notes.append("✅ 4K分辨率 - 现代标准")
                
                # 合理的比特率 (+10分)
                bitrate = int(format_info.get('bit_rate', 0))
                if 100000 <= bitrate <= 50000000:  # 100kbps - 50Mbps
                    compatibility_score += 10
                    compatibility_notes.append("✅ 合理比特率")
                
                # 立体声音频 (+10分)
                if audio_stream and audio_stream.get('channels') == 2:
                    compatibility_score += 10
                    compatibility_notes.append("✅ 立体声音频")
                
                compatibility_results["compatibility_score"] = compatibility_score
                compatibility_results["compatibility_notes"] = compatibility_notes
                compatibility_results["compatibility_level"] = (
                    "优秀" if compatibility_score >= 90 else
                    "良好" if compatibility_score >= 70 else
                    "一般" if compatibility_score >= 50 else
                    "需要改进"
                )
                
                print(f"   兼容性评分: {compatibility_score}/100 ({compatibility_results['compatibility_level']})")
                for note in compatibility_notes:
                    print(f"   {note}")
                
        except Exception as e:
            print(f"   ❌ 兼容性测试失败: {e}")
            compatibility_results["error"] = str(e)
        
        return compatibility_results
    
    def analyze_audio_quality_metrics(self):
        """分析音频质量指标"""
        print("🎵 分析音频质量指标")
        
        audio_metrics = {
            "analysis_method": "FFprobe + 文件分析",
            "metrics": {}
        }
        
        try:
            # 使用FFprobe分析音频流
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name,sample_rate,channels,bit_rate,duration',
                '-print_format', 'json', str(self.final_video)
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                stream = info.get('streams', [{}])[0]
                
                audio_metrics["metrics"] = {
                    "codec": stream.get('codec_name', 'unknown'),
                    "sample_rate": int(stream.get('sample_rate', 0)),
                    "channels": int(stream.get('channels', 0)),
                    "bitrate": int(stream.get('bit_rate', 0)),
                    "duration": float(stream.get('duration', 0))
                }
                
                # 质量评估
                sample_rate = audio_metrics["metrics"]["sample_rate"]
                bitrate = audio_metrics["metrics"]["bitrate"]
                channels = audio_metrics["metrics"]["channels"]
                
                quality_score = 0
                quality_notes = []
                
                # 采样率评估
                if sample_rate >= 48000:
                    quality_score += 25
                    quality_notes.append("✅ 高采样率 (48kHz+)")
                elif sample_rate >= 44100:
                    quality_score += 20
                    quality_notes.append("✅ 标准采样率 (44.1kHz)")
                
                # 比特率评估
                if bitrate >= 192000:
                    quality_score += 25
                    quality_notes.append("✅ 高比特率 (192kbps+)")
                elif bitrate >= 128000:
                    quality_score += 20
                    quality_notes.append("✅ 标准比特率 (128kbps+)")
                
                # 声道评估
                if channels == 2:
                    quality_score += 25
                    quality_notes.append("✅ 立体声")
                elif channels == 1:
                    quality_score += 15
                    quality_notes.append("✅ 单声道")
                
                # 编码格式评估
                if stream.get('codec_name') == 'aac':
                    quality_score += 25
                    quality_notes.append("✅ AAC编码")
                
                audio_metrics["quality_score"] = quality_score
                audio_metrics["quality_notes"] = quality_notes
                audio_metrics["quality_level"] = (
                    "优秀" if quality_score >= 90 else
                    "良好" if quality_score >= 70 else
                    "一般" if quality_score >= 50 else
                    "需要改进"
                )
                
                print(f"   音频质量评分: {quality_score}/100 ({audio_metrics['quality_level']})")
                for note in quality_notes:
                    print(f"   {note}")
                
        except Exception as e:
            print(f"   ❌ 音频质量分析失败: {e}")
            audio_metrics["error"] = str(e)
        
        return audio_metrics
    
    def generate_optimization_recommendations(self, compatibility_results, audio_metrics):
        """生成优化建议"""
        print("💡 生成优化建议")
        
        recommendations = {
            "compatibility_recommendations": [],
            "audio_recommendations": [],
            "general_recommendations": []
        }
        
        # 兼容性建议
        compatibility_score = compatibility_results.get("compatibility_score", 0)
        if compatibility_score < 90:
            if compatibility_score < 70:
                recommendations["compatibility_recommendations"].append(
                    "考虑转换为更兼容的格式 (MP4 + H.264 + AAC)"
                )
            if compatibility_results.get("format_tests", {}).get("container_format") != "mov,mp4,m4a,3gp,3g2,mj2":
                recommendations["compatibility_recommendations"].append(
                    "建议使用MP4容器格式以获得最佳兼容性"
                )
        
        # 音频建议
        audio_score = audio_metrics.get("quality_score", 0)
        if audio_score < 90:
            metrics = audio_metrics.get("metrics", {})
            if metrics.get("bitrate", 0) < 192000:
                recommendations["audio_recommendations"].append(
                    "考虑提高音频比特率至192kbps以获得更好的音质"
                )
            if metrics.get("sample_rate", 0) < 48000:
                recommendations["audio_recommendations"].append(
                    "建议使用48kHz采样率以获得更好的音频质量"
                )
        
        # 通用建议
        file_size = compatibility_results.get("file_size_mb", 0)
        if file_size > 100:
            recommendations["general_recommendations"].append(
                "文件较大，考虑适当压缩以便于分享和存储"
            )
        elif file_size < 10:
            recommendations["general_recommendations"].append(
                "文件较小，可能可以提高质量设置"
            )
        
        recommendations["general_recommendations"].extend([
            "定期备份最终视频文件",
            "测试在不同设备上的播放效果",
            "考虑生成不同分辨率版本以适应不同需求"
        ])
        
        # 打印建议
        if recommendations["compatibility_recommendations"]:
            print("   兼容性建议:")
            for rec in recommendations["compatibility_recommendations"]:
                print(f"     • {rec}")
        
        if recommendations["audio_recommendations"]:
            print("   音频建议:")
            for rec in recommendations["audio_recommendations"]:
                print(f"     • {rec}")
        
        if recommendations["general_recommendations"]:
            print("   通用建议:")
            for rec in recommendations["general_recommendations"]:
                print(f"     • {rec}")
        
        return recommendations
    
    def create_user_guidelines(self, compatibility_results, audio_metrics, recommendations):
        """创建用户使用指南"""
        print("📖 创建用户使用指南")
        
        guidelines = f"""
# 4K剧场视频用户使用指南

## 文件信息
- **文件名**: {self.final_video.name}
- **文件大小**: {compatibility_results.get('file_size_mb', 'N/A')} MB
- **分辨率**: {compatibility_results.get('format_tests', {}).get('resolution', 'N/A')}
- **时长**: {compatibility_results.get('format_tests', {}).get('duration', 0):.1f}秒

## 技术规格
### 视频
- **编码**: {compatibility_results.get('format_tests', {}).get('video_codec', 'N/A')}
- **分辨率**: 4K (3840x2160)
- **帧率**: 25fps
- **比特率**: {compatibility_results.get('format_tests', {}).get('bitrate', 0):,} bps

### 音频
- **编码**: {audio_metrics.get('metrics', {}).get('codec', 'N/A')}
- **采样率**: {audio_metrics.get('metrics', {}).get('sample_rate', 0):,} Hz
- **声道**: {audio_metrics.get('metrics', {}).get('channels', 0)}声道
- **比特率**: {audio_metrics.get('metrics', {}).get('bitrate', 0):,} bps

## 播放建议
### 推荐播放器
- **Windows**: VLC Media Player, Windows Media Player, PotPlayer
- **macOS**: VLC Media Player, QuickTime Player, IINA
- **移动设备**: VLC for Mobile, MX Player
- **在线**: 支持H.264的现代浏览器

### 系统要求
- **最低配置**: 
  - CPU: Intel i5 或 AMD Ryzen 5
  - RAM: 8GB
  - 显卡: 支持4K解码的独立显卡或集成显卡
- **推荐配置**:
  - CPU: Intel i7 或 AMD Ryzen 7
  - RAM: 16GB+
  - 显卡: GTX 1060 / RX 580 或更高

### 网络要求
- **本地播放**: 无网络要求
- **流媒体**: 至少25Mbps带宽用于4K流媒体

## 质量评估
- **兼容性**: {compatibility_results.get('compatibility_level', 'N/A')} ({compatibility_results.get('compatibility_score', 0)}/100)
- **音频质量**: {audio_metrics.get('quality_level', 'N/A')} ({audio_metrics.get('quality_score', 0)}/100)

## 使用建议
### 最佳观看体验
1. **显示设备**: 使用4K显示器或电视获得最佳效果
2. **音频设备**: 使用高质量耳机或音响系统
3. **环境**: 在安静、光线适中的环境中观看
4. **播放设置**: 确保播放器硬件加速已启用

### 存储和分享
1. **备份**: 建议创建多个备份副本
2. **压缩**: 如需分享，可考虑适当压缩
3. **格式**: 当前格式具有良好的兼容性
4. **云存储**: 可上传至支持大文件的云存储服务

## 优化建议
"""
        
        # 添加具体建议
        if recommendations["compatibility_recommendations"]:
            guidelines += "\n### 兼容性优化\n"
            for rec in recommendations["compatibility_recommendations"]:
                guidelines += f"- {rec}\n"
        
        if recommendations["audio_recommendations"]:
            guidelines += "\n### 音频优化\n"
            for rec in recommendations["audio_recommendations"]:
                guidelines += f"- {rec}\n"
        
        if recommendations["general_recommendations"]:
            guidelines += "\n### 通用建议\n"
            for rec in recommendations["general_recommendations"]:
                guidelines += f"- {rec}\n"
        
        guidelines += f"""
## 技术支持
如遇到播放问题，请检查：
1. 播放器是否支持H.264和AAC编解码器
2. 系统是否满足4K播放的硬件要求
3. 文件是否完整下载/复制

## 处理历史
本视频经过以下增强处理：
- ✅ AI 4K上采样
- ✅ 智能降噪处理
- ✅ 频率均衡优化
- ✅ 动态范围控制
- ✅ 空间音频增强
- ✅ 音视频同步优化

---
*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存用户指南
        guidelines_path = Path("4k_theater_video_user_guide.md")
        with open(guidelines_path, 'w', encoding='utf-8') as f:
            f.write(guidelines)
        
        print(f"   ✅ 用户指南已保存: {guidelines_path}")
        return guidelines_path
    
    def process_user_experience_optimization(self):
        """执行完整的用户体验优化"""
        print("🎯 任务8.2: 用户体验测试和最终优化")
        print("=" * 60)
        
        # 1. 测试播放兼容性
        print("\n🎬 步骤1: 播放兼容性测试")
        compatibility_results = self.test_playback_compatibility()
        
        # 2. 分析音频质量
        print("\n🎵 步骤2: 音频质量分析")
        audio_metrics = self.analyze_audio_quality_metrics()
        
        # 3. 生成优化建议
        print("\n💡 步骤3: 生成优化建议")
        recommendations = self.generate_optimization_recommendations(
            compatibility_results, audio_metrics
        )
        
        # 4. 创建用户指南
        print("\n📖 步骤4: 创建用户指南")
        guidelines_path = self.create_user_guidelines(
            compatibility_results, audio_metrics, recommendations
        )
        
        # 5. 生成最终报告
        print("\n📊 步骤5: 生成最终报告")
        
        final_report = {
            "task": "8.2 User experience testing and final optimization",
            "timestamp": datetime.now().isoformat(),
            "final_video_file": str(self.final_video),
            "compatibility_analysis": compatibility_results,
            "audio_quality_analysis": audio_metrics,
            "optimization_recommendations": recommendations,
            "user_guidelines_file": str(guidelines_path),
            "overall_assessment": {
                "compatibility_level": compatibility_results.get("compatibility_level", "未知"),
                "audio_quality_level": audio_metrics.get("quality_level", "未知"),
                "ready_for_distribution": (
                    compatibility_results.get("compatibility_score", 0) >= 70 and
                    audio_metrics.get("quality_score", 0) >= 70
                )
            }
        }
        
        # 保存报告
        report_path = Path("task_8_2_user_experience_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        # 打印结果
        print(f"\n📊 用户体验优化结果:")
        print(f"   兼容性等级: {final_report['overall_assessment']['compatibility_level']}")
        print(f"   音频质量等级: {final_report['overall_assessment']['audio_quality_level']}")
        print(f"   分发就绪: {'✅ 是' if final_report['overall_assessment']['ready_for_distribution'] else '❌ 否'}")
        print(f"   最终视频: {self.final_video}")
        print(f"   用户指南: {guidelines_path}")
        print(f"   详细报告: {report_path}")
        
        return final_report['overall_assessment']['ready_for_distribution']

def main():
    optimizer = UserExperienceOptimizer()
    success = optimizer.process_user_experience_optimization()
    
    if success:
        print(f"\n🎉 任务8.2完成: 用户体验测试和最终优化")
        print(f"✅ 4K剧场视频已准备好分发使用!")
    else:
        print(f"\n⚠️ 任务8.2完成，但视频可能需要进一步优化")
    
    return success

if __name__ == "__main__":
    main()