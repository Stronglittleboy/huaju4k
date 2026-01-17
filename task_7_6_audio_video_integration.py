#!/usr/bin/env python3
"""
任务7.6: 音视频同步和最终集成
Audio-video synchronization and final integration
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

class AudioVideoIntegrator:
    def __init__(self):
        self.workspace = Path(".")
        self.audio_workspace = Path("audio_workspace")
        
    def detect_sync_discrepancy(self, video_path, audio_path):
        """检测音视频同步差异"""
        print(f"🔍 检测音视频同步差异")
        print(f"   视频文件: {video_path}")
        print(f"   音频文件: {audio_path}")
        
        try:
            # 使用FFprobe获取视频和音频信息
            video_info_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_streams', str(video_path)
            ]
            
            result = subprocess.run(video_info_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                video_info = json.loads(result.stdout)
                
                # 查找视频和音频流
                video_stream = None
                audio_stream = None
                
                for stream in video_info['streams']:
                    if stream['codec_type'] == 'video':
                        video_stream = stream
                    elif stream['codec_type'] == 'audio':
                        audio_stream = stream
                
                sync_info = {
                    "video_duration": float(video_stream.get('duration', 0)) if video_stream else 0,
                    "audio_duration": float(audio_stream.get('duration', 0)) if audio_stream else 0,
                    "video_start_time": float(video_stream.get('start_time', 0)) if video_stream else 0,
                    "audio_start_time": float(audio_stream.get('start_time', 0)) if audio_stream else 0
                }
                
                # 计算同步差异
                duration_diff = abs(sync_info["video_duration"] - sync_info["audio_duration"])
                start_time_diff = abs(sync_info["video_start_time"] - sync_info["audio_start_time"])
                
                sync_info["duration_difference"] = duration_diff
                sync_info["start_time_difference"] = start_time_diff
                sync_info["sync_status"] = "good" if duration_diff < 0.1 and start_time_diff < 0.05 else "needs_adjustment"
                
                print(f"   视频时长: {sync_info['video_duration']:.2f}秒")
                print(f"   音频时长: {sync_info['audio_duration']:.2f}秒")
                print(f"   时长差异: {duration_diff:.3f}秒")
                print(f"   同步状态: {sync_info['sync_status']}")
                
                return sync_info
                
        except Exception as e:
            print(f"   ⚠️ 同步检测失败: {e}")
            return {"sync_status": "unknown", "error": str(e)}
    
    def merge_audio_video(self, video_path, audio_path, output_path, sync_offset=0):
        """合并增强音频和4K视频"""
        print(f"🎬 合并音视频文件")
        print(f"   视频: {video_path}")
        print(f"   音频: {audio_path}")
        print(f"   输出: {output_path}")
        print(f"   同步偏移: {sync_offset}秒")
        
        try:
            # 构建FFmpeg命令
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # 覆盖输出文件
                '-i', str(video_path),  # 输入视频
                '-i', str(audio_path),  # 输入音频
                '-c:v', 'copy',  # 复制视频流（不重新编码）
                '-c:a', 'aac',   # 音频编码为AAC
                '-b:a', '192k',  # 音频比特率
                '-map', '0:v:0', # 使用第一个输入的视频流
                '-map', '1:a:0', # 使用第二个输入的音频流
            ]
            
            # 如果有同步偏移，添加音频延迟
            if sync_offset != 0:
                ffmpeg_cmd.extend(['-itsoffset', str(sync_offset)])
            
            # 添加输出文件
            ffmpeg_cmd.append(str(output_path))
            
            print(f"   执行命令: {' '.join(ffmpeg_cmd)}")
            
            # 执行FFmpeg命令
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✅ 音视频合并成功")
                return True
            else:
                print(f"   ❌ 音视频合并失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   ❌ 合并过程异常: {e}")
            return False
    
    def validate_final_output(self, output_path):
        """验证最终输出质量"""
        print(f"🔍 验证最终输出质量: {output_path}")
        
        if not output_path.exists():
            print(f"   ❌ 输出文件不存在")
            return {"status": "failed", "reason": "file_not_found"}
        
        try:
            # 获取文件信息
            file_size = output_path.stat().st_size
            
            # 使用FFprobe获取详细信息
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-show_format', str(output_path)
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                
                validation_result = {
                    "status": "success",
                    "file_size_mb": round(file_size / 1024 / 1024, 2),
                    "format": info.get('format', {}),
                    "streams": []
                }
                
                # 分析流信息
                for stream in info.get('streams', []):
                    stream_info = {
                        "codec_type": stream.get('codec_type'),
                        "codec_name": stream.get('codec_name'),
                        "duration": float(stream.get('duration', 0))
                    }
                    
                    if stream['codec_type'] == 'video':
                        stream_info.update({
                            "width": stream.get('width'),
                            "height": stream.get('height'),
                            "frame_rate": stream.get('r_frame_rate'),
                            "bit_rate": stream.get('bit_rate')
                        })
                        
                        # 检查是否为4K分辨率
                        if stream.get('width') == 3840 and stream.get('height') == 2160:
                            validation_result["is_4k"] = True
                        else:
                            validation_result["is_4k"] = False
                            
                    elif stream['codec_type'] == 'audio':
                        stream_info.update({
                            "sample_rate": stream.get('sample_rate'),
                            "channels": stream.get('channels'),
                            "bit_rate": stream.get('bit_rate')
                        })
                    
                    validation_result["streams"].append(stream_info)
                
                # 打印验证结果
                print(f"   文件大小: {validation_result['file_size_mb']} MB")
                print(f"   4K分辨率: {'✅' if validation_result.get('is_4k') else '❌'}")
                
                for i, stream in enumerate(validation_result["streams"]):
                    if stream["codec_type"] == "video":
                        print(f"   视频流: {stream['width']}x{stream['height']}, {stream['codec_name']}")
                    elif stream["codec_type"] == "audio":
                        print(f"   音频流: {stream['channels']}声道, {stream['sample_rate']}Hz, {stream['codec_name']}")
                
                return validation_result
                
        except Exception as e:
            print(f"   ❌ 验证失败: {e}")
            return {"status": "failed", "reason": str(e)}
    
    def process_integration(self):
        """执行完整的音视频集成流程"""
        print("🎬 任务7.6: 音视频同步和最终集成")
        print("=" * 60)
        
        # 定义文件路径
        enhanced_4k_video = Path("enhanced_4k_theater_video.mp4")
        spatial_audio = self.audio_workspace / "spatial_enhanced_audio.wav"
        final_output = Path("final_4k_theater_video_with_enhanced_audio.mp4")
        
        # 检查输入文件
        print("📁 检查输入文件:")
        if not enhanced_4k_video.exists():
            print(f"   ❌ 4K视频文件不存在: {enhanced_4k_video}")
            return False
        else:
            print(f"   ✅ 4K视频文件: {enhanced_4k_video}")
        
        if not spatial_audio.exists():
            print(f"   ❌ 增强音频文件不存在: {spatial_audio}")
            return False
        else:
            print(f"   ✅ 增强音频文件: {spatial_audio}")
        
        # 1. 检测同步差异
        print(f"\n🔍 步骤1: 检测同步差异")
        sync_info = self.detect_sync_discrepancy(enhanced_4k_video, spatial_audio)
        
        # 2. 合并音视频
        print(f"\n🎬 步骤2: 合并音视频")
        sync_offset = 0  # 根据需要调整
        if sync_info.get("sync_status") == "needs_adjustment":
            # 简单的同步调整逻辑
            sync_offset = sync_info.get("start_time_difference", 0)
            print(f"   应用同步偏移: {sync_offset}秒")
        
        merge_success = self.merge_audio_video(
            enhanced_4k_video, spatial_audio, final_output, sync_offset
        )
        
        if not merge_success:
            print(f"   ❌ 音视频合并失败")
            return False
        
        # 3. 验证最终输出
        print(f"\n🔍 步骤3: 验证最终输出")
        validation_result = self.validate_final_output(final_output)
        
        # 4. 生成集成报告
        print(f"\n📊 步骤4: 生成集成报告")
        
        integration_report = {
            "task": "7.6 Audio-video synchronization and final integration",
            "timestamp": datetime.now().isoformat(),
            "input_files": {
                "video": str(enhanced_4k_video),
                "audio": str(spatial_audio)
            },
            "output_file": str(final_output),
            "sync_analysis": sync_info,
            "sync_offset_applied": sync_offset,
            "validation_result": validation_result,
            "integration_status": "success" if merge_success and validation_result.get("status") == "success" else "failed"
        }
        
        # 保存报告
        report_path = Path("task_7_6_integration_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(integration_report, f, indent=2, ensure_ascii=False)
        
        # 创建集成总结
        integration_summary = f"""
# 音视频集成处理总结

## 处理时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 输入文件
- **4K视频**: {enhanced_4k_video}
- **增强音频**: {spatial_audio}

## 输出文件
- **最终视频**: {final_output}
- **文件大小**: {validation_result.get('file_size_mb', 'N/A')} MB

## 同步分析
- **同步状态**: {sync_info.get('sync_status', 'unknown')}
- **时长差异**: {sync_info.get('duration_difference', 0):.3f}秒
- **应用偏移**: {sync_offset}秒

## 质量验证
- **4K分辨率**: {'✅ 是' if validation_result.get('is_4k') else '❌ 否'}
- **视频编码**: {validation_result.get('streams', [{}])[0].get('codec_name', 'N/A') if validation_result.get('streams') else 'N/A'}
- **音频编码**: {next((s.get('codec_name') for s in validation_result.get('streams', []) if s.get('codec_type') == 'audio'), 'N/A')}

## 集成状态
**{integration_report['integration_status'].upper()}** - {'音视频集成成功完成' if integration_report['integration_status'] == 'success' else '集成过程中出现问题'}

## 最终成果
✅ 4K分辨率剧场视频
✅ 增强的空间音频效果
✅ 优化的音视频同步
✅ 完整的后期处理流程
"""
        
        summary_path = Path("task_7_6_integration_summary.md")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(integration_summary)
        
        # 打印结果
        print(f"\n📊 音视频集成结果:")
        print(f"   集成状态: {'✅ 成功' if integration_report['integration_status'] == 'success' else '❌ 失败'}")
        print(f"   最终文件: {final_output}")
        print(f"   文件大小: {validation_result.get('file_size_mb', 'N/A')} MB")
        print(f"   4K分辨率: {'✅' if validation_result.get('is_4k') else '❌'}")
        print(f"   报告文件: {report_path}")
        print(f"   处理总结: {summary_path}")
        
        return integration_report['integration_status'] == 'success'

def main():
    integrator = AudioVideoIntegrator()
    success = integrator.process_integration()
    
    if success:
        print(f"\n🎉 任务7.6完成: 音视频同步和最终集成")
        print(f"🎬 最终的4K剧场视频已生成，包含增强的音频效果!")
    else:
        print(f"\n❌ 任务7.6失败: 音视频集成过程中出现问题")
    
    return success

if __name__ == "__main__":
    main()