#!/usr/bin/env python3
"""
快速音视频合并器
使用流复制避免重新编码，添加实时进度显示
"""

import os
import sys
import json
import subprocess
import logging
import time
import re
from pathlib import Path
from datetime import datetime

class FastAudioVideoMerger:
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志系统"""
        log_file = f"fast_merger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_video_duration(self, video_file):
        """获取视频时长"""
        cmd = f'ffprobe -v quiet -print_format json -show_streams "{video_file}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            return 0
            
        info = json.loads(result.stdout)
        video_stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
        return float(video_stream.get('duration', 0))
        
    def run_ffmpeg_with_progress(self, command, description, expected_duration=None):
        """执行FFmpeg命令并显示实时进度"""
        self.logger.info(f"🔄 {description}...")
        self.logger.info(f"命令: {command}")
        
        start_time = time.time()
        
        try:
            # 使用Popen获取实时输出
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            current_time = 0
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output:
                    # 解析FFmpeg进度信息
                    time_match = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', output)
                    if time_match:
                        hours = int(time_match.group(1))
                        minutes = int(time_match.group(2))
                        seconds = float(time_match.group(3))
                        current_time = hours * 3600 + minutes * 60 + seconds
                        
                        if expected_duration and expected_duration > 0:
                            progress = min((current_time / expected_duration) * 100, 99.9)
                            elapsed = time.time() - start_time
                            if elapsed > 0:
                                speed = current_time / elapsed
                                eta = (expected_duration - current_time) / speed if speed > 0 else 0
                                self.logger.info(f"📊 {description} 进度: {progress:.1f}% "
                                               f"({current_time:.1f}s/{expected_duration:.1f}s) "
                                               f"速度: {speed:.1f}x ETA: {eta:.0f}s")
                        else:
                            elapsed = time.time() - start_time
                            self.logger.info(f"⏱️ {description} 进行中... "
                                           f"已处理: {current_time:.1f}s "
                                           f"用时: {elapsed:.1f}s")
            
            return_code = process.poll()
            total_time = time.time() - start_time
            
            if return_code == 0:
                self.logger.info(f"✅ {description} 完成! 总用时: {total_time:.1f}s")
                return True
            else:
                self.logger.error(f"❌ {description} 失败 (返回码: {return_code})")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ {description} 异常: {e}")
            return False
            
    def fast_merge_stream_copy(self, video_file, audio_file, output_file):
        """使用流复制快速合并（避免重新编码）"""
        self.logger.info("🚀 使用流复制快速合并...")
        
        # 获取视频时长用于进度计算
        video_duration = self.get_video_duration(video_file)
        self.logger.info(f"📊 视频时长: {video_duration:.1f}s")
        
        # 使用流复制命令（最快）
        cmd = f'''ffmpeg -i "{video_file}" -i "{audio_file}" \
                 -c:v copy \
                 -c:a aac -b:a 192k -ar 48000 -ac 2 \
                 -map 0:v:0 -map 1:a:0 \
                 -shortest \
                 -movflags +faststart \
                 "{output_file}" -y'''
        
        success = self.run_ffmpeg_with_progress(cmd, "快速流复制合并", video_duration)
        
        if success and Path(output_file).exists():
            file_size = Path(output_file).stat().st_size / (1024*1024)
            self.logger.info(f"✅ 快速合并完成: {file_size:.1f}MB")
            return True
        else:
            return False
            
    def high_quality_merge(self, video_file, audio_file, output_file):
        """高质量合并（重新编码，较慢但质量更好）"""
        self.logger.info("🎬 高质量重新编码合并...")
        
        video_duration = self.get_video_duration(video_file)
        self.logger.info(f"📊 视频时长: {video_duration:.1f}s")
        
        # 高质量编码命令
        cmd = f'''ffmpeg -i "{video_file}" -i "{audio_file}" \
                 -c:v libx264 -preset fast -crf 18 \
                 -pix_fmt yuv420p -profile:v high -level:v 5.1 \
                 -c:a aac -b:a 256k -ar 48000 -ac 2 \
                 -map 0:v:0 -map 1:a:0 \
                 -shortest \
                 -movflags +faststart \
                 "{output_file}" -y'''
        
        success = self.run_ffmpeg_with_progress(cmd, "高质量编码合并", video_duration)
        
        if success and Path(output_file).exists():
            file_size = Path(output_file).stat().st_size / (1024*1024)
            self.logger.info(f"✅ 高质量合并完成: {file_size:.1f}MB")
            return True
        else:
            return False
            
    def validate_output(self, output_file):
        """验证输出文件"""
        self.logger.info("✅ 验证输出文件...")
        
        if not Path(output_file).exists():
            self.logger.error("输出文件不存在")
            return False
            
        # 获取文件信息
        cmd = f'ffprobe -v quiet -print_format json -show_streams "{output_file}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            self.logger.error("无法读取输出文件信息")
            return False
            
        info = json.loads(result.stdout)
        
        # 检查视频和音频流
        video_streams = [s for s in info['streams'] if s['codec_type'] == 'video']
        audio_streams = [s for s in info['streams'] if s['codec_type'] == 'audio']
        
        if not video_streams:
            self.logger.error("输出文件缺少视频流")
            return False
            
        if not audio_streams:
            self.logger.error("输出文件缺少音频流")
            return False
            
        video_stream = video_streams[0]
        audio_stream = audio_streams[0]
        
        validation_info = {
            "video": {
                "codec": video_stream['codec_name'],
                "resolution": f"{video_stream['width']}x{video_stream['height']}",
                "duration": float(video_stream.get('duration', 0)),
                "fps": eval(video_stream['r_frame_rate']),
                "is_4k": video_stream['width'] == 3840 and video_stream['height'] == 2160
            },
            "audio": {
                "codec": audio_stream['codec_name'],
                "sample_rate": int(audio_stream['sample_rate']),
                "channels": int(audio_stream['channels']),
                "duration": float(audio_stream.get('duration', 0))
            },
            "file_size_mb": Path(output_file).stat().st_size / (1024*1024)
        }
        
        self.logger.info("验证结果:")
        self.logger.info(f"  - 视频: {validation_info['video']['resolution']} {validation_info['video']['codec']}")
        self.logger.info(f"  - 音频: {validation_info['audio']['sample_rate']}Hz {validation_info['audio']['codec']}")
        self.logger.info(f"  - 4K: {'是' if validation_info['video']['is_4k'] else '否'}")
        self.logger.info(f"  - 文件大小: {validation_info['file_size_mb']:.1f}MB")
        self.logger.info(f"  - 时长: {validation_info['video']['duration']:.1f}s")
        
        return validation_info
        
    def merge_with_options(self, video_file, audio_file, output_file, method="fast"):
        """根据选择的方法合并音视频"""
        start_time = datetime.now()
        
        try:
            self.logger.info("🎬 开始快速音视频合并")
            self.logger.info(f"📁 输入视频: {video_file}")
            self.logger.info(f"📁 输入音频: {audio_file}")
            self.logger.info(f"📁 输出文件: {output_file}")
            self.logger.info(f"🔧 合并方法: {method}")
            
            # 检查输入文件
            if not Path(video_file).exists():
                raise FileNotFoundError(f"视频文件不存在: {video_file}")
            if not Path(audio_file).exists():
                raise FileNotFoundError(f"音频文件不存在: {audio_file}")
                
            # 显示文件大小
            video_size = Path(video_file).stat().st_size / (1024*1024)
            audio_size = Path(audio_file).stat().st_size / (1024*1024)
            self.logger.info(f"📊 输入文件大小: 视频 {video_size:.1f}MB, 音频 {audio_size:.1f}MB")
            
            # 根据方法选择合并策略
            if method == "fast":
                success = self.fast_merge_stream_copy(video_file, audio_file, output_file)
            else:  # high_quality
                success = self.high_quality_merge(video_file, audio_file, output_file)
                
            if not success:
                raise RuntimeError("音视频合并失败")
                
            # 验证输出
            validation_info = self.validate_output(output_file)
            if not validation_info:
                raise RuntimeError("输出文件验证失败")
                
            # 生成报告
            processing_time = (datetime.now() - start_time).total_seconds()
            
            report = {
                "task": "Fast audio-video merging",
                "method": method,
                "input_video": video_file,
                "input_audio": audio_file,
                "output_file": output_file,
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "validation_info": validation_info,
                "processing_successful": True
            }
            
            # 保存报告
            with open("fast_merge_report.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"🎉 音视频合并完成! 总用时: {processing_time:.1f}秒")
            self.logger.info(f"✅ 输出文件: {output_file}")
            self.logger.info(f"📊 最终大小: {validation_info['file_size_mb']:.1f}MB")
            
            return output_file, report
            
        except Exception as e:
            self.logger.error(f"❌ 音视频合并失败: {str(e)}")
            raise

def main():
    if len(sys.argv) < 4:
        print("使用方法: python3 fast_audio_video_merger.py video_file.mp4 audio_file.wav output_file.mp4 [method]")
        print("方法选项:")
        print("  fast - 快速合并（流复制，推荐）")
        print("  quality - 高质量合并（重新编码）")
        sys.exit(1)
    
    video_file = sys.argv[1]
    audio_file = sys.argv[2]
    output_file = sys.argv[3]
    method = sys.argv[4] if len(sys.argv) > 4 else "fast"
    
    merger = FastAudioVideoMerger()
    final_output, report = merger.merge_with_options(video_file, audio_file, output_file, method)
    
    print(f"\n🎉 合并完成!")
    print(f"✅ 输出文件: {final_output}")
    print(f"📊 处理时间: {report['processing_time_seconds']:.1f} 秒")
    print(f"📊 文件大小: {report['validation_info']['file_size_mb']:.1f} MB")

if __name__ == "__main__":
    main()