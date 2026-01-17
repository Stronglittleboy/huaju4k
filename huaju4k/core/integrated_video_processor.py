"""
集成视频处理器 - 基于FFmpeg主控的完整视频增强系统

这个模块实现了"FFmpeg主控路径"的架构升级，将原有的80%核心代码
与新的媒体控制层无缝集成，解决音频、编码兼容性和磁盘空间问题。

架构职责：
✅ 协调 FFmpegMediaController 和 StreamingVideoProcessor
✅ 管理完整的视频增强流程
✅ 处理音频增强集成
✅ 提供统一的处理接口

不负责：
❌ 具体的帧级AI处理（由StreamingVideoProcessor负责）
❌ 底层媒体编解码（由FFmpegMediaController负责）
❌ 音频DSP算法（由AudioEnhancer负责）
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import threading
from queue import Queue, Empty

from ..models.data_models import (
    VideoInfo, ProcessingStrategy, ProcessResult, 
    AudioConfig, TheaterFeatures
)
from ..media.ffmpeg_media_controller import FFmpegMediaController, MediaPipelineError
from .streaming_video_processor import StreamingVideoProcessor
from .theater_audio_enhancer import TheaterAudioEnhancer
from .progress_tracker import MultiStageProgressTracker
from .interfaces import VideoProcessor

logger = logging.getLogger(__name__)


class IntegratedVideoProcessor(VideoProcessor):
    """
    集成视频处理器 - FFmpeg主控架构
    
    这是huaju4k v2架构的核心组件，实现了：
    - FFmpeg专业媒体管道
    - 流式AI增强处理
    - 剧院音频优化
    - 统一的错误处理和进度跟踪
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化集成视频处理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.media_controller = FFmpegMediaController(
            temp_dir=self.config.get('temp_dir')
        )
        self.streaming_processor = StreamingVideoProcessor(
            config_path=self.config.get('config_path')
        )
        self.audio_enhancer = TheaterAudioEnhancer()
        self.progress_tracker = MultiStageProgressTracker()
        
        # 处理状态
        self.current_video_info: Optional[VideoInfo] = None
        self.processing_active = False
        
        logger.info("IntegratedVideoProcessor 初始化完成")
    
    def analyze_video(self, input_path: str) -> VideoInfo:
        """
        分析视频属性和特征
        
        Args:
            input_path: 输入视频文件路径
            
        Returns:
            VideoInfo对象，包含完整视频特征
        """
        try:
            # 使用FFmpeg进行专业视频分析
            video_info = self.media_controller.analyze_input_video(input_path)
            self.current_video_info = video_info
            
            # 同时使用流式处理器进行深度分析（如果需要）
            try:
                enhanced_info = self.streaming_processor.analyze_video(input_path)
                # 可以在这里合并更详细的分析结果
            except Exception as e:
                logger.warning(f"流式处理器分析失败，使用基础分析: {e}")
            
            logger.info(f"视频分析完成: {video_info.resolution[0]}x{video_info.resolution[1]}, "
                       f"{video_info.duration:.1f}s, 音频: {video_info.has_audio}")
            
            return video_info
            
        except Exception as e:
            logger.error(f"视频分析失败: {e}")
            raise RuntimeError(f"无法分析输入视频: {e}")
    
    def process(self, input_path: str, output_path: str, 
                strategy: ProcessingStrategy) -> ProcessResult:
        """
        处理视频 - 主入口方法
        
        Args:
            input_path: 输入视频文件路径
            output_path: 输出视频文件路径
            strategy: 处理策略配置
            
        Returns:
            ProcessResult包含处理结果
        """
        logger.info(f"开始集成视频处理: {input_path} -> {output_path}")
        start_time = time.time()
        
        try:
            # 1. 视频分析阶段
            self.progress_tracker.update_stage_progress("analyzing", 0.0, "分析输入视频")
            if not self.current_video_info:
                self.current_video_info = self.analyze_video(input_path)
            self.progress_tracker.update_stage_progress("analyzing", 1.0, "视频分析完成")
            
            # 2. 策略验证和调整
            self.progress_tracker.update_stage_progress("preparing", 0.0, "准备处理策略")
            adjusted_strategy = self._adjust_strategy_for_video(strategy, self.current_video_info)
            self.progress_tracker.update_stage_progress("preparing", 1.0, "策略准备完成")
            
            # 3. 执行FFmpeg主控处理
            result = self._process_with_ffmpeg_pipeline(
                input_path, output_path, adjusted_strategy
            )
            
            # 4. 计算最终结果
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            if result.success:
                logger.info(f"视频处理成功完成，耗时: {processing_time:.2f}秒")
            else:
                logger.error(f"视频处理失败: {result.error}")
            
            return result
            
        except Exception as e:
            logger.error(f"集成视频处理出错: {e}")
            return ProcessResult(
                success=False,
                output_path=output_path,
                processing_time=time.time() - start_time,
                error=str(e)
            )
        finally:
            self.processing_active = False
    
    def _process_with_ffmpeg_pipeline(self, input_path: str, output_path: str, 
                                    strategy: ProcessingStrategy) -> ProcessResult:
        """
        使用FFmpeg管道进行处理
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            strategy: 处理策略
            
        Returns:
            ProcessResult: 处理结果
        """
        self.processing_active = True
        
        try:
            with self.media_controller:
                # 1. 启动解码器管道
                self.progress_tracker.update_stage_progress("decoding", 0.0, "启动视频解码")
                target_resolution = strategy.target_resolution
                decoder_process = self.media_controller.start_decoder_pipeline(
                    input_path, target_resolution
                )
                
                # 2. 启动编码器管道
                self.progress_tracker.update_stage_progress("encoding", 0.0, "启动视频编码")
                audio_mode = self._determine_audio_mode(strategy)
                audio_config = self._get_audio_config(strategy)
                
                encoder_process = self.media_controller.start_encoder_pipeline(
                    output_path, input_path, 
                    audio_mode=audio_mode,
                    video_codec="libx264",  # 使用标准H.264编码器
                    audio_config=audio_config
                )
                
                # 3. 设置帧处理管道
                self.progress_tracker.update_stage_progress("enhancing", 0.0, "开始AI增强处理")
                
                # 创建帧生成器和回调
                frame_generator = self.media_controller.read_raw_frames()
                
                def frame_callback(enhanced_frame):
                    """将增强帧写入编码器"""
                    success = self.media_controller.write_raw_frame(enhanced_frame)
                    if not success:
                        logger.warning("编码器管道已关闭")
                
                # 4. 执行流式帧处理
                success = self.streaming_processor.process_raw_frame_stream(
                    frame_generator, frame_callback
                )
                
                if not success:
                    raise MediaPipelineError("流式帧处理失败")
                
                # 5. 完成编码
                self.progress_tracker.update_stage_progress("finalizing", 0.0, "完成视频编码")
                encoding_success = self.media_controller.finalize_encoding()
                
                if not encoding_success:
                    raise MediaPipelineError("视频编码完成失败")
                
                self.progress_tracker.update_stage_progress("finalizing", 1.0, "处理完成")
                
                # 6. 验证输出文件
                output_file = Path(output_path)
                if not output_file.exists() or output_file.stat().st_size == 0:
                    raise MediaPipelineError("输出文件无效或为空")
                
                return ProcessResult(
                    success=True,
                    output_path=output_path,
                    frames_processed=self.streaming_processor.current_frame_index,
                    quality_metrics=self._calculate_quality_metrics(input_path, output_path)
                )
                
        except Exception as e:
            logger.error(f"FFmpeg管道处理失败: {e}")
            return ProcessResult(
                success=False,
                output_path=output_path,
                error=str(e)
            )
    
    def _adjust_strategy_for_video(self, strategy: ProcessingStrategy, 
                                 video_info: VideoInfo) -> ProcessingStrategy:
        """
        根据视频特征调整处理策略
        
        Args:
            strategy: 原始策略
            video_info: 视频信息
            
        Returns:
            ProcessingStrategy: 调整后的策略
        """
        # 创建策略副本
        adjusted = ProcessingStrategy(
            tile_size=strategy.tile_size,
            overlap_pixels=strategy.overlap_pixels,
            batch_size=strategy.batch_size,
            use_gpu=strategy.use_gpu,
            ai_model=strategy.ai_model,
            quality_preset=strategy.quality_preset,
            memory_limit_mb=strategy.memory_limit_mb,
            target_resolution=strategy.target_resolution,
            denoise_strength=strategy.denoise_strength
        )
        
        # 根据输入分辨率调整目标分辨率
        if adjusted.target_resolution == (0, 0):
            # 默认2倍放大（而不是4倍）
            width, height = video_info.resolution
            adjusted.target_resolution = (width * 2, height * 2)  # 1080p -> 4K
        
        # 根据视频长度调整内存策略
        if video_info.duration > 3600:  # 超过1小时的视频
            adjusted.memory_limit_mb = min(adjusted.memory_limit_mb, 4096)
            adjusted.batch_size = 1
        
        # 根据GPU可用性调整处理模式
        if not strategy.use_gpu:
            adjusted.ai_model = "opencv_bicubic"  # 回退到CPU处理
        
        logger.info(f"策略调整完成: 目标分辨率 {adjusted.target_resolution}, "
                   f"内存限制 {adjusted.memory_limit_mb}MB")
        
        return adjusted
    
    def _determine_audio_mode(self, strategy: ProcessingStrategy) -> str:
        """
        确定音频处理模式
        
        Args:
            strategy: 处理策略
            
        Returns:
            str: 音频模式 ("copy" 或 "theater_enhanced")
        """
        # 检查是否启用了音频增强
        if hasattr(strategy, 'audio_enhancement') and strategy.audio_enhancement:
            return "theater_enhanced"
        
        # 检查配置中的音频设置
        audio_config = self.config.get('audio', {})
        if audio_config.get('theater_enhancement_enabled', False):
            return "theater_enhanced"
        
        # 默认直接复制音频
        return "copy"
    
    def _get_audio_config(self, strategy: ProcessingStrategy) -> Optional[AudioConfig]:
        """
        获取音频配置
        
        Args:
            strategy: 处理策略
            
        Returns:
            Optional[AudioConfig]: 音频配置对象
        """
        if self._determine_audio_mode(strategy) == "theater_enhanced":
            # 创建剧院音频配置
            return AudioConfig(
                theater_presets={
                    "medium": {
                        "reverb_reduction": 0.6,
                        "dialogue_boost": 4.0,
                        "noise_reduction": 0.5
                    }
                },
                dialogue_enhancement=0.7,
                source_separation_enabled=True
            )
        
        return None
    
    def _calculate_quality_metrics(self, input_path: str, output_path: str) -> Dict[str, float]:
        """
        计算质量指标
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            
        Returns:
            Dict[str, float]: 质量指标字典
        """
        try:
            input_size = Path(input_path).stat().st_size
            output_size = Path(output_path).stat().st_size
            
            return {
                "file_size_ratio": output_size / input_size if input_size > 0 else 0,
                "compression_efficiency": 1.0 - (output_size / (input_size * 16)) if input_size > 0 else 0,
                "processing_success_rate": 1.0  # 如果到这里说明处理成功
            }
        except Exception as e:
            logger.warning(f"质量指标计算失败: {e}")
            return {"processing_success_rate": 1.0}
    
    def process_tiles(self, frames, strategy: ProcessingStrategy):
        """
        瓦片处理方法（兼容接口）
        
        注意：集成处理器使用流式处理，不需要瓦片方法
        """
        logger.warning("IntegratedVideoProcessor 使用流式处理，不支持瓦片方法")
        return frames
    
    def get_processing_progress(self) -> Dict[str, Any]:
        """
        获取处理进度信息
        
        Returns:
            Dict[str, Any]: 进度信息
        """
        if self.progress_tracker:
            return {
                "overall_progress": self.progress_tracker.get_overall_progress(),
                "current_stage": getattr(self.progress_tracker, 'current_stage', 'idle'),
                "eta_seconds": self.progress_tracker.calculate_eta(),
                "frames_processed": getattr(self.streaming_processor, 'current_frame_index', 0),
                "is_active": self.processing_active
            }
        
        return {
            "overall_progress": 0.0,
            "current_stage": "idle",
            "eta_seconds": 0.0,
            "frames_processed": 0,
            "is_active": False
        }
    
    def stop_processing(self) -> bool:
        """
        停止当前处理
        
        Returns:
            bool: 停止是否成功
        """
        try:
            self.processing_active = False
            
            # 清理媒体控制器
            if self.media_controller:
                self.media_controller.cleanup()
            
            logger.info("处理已停止")
            return True
            
        except Exception as e:
            logger.error(f"停止处理失败: {e}")
            return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_processing()


def create_integrated_processor(config: Optional[Dict[str, Any]] = None) -> IntegratedVideoProcessor:
    """
    创建集成视频处理器的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        IntegratedVideoProcessor: 配置好的处理器实例
    """
    # 检查FFmpeg可用性
    from ..media.ffmpeg_media_controller import check_ffmpeg_availability
    
    ffmpeg_status = check_ffmpeg_availability()
    if not ffmpeg_status["ffmpeg"] or not ffmpeg_status["ffprobe"]:
        raise RuntimeError(
            "FFmpeg 不可用。请安装 FFmpeg 并确保 ffmpeg 和 ffprobe 在 PATH 中。\n"
            "安装指南: https://ffmpeg.org/download.html"
        )
    
    logger.info("FFmpeg 可用性检查通过")
    logger.info(f"支持的编解码器: {ffmpeg_status['codecs']}")
    
    return IntegratedVideoProcessor(config)