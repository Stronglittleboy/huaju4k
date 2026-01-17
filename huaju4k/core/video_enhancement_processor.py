"""
主视频处理管道 - VideoEnhancementProcessor

这个模块实现了huaju4k的主要视频增强处理器，集成所有组件到主处理管道中，
添加处理编排和协调，实现输出验证和质量指标。

实现任务11.1的要求：
- 集成所有组件到主处理管道
- 添加处理编排和协调
- 实现输出验证和质量指标
- 需求: 3.5, 11.1, 11.2
"""

import os
import time
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# 视频处理相关导入
try:
    import cv2
    import ffmpeg
    HAS_VIDEO_LIBS = True
except ImportError:
    HAS_VIDEO_LIBS = False
    cv2 = None
    ffmpeg = None

from ..models.data_models import (
    VideoInfo, ProcessingStrategy, ProcessResult, 
    TileConfiguration, TheaterFeatures, AudioResult,
    StructureFeatures, EnhancementStrategy
)
from ..configs.simple_config_manager import SimpleConfigManager as ConfigManager
from .interfaces import VideoProcessor
from .video_analyzer import VideoAnalyzer
from .ai_model_manager import AIModelManager, StrategyDrivenModelManager
from .theater_audio_enhancer import TheaterAudioEnhancer
from .memory_manager import ConservativeMemoryManager
from .progress_tracker import MultiStageProgressTracker
from .tile_processor import TileProcessor
from .performance_optimizer import PerformanceOptimizer
from .master_grade_quality_validator import MasterGradeQualityValidator

logger = logging.getLogger(__name__)


class VideoEnhancementProcessor(VideoProcessor):
    """
    主视频增强处理器 - 集成所有组件的核心处理管道
    
    这个类是huaju4k系统的核心，负责协调所有处理组件，
    实现完整的视频增强工作流程，包括GPU加速支持。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化视频增强处理器
        
        Args:
            config_path: 配置文件路径（可选）
        """
        if not HAS_VIDEO_LIBS:
            raise ImportError(
                "视频处理库不可用。请安装: opencv-python, ffmpeg-python"
            )
        
        # 加载配置
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config(config_path)
        
        # 初始化核心组件
        from ..models.data_models import VideoConfig, PerformanceConfig
        
        # 创建配置对象
        video_config = VideoConfig()
        if 'video' in self.config:
            video_dict = self.config['video']
            if 'ai_model' in video_dict:
                video_config.ai_model = video_dict['ai_model']
            if 'quality_presets' in video_dict:
                video_config.quality_presets = video_dict['quality_presets']
        
        performance_config = PerformanceConfig()
        if 'performance' in self.config:
            perf_dict = self.config['performance']
            if 'use_gpu' in perf_dict:
                performance_config.use_gpu = perf_dict['use_gpu']
            if 'max_memory_usage' in perf_dict:
                performance_config.max_memory_usage = perf_dict['max_memory_usage']
            if 'num_workers' in perf_dict:
                performance_config.num_workers = perf_dict['num_workers']
        
        self.video_analyzer = VideoAnalyzer(
            config=video_config,
            performance_config=performance_config
        )
        
        self.ai_model_manager = StrategyDrivenModelManager(
            models_dir=self.config.get('models_dir', './models'),
            cache_size=self.config.get('model_cache_size', 2)
        )
        
        self.audio_enhancer = None
        try:
            self.audio_enhancer = TheaterAudioEnhancer(
                theater_preset=self.config.get('theater_preset', 'medium'),
                config=self.config.get('audio', {})
            )
        except ImportError as e:
            logger.warning(f"音频增强器初始化失败，将跳过音频处理: {e}")
            self.audio_enhancer = None
        
        self.memory_manager = ConservativeMemoryManager(
            safety_margin=self.config.get('memory_safety_margin', 0.7),
            temp_dir=self.config.get('temp_dir', None)
        )
        
        self.progress_tracker = MultiStageProgressTracker(
            update_interval=self.config.get('progress_update_interval', 0.5)
        )
        
        self.tile_processor = TileProcessor(
            memory_manager=self.memory_manager,
            progress_tracker=self.progress_tracker
        )
        
        # 性能优化器（任务11.3）
        self.performance_optimizer = None  # 将在处理开始时初始化
        
        # 处理状态
        self.current_task_id = None
        self.processing_stats = {}
        self._current_processing_strategy = None
        self._current_strategy_cache_path = None
        
        logger.info("VideoEnhancementProcessor 初始化完成")
    
    def process(self, input_path: str, output_path: str = None, 
                preset: str = "theater_medium", quality: str = "balanced") -> ProcessResult:
        """
        升级后的主处理管道 - 集成新的剧院增强组件
        
        实现需求3.5: 处理完成时验证输出质量并生成处理指标
        实现需求11.1: 集成所有组件到主处理管道
        实现需求11.2: 添加处理编排和协调
        实现Task 7: 集成新组件到主处理流程
        
        Args:
            input_path: 输入视频文件路径
            output_path: 输出视频文件路径（可选）
            preset: 剧院预设（theater_small, theater_medium, theater_large）
            quality: 质量级别（fast, balanced, high）
            
        Returns:
            ProcessResult 包含处理结果
        """
        start_time = time.time()
        
        try:
            logger.info(f"开始视频增强处理: {input_path}")
            
            # 验证输入文件
            if not os.path.exists(input_path):
                return ProcessResult(
                    success=False,
                    error=f"输入文件不存在: {input_path}"
                )
            
            # 生成输出路径
            if output_path is None:
                output_path = self._generate_output_path(input_path, preset, quality)
            
            # 确保输出目录存在
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 生成任务ID
            self.current_task_id = f"task_{int(time.time())}"
            
            # 设置进度跟踪阶段（升级版）
            self._setup_progress_stages()
            
            # 开始处理
            self.progress_tracker.start_processing()
            
            # Stage 1: 视频分析 + 舞台结构分析
            self.progress_tracker.start_stage("analyzing", "分析视频和舞台结构")
            video_info = self.analyze_video(input_path)
            structure_features = self._analyze_stage_structure(input_path)
            self.progress_tracker.complete_stage("analyzing", f"检测到 {video_info.resolution[0]}x{video_info.resolution[1]} 视频")
            
            # Stage 2: 策略计算 + 增强策略生成
            self.progress_tracker.start_stage("strategy", "生成增强策略")
            processing_strategy = self._calculate_processing_strategy(video_info, quality, preset)
            enhancement_strategy = self._generate_enhancement_strategy(structure_features)
            self._save_strategy_cache(enhancement_strategy)
            self.progress_tracker.complete_stage("strategy", f"使用 {processing_strategy.tile_size} 瓦片大小")
            
            # Stage 3: 加载AI模型（策略驱动）
            self.progress_tracker.start_stage("model_loading", "加载AI增强模型")
            self._setup_strategy_driven_model_manager(enhancement_strategy)
            model_loaded = self._load_ai_model(processing_strategy)
            if not model_loaded:
                return ProcessResult(
                    success=False,
                    error="AI模型加载失败"
                )
            self.progress_tracker.complete_stage("model_loading", f"已加载 {processing_strategy.ai_model} 模型")
            
            # Stage 4: 三阶段视频增强
            self.progress_tracker.start_stage("video_enhancement", "执行三阶段视频增强")
            enhanced_video_path = self._three_stage_video_enhancement(input_path, enhancement_strategy)
            if not enhanced_video_path:
                # 回退到原有视频增强
                logger.warning("三阶段增强失败，回退到原有处理流程")
                enhanced_video_path = self._legacy_enhance_video(input_path, processing_strategy)
                if not enhanced_video_path:
                    return ProcessResult(
                        success=False,
                        error="视频增强处理失败"
                    )
            self.progress_tracker.complete_stage("video_enhancement", "视频增强完成")
            
            # Stage 5: 母版级音频增强
            audio_result = None
            if video_info.has_audio:
                self.progress_tracker.start_stage("audio_enhancement", "执行母版级音频增强")
                audio_result = self._master_grade_audio_enhancement(input_path, enhancement_strategy)
                if audio_result and audio_result.success:
                    self.progress_tracker.complete_stage("audio_enhancement", "音频增强完成")
                else:
                    # 回退到原有音频增强
                    logger.warning("母版级音频增强失败，回退到原有音频处理")
                    audio_result = self._legacy_enhance_audio(input_path, processing_strategy, preset)
                    if audio_result and audio_result.success:
                        self.progress_tracker.complete_stage("audio_enhancement", "音频增强完成（回退模式）")
                    else:
                        self.progress_tracker.skip_stage("audio_enhancement", "音频增强跳过或失败")
            else:
                self.progress_tracker.skip_stage("audio_enhancement", "无音频轨道")
            
            # Stage 6: 最终合成
            self.progress_tracker.start_stage("finalizing", "合成最终输出")
            final_result = self._finalize_output(
                enhanced_video_path, 
                audio_result.output_path if audio_result and audio_result.success else None,
                output_path,
                video_info
            )
            
            if not final_result:
                return ProcessResult(
                    success=False,
                    error="最终输出合成失败"
                )
            
            self.progress_tracker.complete_stage("finalizing", "输出文件已生成")
            
            # Stage 7: 质量验证
            self.progress_tracker.start_stage("validation", "验证输出质量")
            quality_metrics = self._validate_output_quality(input_path, output_path, video_info, enhancement_strategy)
            
            # 生成性能报告（任务11.3）
            performance_report = self.performance_optimizer.get_performance_report() if self.performance_optimizer else {}
            
            self.progress_tracker.complete_stage("validation", "质量验证完成")
            
            # 完成处理
            processing_time = time.time() - start_time
            self.progress_tracker.finish_processing(success=True)
            
            # 清理资源
            self._cleanup_processing_resources()
            
            logger.info(f"视频增强处理完成: {output_path} ({processing_time:.1f}秒)")
            
            return ProcessResult(
                success=True,
                output_path=output_path,
                processing_time=processing_time,
                quality_metrics=quality_metrics,
                performance_report=performance_report,  # 添加性能报告
                memory_peak_mb=self.memory_manager.get_available_memory().get('system', 0),
                frames_processed=video_info.total_frames
            )
            
        except Exception as e:
            logger.error(f"视频增强处理失败: {e}")
            self.progress_tracker.finish_processing(success=False)
            self._cleanup_processing_resources()
            
            # 错误时回退到原有处理流程
            logger.warning(f"Enhanced processing failed: {e}, falling back to legacy processing")
            return self._legacy_process(input_path, output_path, preset, quality)
    
    def _analyze_stage_structure(self, input_path: str) -> StructureFeatures:
        """Stage 1扩展: 舞台结构分析"""
        try:
            from ..analysis.stage_structure_analyzer import StageStructureAnalyzer
            
            analyzer = StageStructureAnalyzer()
            return analyzer.analyze_structure(input_path)
        except ImportError as e:
            logger.warning(f"Stage structure analyzer not available: {e}")
            # 返回默认的结构特征
            return StructureFeatures(
                lighting_score=0.5,
                edge_density=0.3,
                motion_score=0.4,
                noise_score=0.2,
                highlight_ratio=0.1,
                is_static_camera=False,
                frame_diff_mean=0.05
            )
        except Exception as e:
            logger.error(f"Stage structure analysis failed: {e}")
            # 返回默认的结构特征
            return StructureFeatures(
                lighting_score=0.5,
                edge_density=0.3,
                motion_score=0.4,
                noise_score=0.2,
                highlight_ratio=0.1,
                is_static_camera=False,
                frame_diff_mean=0.05
            )
    
    def _generate_enhancement_strategy(self, features: StructureFeatures) -> EnhancementStrategy:
        """Stage 2扩展: 增强策略生成"""
        try:
            from ..strategy.enhancement_planner import EnhancementStrategyPlanner
            
            planner = EnhancementStrategyPlanner()
            return planner.generate_strategy(features)
        except ImportError as e:
            logger.warning(f"Enhancement strategy planner not available: {e}")
            # 返回默认策略
            from ..models.data_models import GANPolicy, TemporalConfig, MemoryConfig, AudioConfig
            return EnhancementStrategy(
                resolution_plan=["x2", "x2"],
                gan_policy=GANPolicy(global_allowed=True, strength="medium"),
                temporal_strategy=TemporalConfig(),
                memory_policy=MemoryConfig(),
                audio_strategy=AudioConfig()
            )
        except Exception as e:
            logger.error(f"Enhancement strategy generation failed: {e}")
            # 返回默认策略
            from ..models.data_models import GANPolicy, TemporalConfig, MemoryConfig, AudioConfig
            return EnhancementStrategy(
                resolution_plan=["x2", "x2"],
                gan_policy=GANPolicy(global_allowed=True, strength="medium"),
                temporal_strategy=TemporalConfig(),
                memory_policy=MemoryConfig(),
                audio_strategy=AudioConfig()
            )
    
    def _setup_strategy_driven_model_manager(self, strategy: EnhancementStrategy) -> None:
        """Stage 3扩展: 设置策略驱动的模型管理器"""
        try:
            # 升级现有的AI模型管理器
            if hasattr(self.ai_model_manager, 'set_strategy'):
                self.ai_model_manager.set_strategy(strategy)
                logger.info("AI model manager configured with strategy-driven mode")
            else:
                logger.warning("AI model manager does not support strategy-driven mode")
        except Exception as e:
            logger.warning(f"Failed to setup strategy-driven model manager: {e}")
    
    def _three_stage_video_enhancement(self, input_path: str, 
                                     strategy: EnhancementStrategy) -> Optional[str]:
        """Stage 4扩展: 三阶段视频增强"""
        try:
            from ..core.three_stage_enhancer import ThreeStageVideoEnhancer
            
            enhancer = ThreeStageVideoEnhancer(
                self.ai_model_manager, 
                self.progress_tracker
            )
            
            # 生成临时输出路径
            temp_output_path = self.memory_manager.create_temp_file(
                prefix="three_stage_enhanced_", suffix=".mp4"
            )
            
            # 调用三阶段增强器，传入正确的参数
            success = enhancer.enhance_video(input_path, temp_output_path, strategy)
            
            if success:
                return temp_output_path
            else:
                return None
            
        except ImportError as e:
            logger.warning(f"Three-stage enhancer not available: {e}, using legacy enhancement")
            return None
        except Exception as e:
            logger.error(f"Three-stage enhancement failed: {e}, falling back to legacy")
            return None
    
    def _master_grade_audio_enhancement(self, input_path: str, 
                                      strategy: EnhancementStrategy) -> Optional[AudioResult]:
        """Stage 5扩展: 母版级音频增强"""
        try:
            from ..audio.master_grade_enhancer import MasterGradeAudioEnhancer
            
            enhancer = MasterGradeAudioEnhancer()
            return enhancer.enhance_audio(input_path, strategy)
            
        except ImportError as e:
            logger.warning(f"Master grade audio enhancer not available: {e}, using legacy audio")
            return None
        except Exception as e:
            logger.error(f"Master grade audio enhancement failed: {e}, falling back to legacy")
            return None
    
    def _save_strategy_cache(self, strategy: EnhancementStrategy) -> None:
        """保存策略到缓存文件"""
        try:
            cache_path = self.memory_manager.create_temp_file(
                prefix="enhancement_strategy_", suffix=".json"
            )
            strategy.save_to_file(cache_path)
            self._current_strategy_cache_path = cache_path
            logger.info(f"Enhancement strategy cached to: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache strategy: {e}")
    
    def _load_strategy_cache(self, cache_path: str) -> Optional[EnhancementStrategy]:
        """从缓存加载策略"""
        try:
            import json
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 重建策略对象
            from ..models.data_models import EnhancementStrategy
            return EnhancementStrategy(**data)
            
        except Exception as e:
            logger.warning(f"Failed to load strategy cache: {e}")
            return None
    
    def _legacy_process(self, input_path: str, output_path: str, 
                       preset: str, quality: str) -> ProcessResult:
        """回退到原有处理流程"""
        logger.info("Using legacy processing pipeline")
        
        start_time = time.time()
        
        try:
            # 生成任务ID
            self.current_task_id = f"legacy_task_{int(time.time())}"
            
            # 设置原有进度跟踪阶段
            self._setup_legacy_progress_stages()
            
            # 开始处理
            self.progress_tracker.start_processing()
            
            # 阶段1: 分析视频
            self.progress_tracker.start_stage("analyzing", "分析视频属性和特征")
            video_info = self.analyze_video(input_path)
            self.progress_tracker.complete_stage("analyzing", f"检测到 {video_info.resolution[0]}x{video_info.resolution[1]} 视频")
            
            # 阶段2: 计算处理策略
            self.progress_tracker.start_stage("strategy", "计算最优处理策略")
            strategy = self._calculate_processing_strategy(video_info, quality, preset)
            self._current_processing_strategy = strategy
            
            # 初始化性能优化器（任务11.3）
            self.performance_optimizer = PerformanceOptimizer(strategy)
            
            self.progress_tracker.complete_stage("strategy", f"使用 {strategy.tile_size} 瓦片大小")
            
            # 阶段3: 加载AI模型
            self.progress_tracker.start_stage("model_loading", "加载AI增强模型")
            model_loaded = self._load_ai_model(strategy)
            if not model_loaded:
                return ProcessResult(
                    success=False,
                    error="AI模型加载失败"
                )
            self.progress_tracker.complete_stage("model_loading", f"已加载 {strategy.ai_model} 模型")
            
            # 阶段4: 视频增强处理
            self.progress_tracker.start_stage("video_enhancement", "执行视频增强")
            enhanced_video_path = self._legacy_enhance_video(input_path, strategy)
            if not enhanced_video_path:
                return ProcessResult(
                    success=False,
                    error="视频增强处理失败"
                )
            self.progress_tracker.complete_stage("video_enhancement", "视频增强完成")
            
            # 阶段5: 音频增强处理
            audio_result = None
            if video_info.has_audio and self.audio_enhancer is not None:
                self.progress_tracker.start_stage("audio_enhancement", "执行音频增强")
                audio_result = self._legacy_enhance_audio(input_path, strategy, preset)
                if audio_result and audio_result.success:
                    self.progress_tracker.complete_stage("audio_enhancement", "音频增强完成")
                else:
                    self.progress_tracker.skip_stage("audio_enhancement", "音频增强跳过或失败")
            else:
                skip_reason = "无音频轨道" if not video_info.has_audio else "音频增强器不可用"
                self.progress_tracker.skip_stage("audio_enhancement", skip_reason)
            
            # 阶段6: 最终合成
            self.progress_tracker.start_stage("finalizing", "合成最终输出")
            final_result = self._finalize_output(
                enhanced_video_path, 
                audio_result.output_path if audio_result and audio_result.success else None,
                output_path,
                video_info
            )
            
            if not final_result:
                return ProcessResult(
                    success=False,
                    error="最终输出合成失败"
                )
            
            self.progress_tracker.complete_stage("finalizing", "输出文件已生成")
            
            # 阶段7: 质量验证
            self.progress_tracker.start_stage("validation", "验证输出质量")
            quality_metrics = self._validate_output_quality(input_path, output_path, video_info, None)
            
            # 生成性能报告（任务11.3）
            performance_report = self.performance_optimizer.get_performance_report() if self.performance_optimizer else {}
            
            self.progress_tracker.complete_stage("validation", "质量验证完成")
            
            # 完成处理
            processing_time = time.time() - start_time
            self.progress_tracker.finish_processing(success=True)
            
            # 清理资源
            self._cleanup_processing_resources()
            
            logger.info(f"Legacy视频增强处理完成: {output_path} ({processing_time:.1f}秒)")
            
            return ProcessResult(
                success=True,
                output_path=output_path,
                processing_time=processing_time,
                quality_metrics=quality_metrics,
                performance_report=performance_report,
                memory_peak_mb=self.memory_manager.get_available_memory().get('system', 0),
                frames_processed=video_info.total_frames
            )
            
        except Exception as e:
            logger.error(f"Legacy处理失败: {e}")
            self.progress_tracker.finish_processing(success=False)
            self._cleanup_processing_resources()
            
            return ProcessResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _setup_legacy_progress_stages(self) -> None:
        """设置原有的进度跟踪阶段"""
        stages = [
            ("analyzing", "分析视频", 1.0),
            ("strategy", "计算策略", 0.5),
            ("model_loading", "加载模型", 1.0),
            ("video_enhancement", "视频增强", 10.0),  # 主要处理阶段
            ("audio_enhancement", "音频增强", 3.0),
            ("finalizing", "最终合成", 2.0),
            ("validation", "质量验证", 1.0)
        ]
        
        for stage_name, display_name, weight in stages:
            self.progress_tracker.add_stage(stage_name, display_name, weight)
    
    def _legacy_enhance_video(self, input_path: str, strategy: ProcessingStrategy) -> Optional[str]:
        """回退到原有视频增强"""
        # 使用现有的视频增强逻辑
        return self._enhance_video(input_path, strategy)
    
    def _legacy_enhance_audio(self, input_path: str, strategy: ProcessingStrategy, preset: str) -> Optional[AudioResult]:
        """回退到原有音频增强"""
        if self.audio_enhancer:
            # 使用现有的TheaterAudioEnhancer
            theater_preset = preset.replace("theater_", "") if preset.startswith("theater_") else "medium"
            return self._enhance_audio(input_path, strategy, preset)
        return None
    
    def analyze_video(self, input_path: str) -> VideoInfo:
        """
        分析视频属性并确定处理策略
        
        实现需求3.1: 处理开始时分析输入视频属性并确定最优处理策略
        
        Args:
            input_path: 输入视频文件路径
            
        Returns:
            VideoInfo 包含视频特征信息
        """
        try:
            return self.video_analyzer.analyze_video(input_path)
        except Exception as e:
            logger.error(f"视频分析失败: {e}")
            raise RuntimeError(f"视频分析失败: {e}")
    
    def process_tiles(self, frames: List[np.ndarray], 
                     strategy: ProcessingStrategy) -> List[np.ndarray]:
        """
        使用自适应瓦片大小处理视频帧
        
        实现需求3.4: 处理大视频时实现智能瓦片处理以管理内存使用
        
        Args:
            frames: 视频帧列表（numpy数组）
            strategy: 处理策略配置
            
        Returns:
            处理后的帧列表
        """
        try:
            # 创建瓦片配置
            tile_config = TileConfiguration(
                tile_width=strategy.tile_size[0],
                tile_height=strategy.tile_size[1],
                overlap=strategy.overlap_pixels,
                batch_size=strategy.batch_size,
                memory_usage_mb=strategy.memory_limit_mb,
                processing_mode="gpu" if strategy.use_gpu else "cpu"
            )
            
            processed_frames = []
            total_frames = len(frames)
            
            for i, frame in enumerate(frames):
                # 更新进度
                progress = i / total_frames
                self.progress_tracker.update_stage_progress(
                    "video_enhancement", 
                    progress, 
                    f"处理帧 {i+1}/{total_frames}"
                )
                
                # 处理单帧
                processed_frame = self.tile_processor.process_image_tiles(
                    frame, 
                    tile_config,
                    progress_callback=lambda p: self.progress_tracker.update_stage_progress(
                        "video_enhancement", progress + (p * 0.8 / total_frames)
                    )
                )
                
                processed_frames.append(processed_frame)
            
            return processed_frames
            
        except Exception as e:
            logger.error(f"瓦片处理失败: {e}")
            raise RuntimeError(f"瓦片处理失败: {e}")
    
    def _setup_progress_stages(self) -> None:
        """设置增强后的进度跟踪阶段"""
        stages = [
            ("analyzing", "分析视频和舞台结构", 2.0),  # 扩展分析阶段
            ("strategy", "生成增强策略", 1.0),        # 新增策略阶段
            ("model_loading", "加载AI模型", 1.0),
            ("video_enhancement", "三阶段视频增强", 12.0),  # 扩展视频增强
            ("audio_enhancement", "母版级音频增强", 4.0),   # 扩展音频增强
            ("finalizing", "最终合成", 2.0),
            ("validation", "质量验证", 1.0)
        ]
        
        for stage_name, display_name, weight in stages:
            self.progress_tracker.add_stage(stage_name, display_name, weight)
        
        # 添加视频增强子阶段
        self.progress_tracker.add_substage("video_enhancement", "structure_sr", "结构重建", 3.0)
        self.progress_tracker.add_substage("video_enhancement", "gan_enhance", "GAN增强", 6.0)
        self.progress_tracker.add_substage("video_enhancement", "temporal_lock", "时序锁定", 3.0)
    
    def _generate_output_path(self, input_path: str, preset: str, quality: str) -> str:
        """生成输出文件路径"""
        input_path_obj = Path(input_path)
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        
        # 生成输出文件名
        output_name = f"{stem}_enhanced_{preset}_{quality}_4k{suffix}"
        output_dir = input_path_obj.parent / "enhanced"
        output_dir.mkdir(exist_ok=True)
        
        return str(output_dir / output_name)
    
    def _calculate_processing_strategy(self, video_info: VideoInfo, 
                                     quality: str, preset: str) -> ProcessingStrategy:
        """计算处理策略"""
        try:
            # 确定目标分辨率（4K）
            aspect_ratio = video_info.aspect_ratio
            if aspect_ratio >= 16/9 - 0.1:  # 16:9 或更宽
                target_resolution = (3840, 2160)
            else:  # 4:3 或其他比例
                target_resolution = (2880, 2160)
            
            # 使用视频分析器计算策略
            strategy = self.video_analyzer.calculate_processing_strategy(
                video_info, quality, target_resolution
            )
            
            # 根据预设调整参数
            strategy = self._adjust_strategy_for_preset(strategy, preset)
            
            return strategy
            
        except Exception as e:
            logger.error(f"处理策略计算失败: {e}")
            raise RuntimeError(f"处理策略计算失败: {e}")
    
    def _adjust_strategy_for_preset(self, strategy: ProcessingStrategy, preset: str) -> ProcessingStrategy:
        """根据剧院预设调整处理策略"""
        # 剧院预设调整因子
        preset_adjustments = {
            "theater_small": {"denoise_strength": 0.8, "batch_size_factor": 1.2},
            "theater_medium": {"denoise_strength": 1.0, "batch_size_factor": 1.0},
            "theater_large": {"denoise_strength": 1.2, "batch_size_factor": 0.8}
        }
        
        adjustments = preset_adjustments.get(preset, preset_adjustments["theater_medium"])
        
        # 应用调整
        strategy.denoise_strength *= adjustments["denoise_strength"]
        strategy.batch_size = max(1, int(strategy.batch_size * adjustments["batch_size_factor"]))
        
        return strategy
    
    def _load_ai_model(self, strategy: ProcessingStrategy) -> bool:
        """加载AI模型"""
        try:
            # 获取可用内存
            available_memory = self.memory_manager.get_available_memory()
            
            # 自动选择最佳模型
            model_name = self.ai_model_manager.auto_select_model(
                strategy.target_resolution,
                available_memory['system']
            )
            
            if not model_name:
                logger.error("无法选择合适的AI模型")
                return False
            
            # 加载模型
            success = self.ai_model_manager.load_model(model_name, strategy.use_gpu)
            if success:
                logger.info(f"成功加载AI模型: {model_name}")
                # 更新策略中的模型名称
                strategy.ai_model = model_name
                # 设置瓦片处理器的AI模型管理器
                self.tile_processor.set_ai_model_manager(self.ai_model_manager)
            
            return success
            
        except Exception as e:
            logger.error(f"AI模型加载失败: {e}")
            return False
    
    def _enhance_video(self, input_path: str, strategy: ProcessingStrategy) -> Optional[str]:
        """执行视频增强处理"""
        try:
            # 创建临时输出路径
            temp_video_path = self.memory_manager.create_temp_file(
                prefix="enhanced_video_", suffix=".mp4"
            )
            
            # 使用FFmpeg提取帧并处理
            success = self._process_video_with_ffmpeg(
                input_path, str(temp_video_path), strategy
            )
            
            if success:
                return str(temp_video_path)
            else:
                return None
                
        except Exception as e:
            logger.error(f"视频增强失败: {e}")
            return None
    
    def _process_video_with_ffmpeg(self, input_path: str, output_path: str, 
                                  strategy: ProcessingStrategy) -> bool:
        """使用FFmpeg处理视频"""
        try:
            # 打开输入视频
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {input_path}")
                return False
            
            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 设置输出视频编写器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                strategy.target_resolution
            )
            
            # 批处理帧
            frame_batch = []
            processed_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_batch.append(frame)
                
                # 当批次满了或到达最后一帧时处理
                if len(frame_batch) >= strategy.batch_size or processed_count + len(frame_batch) >= total_frames:
                    # 处理批次
                    processed_batch = self._process_frame_batch(frame_batch, strategy)
                    
                    # 写入处理后的帧
                    for processed_frame in processed_batch:
                        out.write(processed_frame)
                    
                    processed_count += len(frame_batch)
                    
                    # 更新进度
                    progress = processed_count / total_frames
                    self.progress_tracker.update_stage_progress(
                        "video_enhancement", 
                        progress, 
                        f"已处理 {processed_count}/{total_frames} 帧"
                    )
                    
                    # 清空批次
                    frame_batch = []
            
            # 释放资源
            cap.release()
            out.release()
            
            logger.info(f"视频处理完成: {processed_count} 帧")
            return True
            
        except Exception as e:
            logger.error(f"FFmpeg视频处理失败: {e}")
            return False
    
    def _process_frame_batch(self, frames: List[np.ndarray], 
                           strategy: ProcessingStrategy) -> List[np.ndarray]:
        """处理帧批次 - 使用性能优化器"""
        try:
            if self.performance_optimizer:
                # 使用性能优化器处理批次
                def process_single_frame(frame):
                    # 使用AI模型增强帧
                    enhanced_frame = self.ai_model_manager.predict(frame)
                    
                    # 调整到目标分辨率
                    if enhanced_frame.shape[:2] != strategy.target_resolution[::-1]:
                        enhanced_frame = cv2.resize(
                            enhanced_frame, 
                            strategy.target_resolution,
                            interpolation=cv2.INTER_LANCZOS4
                        )
                    
                    return enhanced_frame
                
                # 使用性能优化器的并行处理
                processed_frames = self.performance_optimizer.optimize_processing(
                    frames, 
                    process_single_frame,
                    progress_callback=lambda p, msg: self.progress_tracker.update_stage_progress(
                        "video_enhancement", p * 0.8, msg
                    )
                )
                
                # 应用动态优化
                self.performance_optimizer.apply_dynamic_optimization()
                
                return processed_frames
            else:
                # 回退到原始处理方式
                processed_frames = []
                
                for frame in frames:
                    # 使用AI模型增强帧
                    enhanced_frame = self.ai_model_manager.predict(frame)
                    
                    # 调整到目标分辨率
                    if enhanced_frame.shape[:2] != strategy.target_resolution[::-1]:
                        enhanced_frame = cv2.resize(
                            enhanced_frame, 
                            strategy.target_resolution,
                            interpolation=cv2.INTER_LANCZOS4
                        )
                    
                    processed_frames.append(enhanced_frame)
                
                return processed_frames
            
        except Exception as e:
            logger.error(f"帧批次处理失败: {e}")
            # 返回原始帧作为回退
            return [cv2.resize(f, strategy.target_resolution, interpolation=cv2.INTER_LANCZOS4) 
                   for f in frames]
    
    def _enhance_audio(self, input_path: str, strategy: ProcessingStrategy, 
                      preset: str) -> Optional[AudioResult]:
        """增强音频处理"""
        try:
            if self.audio_enhancer is None:
                logger.warning("音频增强器不可用，跳过音频增强")
                return None
            
            # 提取音频到临时文件
            temp_audio_input = self.memory_manager.create_temp_file(
                prefix="audio_input_", suffix=".wav"
            )
            temp_audio_output = self.memory_manager.create_temp_file(
                prefix="audio_enhanced_", suffix=".wav"
            )
            
            # 使用FFmpeg提取音频
            extract_success = self._extract_audio_ffmpeg(input_path, str(temp_audio_input))
            if not extract_success:
                logger.warning("音频提取失败，跳过音频增强")
                return None
            
            # 确定剧院预设
            theater_preset = preset.replace("theater_", "") if preset.startswith("theater_") else "medium"
            
            # 执行音频增强
            audio_result = self.audio_enhancer.enhance(
                str(temp_audio_input),
                str(temp_audio_output),
                theater_preset
            )
            
            return audio_result
            
        except Exception as e:
            logger.error(f"音频增强失败: {e}")
            return None
    
    def _extract_audio_ffmpeg(self, video_path: str, audio_path: str) -> bool:
        """使用FFmpeg提取音频"""
        try:
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le', ac=2, ar=48000)
                .overwrite_output()
                .run(quiet=True)
            )
            return True
        except Exception as e:
            logger.error(f"FFmpeg音频提取失败: {e}")
            return False
    
    def _finalize_output(self, video_path: str, audio_path: Optional[str], 
                        output_path: str, video_info: VideoInfo) -> bool:
        """合成最终输出"""
        try:
            if audio_path and os.path.exists(audio_path):
                # 合并视频和音频
                return self._merge_video_audio_ffmpeg(video_path, audio_path, output_path)
            else:
                # 只有视频，直接复制
                import shutil
                shutil.copy2(video_path, output_path)
                return True
                
        except Exception as e:
            logger.error(f"最终输出合成失败: {e}")
            return False
    
    def _merge_video_audio_ffmpeg(self, video_path: str, audio_path: str, 
                                 output_path: str) -> bool:
        """使用FFmpeg合并视频和音频"""
        try:
            video_input = ffmpeg.input(video_path)
            audio_input = ffmpeg.input(audio_path)
            
            (
                ffmpeg
                .output(
                    video_input, audio_input, output_path,
                    vcodec='libx264',
                    acodec='aac',
                    preset='slow',
                    crf=18
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"FFmpeg视频音频合并失败: {e}")
            return False
    
    def _validate_output_quality(self, input_path: str, output_path: str, 
                               video_info: VideoInfo, 
                               enhancement_strategy: Optional[EnhancementStrategy] = None) -> Dict[str, float]:
        """
        升级后的质量验证 - 使用MasterGradeQualityValidator
        
        实现需求3.5: 处理完成时验证输出质量并生成处理指标
        实现需求11.5: 生成质量指标
        实现Task 8: 升级质量验证系统
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            video_info: 原始视频信息
            enhancement_strategy: 增强策略（用于高级验证）
            
        Returns:
            质量指标字典
        """
        try:
            # 使用新的MasterGradeQualityValidator进行高级验证
            if enhancement_strategy is not None:
                logger.info("Using master-grade quality validation")
                validator = MasterGradeQualityValidator()
                quality_report = validator.validate_master_quality(
                    input_path, output_path, enhancement_strategy
                )
                
                # 将QualityReport转换为字典格式
                metrics = quality_report.to_dict()
                
                # 展平嵌套字典以保持向后兼容
                flattened_metrics = {}
                for category, category_metrics in metrics.items():
                    if isinstance(category_metrics, dict):
                        for key, value in category_metrics.items():
                            flattened_metrics[f"{category}_{key}"] = value
                    else:
                        flattened_metrics[category] = category_metrics
                
                logger.info(f"Master-grade quality validation completed. Overall score: {quality_report.overall_score:.3f}")
                return flattened_metrics
            
            # 回退到原有的基础验证
            logger.info("Using legacy quality validation")
            metrics = {}
            
            # 验证输出文件存在且不为空
            if not os.path.exists(output_path):
                metrics['file_exists'] = 0.0
                return metrics
            
            output_size = os.path.getsize(output_path)
            if output_size == 0:
                metrics['file_exists'] = 0.0
                return metrics
            
            metrics['file_exists'] = 1.0
            metrics['output_file_size_mb'] = output_size / (1024 * 1024)
            
            # 分析输出视频属性
            try:
                output_info = self.video_analyzer.analyze_video(output_path)
                
                # 分辨率提升比率
                input_pixels = video_info.resolution[0] * video_info.resolution[1]
                output_pixels = output_info.resolution[0] * output_info.resolution[1]
                metrics['resolution_improvement_ratio'] = output_pixels / input_pixels
                
                # 帧率保持
                metrics['framerate_preserved'] = 1.0 if abs(output_info.framerate - video_info.framerate) < 0.1 else 0.0
                
                # 时长保持
                duration_diff = abs(output_info.duration - video_info.duration)
                metrics['duration_preserved'] = 1.0 if duration_diff < 1.0 else max(0.0, 1.0 - duration_diff / video_info.duration)
                
                # 音频保持
                metrics['audio_preserved'] = 1.0 if output_info.has_audio == video_info.has_audio else 0.0
                
                # 文件大小比率
                metrics['file_size_ratio'] = output_size / video_info.file_size
                
                logger.info(f"质量验证完成: 分辨率提升 {metrics['resolution_improvement_ratio']:.1f}x")
                
            except Exception as e:
                logger.warning(f"输出视频分析失败: {e}")
                metrics['analysis_failed'] = 1.0
            
            # 添加处理统计
            processing_stats = self.tile_processor.get_processing_stats()
            metrics.update(processing_stats)
            
            return metrics
            
        except Exception as e:
            logger.error(f"质量验证失败: {e}")
            return {'validation_failed': 1.0}
    
    def _cleanup_processing_resources(self) -> None:
        """清理处理资源"""
        try:
            # 清理内存管理器资源
            self.memory_manager.cleanup_resources()
            
            # 清理性能优化器资源（任务11.3）
            if self.performance_optimizer:
                self.performance_optimizer.cleanup()
                self.performance_optimizer = None
            
            # 卸载AI模型（如果需要）
            # self.ai_model_manager.clear_cache()  # 可选，保留缓存以供后续使用
            
            # 重置处理状态
            self.current_task_id = None
            self.processing_stats = {}
            
            logger.debug("处理资源清理完成")
            
        except Exception as e:
            logger.warning(f"资源清理时出现错误: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        获取处理统计信息
        
        Returns:
            包含处理统计的字典
        """
        stats = {
            'current_task_id': self.current_task_id,
            'memory_usage': self.memory_manager.get_available_memory(),
            'model_cache': self.ai_model_manager.get_memory_usage(),
            'progress_summary': self.progress_tracker.get_stage_summary()
        }
        
        # 添加瓦片处理统计
        tile_stats = self.tile_processor.get_processing_stats()
        stats.update(tile_stats)
        
        # 添加性能优化统计（任务11.3）
        if self.performance_optimizer:
            performance_stats = self.performance_optimizer.get_performance_report()
            stats['performance_optimization'] = performance_stats
        
        return stats
    
    def cancel_processing(self) -> bool:
        """
        取消当前处理任务
        
        Returns:
            取消是否成功
        """
        try:
            logger.info("正在取消处理任务...")
            
            # 停止进度跟踪
            self.progress_tracker.finish_processing(success=False)
            
            # 清理资源
            self._cleanup_processing_resources()
            
            logger.info("处理任务已取消")
            return True
            
        except Exception as e:
            logger.error(f"取消处理失败: {e}")
            return False