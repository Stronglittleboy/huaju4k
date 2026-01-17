"""
配置管理CLI - 配置和预设管理功能

实现任务12.1的配置管理要求：
- 实现配置验证和管理
- 添加预设创建和管理
- 创建配置文件操作
- 需求: 8.1, 8.2, 8.3, 8.5, 8.6
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import click

from ..configs.simple_config_manager import SimpleConfigManager
from ..models.data_models import VideoConfig, AudioConfig, PerformanceConfig

logger = logging.getLogger(__name__)


class ConfigCLI:
    """
    配置管理命令行界面
    
    实现需求8.1, 8.2, 8.3, 8.5, 8.6: 配置和预设管理
    """
    
    def __init__(self):
        """初始化配置CLI"""
        self.config_manager = SimpleConfigManager()
        self.presets_dir = Path.home() / ".huaju4k" / "presets"
        self.presets_dir.mkdir(parents=True, exist_ok=True)
    
    def list_presets(self) -> None:
        """列出所有可用预设"""
        click.echo("=" * 60)
        click.echo("可用预设列表")
        click.echo("=" * 60)
        
        # 内置预设
        click.echo("内置剧院预设:")
        builtin_presets = {
            "theater_small": "小型剧院 - 适合小空间，强化对话清晰度",
            "theater_medium": "中型剧院 - 平衡的音频增强效果",
            "theater_large": "大型剧院 - 适合大空间，保持自然音效"
        }
        
        for preset_name, description in builtin_presets.items():
            click.echo(f"  • {preset_name}: {description}")
        
        # 质量预设
        click.echo("\n质量级别预设:")
        quality_presets = {
            "fast": "快速处理 - 较低质量，处理速度快",
            "balanced": "平衡模式 - 质量与速度的平衡",
            "high": "高质量 - 最佳质量，处理时间较长"
        }
        
        for preset_name, description in quality_presets.items():
            click.echo(f"  • {preset_name}: {description}")
        
        # 自定义预设
        custom_presets = self._get_custom_presets()
        if custom_presets:
            click.echo("\n自定义预设:")
            for preset_name, preset_info in custom_presets.items():
                description = preset_info.get('description', '无描述')
                click.echo(f"  • {preset_name}: {description}")
        else:
            click.echo("\n自定义预设: 无")
        
        click.echo(f"\n预设存储位置: {self.presets_dir}")
    
    def validate_config(self, config_path: str) -> None:
        """验证配置文件"""
        click.echo(f"验证配置文件: {config_path}")
        
        try:
            # 加载配置
            config = self.config_manager.load_config(config_path)
            
            # 验证配置结构
            validation_results = self._validate_config_structure(config)
            
            if validation_results['valid']:
                click.echo("✅ 配置文件验证通过")
                
                # 显示配置摘要
                self._display_config_summary(config)
                
                # 显示验证详情
                if validation_results['warnings']:
                    click.echo("\n⚠️ 警告:")
                    for warning in validation_results['warnings']:
                        click.echo(f"  • {warning}")
                
            else:
                click.echo("❌ 配置文件验证失败")
                click.echo("\n错误:")
                for error in validation_results['errors']:
                    click.echo(f"  • {error}")
                
                if validation_results['warnings']:
                    click.echo("\n警告:")
                    for warning in validation_results['warnings']:
                        click.echo(f"  • {warning}")
        
        except Exception as e:
            click.echo(f"❌ 配置文件验证失败: {e}")
    
    def create_preset(self, preset_name: str) -> None:
        """创建新预设"""
        click.echo(f"创建新预设: {preset_name}")
        
        # 检查预设是否已存在
        preset_path = self.presets_dir / f"{preset_name}.yaml"
        if preset_path.exists():
            if not click.confirm(f"预设 '{preset_name}' 已存在，是否覆盖？"):
                click.echo("预设创建已取消")
                return
        
        try:
            # 交互式创建预设
            preset_config = self._interactive_preset_creation()
            
            # 保存预设
            with open(preset_path, 'w', encoding='utf-8') as f:
                yaml.dump(preset_config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            click.echo(f"✅ 预设已创建: {preset_path}")
            
            # 显示预设摘要
            self._display_preset_summary(preset_config)
            
        except Exception as e:
            click.echo(f"❌ 预设创建失败: {e}")
    
    def _get_custom_presets(self) -> Dict[str, Dict[str, Any]]:
        """获取自定义预设列表"""
        custom_presets = {}
        
        try:
            for preset_file in self.presets_dir.glob("*.yaml"):
                try:
                    with open(preset_file, 'r', encoding='utf-8') as f:
                        preset_config = yaml.safe_load(f)
                    
                    preset_name = preset_file.stem
                    custom_presets[preset_name] = preset_config
                    
                except Exception as e:
                    logger.warning(f"加载预设文件失败 {preset_file}: {e}")
        
        except Exception as e:
            logger.error(f"扫描预设目录失败: {e}")
        
        return custom_presets
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置文件结构"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # 验证必需的顶级键
            required_keys = ['video', 'audio', 'performance']
            for key in required_keys:
                if key not in config:
                    validation_result['errors'].append(f"缺少必需的配置节: {key}")
                    validation_result['valid'] = False
            
            # 验证视频配置
            if 'video' in config:
                video_errors = self._validate_video_config(config['video'])
                validation_result['errors'].extend(video_errors)
                if video_errors:
                    validation_result['valid'] = False
            
            # 验证音频配置
            if 'audio' in config:
                audio_errors = self._validate_audio_config(config['audio'])
                validation_result['errors'].extend(audio_errors)
                if audio_errors:
                    validation_result['valid'] = False
            
            # 验证性能配置
            if 'performance' in config:
                perf_errors, perf_warnings = self._validate_performance_config(config['performance'])
                validation_result['errors'].extend(perf_errors)
                validation_result['warnings'].extend(perf_warnings)
                if perf_errors:
                    validation_result['valid'] = False
            
            # 检查未知的顶级键
            known_keys = {'video', 'audio', 'performance', 'models_dir', 'temp_dir', 
                         'theater_preset', 'model_cache_size', 'memory_safety_margin',
                         'progress_update_interval'}
            unknown_keys = set(config.keys()) - known_keys
            if unknown_keys:
                validation_result['warnings'].append(f"未知的配置键: {', '.join(unknown_keys)}")
        
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"配置验证异常: {e}")
        
        return validation_result
    
    def _validate_video_config(self, video_config: Dict[str, Any]) -> List[str]:
        """验证视频配置"""
        errors = []
        
        # 验证AI模型
        if 'ai_model' in video_config:
            valid_models = ['real_esrgan', 'esrgan', 'waifu2x']
            if video_config['ai_model'] not in valid_models:
                errors.append(f"无效的AI模型: {video_config['ai_model']}")
        
        # 验证质量预设
        if 'quality_presets' in video_config:
            for preset_name, preset_config in video_config['quality_presets'].items():
                if not isinstance(preset_config, dict):
                    errors.append(f"质量预设 '{preset_name}' 必须是字典")
                    continue
                
                # 验证预设参数
                required_params = ['tile_size', 'batch_size', 'denoise_strength']
                for param in required_params:
                    if param not in preset_config:
                        errors.append(f"质量预设 '{preset_name}' 缺少参数: {param}")
        
        return errors
    
    def _validate_audio_config(self, audio_config: Dict[str, Any]) -> List[str]:
        """验证音频配置"""
        errors = []
        
        # 验证剧院预设
        if 'theater_presets' in audio_config:
            for preset_name, preset_config in audio_config['theater_presets'].items():
                if not isinstance(preset_config, dict):
                    errors.append(f"剧院预设 '{preset_name}' 必须是字典")
                    continue
                
                # 验证预设参数
                required_params = ['reverb_reduction', 'dialogue_boost', 'noise_reduction']
                for param in required_params:
                    if param not in preset_config:
                        errors.append(f"剧院预设 '{preset_name}' 缺少参数: {param}")
        
        # 验证采样率
        if 'sample_rate' in audio_config:
            valid_rates = [44100, 48000, 96000]
            if audio_config['sample_rate'] not in valid_rates:
                errors.append(f"无效的采样率: {audio_config['sample_rate']}")
        
        return errors
    
    def _validate_performance_config(self, perf_config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """验证性能配置"""
        errors = []
        warnings = []
        
        # 验证内存使用限制
        if 'max_memory_usage' in perf_config:
            max_mem = perf_config['max_memory_usage']
            if not isinstance(max_mem, (int, float)) or not 0.1 <= max_mem <= 0.9:
                errors.append("max_memory_usage 必须在 0.1 到 0.9 之间")
        
        # 验证工作线程数
        if 'num_workers' in perf_config:
            num_workers = perf_config['num_workers']
            if not isinstance(num_workers, int) or num_workers < 1:
                errors.append("num_workers 必须是大于0的整数")
            elif num_workers > 8:
                warnings.append("num_workers 过大可能导致性能下降")
        
        # 验证GPU设置
        if 'use_gpu' in perf_config and perf_config['use_gpu']:
            if 'gpu_id' in perf_config:
                gpu_id = perf_config['gpu_id']
                if not isinstance(gpu_id, int) or gpu_id < 0:
                    errors.append("gpu_id 必须是非负整数")
        
        return errors, warnings
    
    def _display_config_summary(self, config: Dict[str, Any]) -> None:
        """显示配置摘要"""
        click.echo("\n配置摘要:")
        
        # 视频配置
        if 'video' in config:
            video_config = config['video']
            click.echo(f"  视频:")
            click.echo(f"    AI模型: {video_config.get('ai_model', '默认')}")
            click.echo(f"    质量预设数量: {len(video_config.get('quality_presets', {}))}")
        
        # 音频配置
        if 'audio' in config:
            audio_config = config['audio']
            click.echo(f"  音频:")
            click.echo(f"    采样率: {audio_config.get('sample_rate', '默认')} Hz")
            click.echo(f"    剧院预设数量: {len(audio_config.get('theater_presets', {}))}")
        
        # 性能配置
        if 'performance' in config:
            perf_config = config['performance']
            click.echo(f"  性能:")
            click.echo(f"    GPU加速: {'启用' if perf_config.get('use_gpu', True) else '禁用'}")
            click.echo(f"    最大内存使用: {perf_config.get('max_memory_usage', 0.7) * 100:.0f}%")
            click.echo(f"    工作线程: {perf_config.get('num_workers', 2)}")
    
    def _interactive_preset_creation(self) -> Dict[str, Any]:
        """交互式创建预设"""
        click.echo("\n开始交互式预设创建...")
        
        # 基本信息
        description = click.prompt("预设描述", default="自定义预设")
        preset_type = click.prompt(
            "预设类型", 
            type=click.Choice(['theater', 'quality', 'custom']),
            default='custom'
        )
        
        preset_config = {
            'name': click.prompt("预设名称"),
            'description': description,
            'type': preset_type,
            'created_at': str(datetime.now()),
            'version': '1.0'
        }
        
        if preset_type == 'theater':
            # 剧院预设配置
            preset_config['audio'] = self._create_theater_audio_config()
        elif preset_type == 'quality':
            # 质量预设配置
            preset_config['video'] = self._create_quality_video_config()
        else:
            # 自定义预设配置
            if click.confirm("配置视频参数？", default=True):
                preset_config['video'] = self._create_quality_video_config()
            
            if click.confirm("配置音频参数？", default=True):
                preset_config['audio'] = self._create_theater_audio_config()
            
            if click.confirm("配置性能参数？", default=False):
                preset_config['performance'] = self._create_performance_config()
        
        return preset_config
    
    def _create_theater_audio_config(self) -> Dict[str, Any]:
        """创建剧院音频配置"""
        click.echo("\n配置剧院音频参数:")
        
        reverb_reduction = click.prompt(
            "混响减少强度 (0.0-1.0)", 
            type=float, 
            default=0.6
        )
        dialogue_boost = click.prompt(
            "对话增强 (dB)", 
            type=float, 
            default=4.0
        )
        noise_reduction = click.prompt(
            "噪声减少强度 (0.0-1.0)", 
            type=float, 
            default=0.5
        )
        
        return {
            'reverb_reduction': reverb_reduction,
            'dialogue_boost': dialogue_boost,
            'noise_reduction': noise_reduction
        }
    
    def _create_quality_video_config(self) -> Dict[str, Any]:
        """创建质量视频配置"""
        click.echo("\n配置视频质量参数:")
        
        tile_size = click.prompt(
            "瓦片大小 (像素)", 
            type=int, 
            default=768
        )
        batch_size = click.prompt(
            "批处理大小", 
            type=int, 
            default=2
        )
        denoise_strength = click.prompt(
            "降噪强度 (0.0-1.0)", 
            type=float, 
            default=0.7
        )
        
        return {
            'tile_size': tile_size,
            'batch_size': batch_size,
            'denoise_strength': denoise_strength
        }
    
    def _create_performance_config(self) -> Dict[str, Any]:
        """创建性能配置"""
        click.echo("\n配置性能参数:")
        
        use_gpu = click.confirm("启用GPU加速？", default=True)
        max_memory = click.prompt(
            "最大内存使用比例 (0.1-0.9)", 
            type=float, 
            default=0.7
        )
        num_workers = click.prompt(
            "工作线程数", 
            type=int, 
            default=2
        )
        
        config = {
            'use_gpu': use_gpu,
            'max_memory_usage': max_memory,
            'num_workers': num_workers
        }
        
        if use_gpu:
            gpu_id = click.prompt("GPU ID", type=int, default=0)
            config['gpu_id'] = gpu_id
        
        return config
    
    def _display_preset_summary(self, preset_config: Dict[str, Any]) -> None:
        """显示预设摘要"""
        click.echo("\n预设摘要:")
        click.echo(f"  名称: {preset_config.get('name', '未知')}")
        click.echo(f"  描述: {preset_config.get('description', '无')}")
        click.echo(f"  类型: {preset_config.get('type', '未知')}")
        
        if 'video' in preset_config:
            video_config = preset_config['video']
            click.echo(f"  视频配置:")
            for key, value in video_config.items():
                click.echo(f"    {key}: {value}")
        
        if 'audio' in preset_config:
            audio_config = preset_config['audio']
            click.echo(f"  音频配置:")
            for key, value in audio_config.items():
                click.echo(f"    {key}: {value}")
        
        if 'performance' in preset_config:
            perf_config = preset_config['performance']
            click.echo(f"  性能配置:")
            for key, value in perf_config.items():
                click.echo(f"    {key}: {value}")


# 导入datetime用于预设创建时间戳
from datetime import datetime