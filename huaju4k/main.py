"""
Main entry point for huaju4k video enhancement tool.

This module provides the command-line interface and application initialization.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import click

from .configs.config_manager import ConfigManager
from .utils.system_utils import get_system_info, check_dependencies
from .utils.validation_utils import validate_system_requirements
from .cli.system_check import system


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to configuration file')
@click.option('--log-level', default='INFO', 
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
              help='Logging level')
@click.option('--log-file', type=click.Path(), 
              help='Path to log file')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose output')
@click.pass_context
def cli(ctx, config, log_level, log_file, verbose):
    """
    huaju4k - Theater Video Enhancement Tool
    
    Transform theater drama videos to 4K resolution with specialized audio optimization.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Set up logging
    if verbose:
        log_level = 'DEBUG'
    setup_logging(log_level, log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting huaju4k video enhancement tool")
    
    # Load configuration
    try:
        config_manager = ConfigManager()
        if config:
            app_config = config_manager.load_config(config)
        else:
            app_config = config_manager.load_config()
        
        ctx.obj['config'] = app_config
        ctx.obj['config_manager'] = config_manager
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        click.echo(f"Error: Failed to load configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), 
              help='Output file path (default: auto-generated)')
@click.option('--preset', '-p', default='theater_medium_balanced',
              help='Theater preset (theater_small_fast, theater_medium_balanced, theater_large_high)')
@click.option('--quality', '-q', default='balanced',
              type=click.Choice(['fast', 'balanced', 'high']),
              help='Quality level')
@click.option('--no-audio', is_flag=True,
              help='Skip audio enhancement')
@click.option('--gpu/--no-gpu', default=None,
              help='Force GPU usage on/off')
@click.option('--use-ffmpeg/--use-opencv', default=True,
              help='Use FFmpeg pipeline (recommended) or OpenCV fallback')
@click.pass_context
def process(ctx, input_path, output, preset, quality, no_audio, gpu, use_ffmpeg):
    """
    Process a video file with theater enhancement.
    
    INPUT_PATH: Path to input video file
    """
    logger = logging.getLogger(__name__)
    
    try:
        from .core.integrated_video_processor import create_integrated_processor
        from .models.data_models import ProcessingStrategy
        
        click.echo(f"Processing video: {input_path}")
        click.echo(f"Preset: {preset}")
        click.echo(f"Quality: {quality}")
        click.echo(f"Audio enhancement: {'disabled' if no_audio else 'enabled'}")
        click.echo(f"Pipeline: {'FFmpeg (recommended)' if use_ffmpeg else 'OpenCV (fallback)'}")
        
        if gpu is not None:
            click.echo(f"GPU usage: {'forced on' if gpu else 'forced off'}")
        
        # 生成输出路径
        if output:
            output_path = output
        else:
            input_file = Path(input_path)
            suffix = "_4k_enhanced_ffmpeg" if use_ffmpeg else "_4k_enhanced_opencv"
            output_path = input_file.parent / f"{input_file.stem}{suffix}{input_file.suffix}"
        
        click.echo(f"Output path: {output_path}")
        
        # 创建处理策略
        strategy = ProcessingStrategy(
            tile_size=(512, 512),
            overlap_pixels=32,
            batch_size=1,
            use_gpu=gpu if gpu is not None else True,
            ai_model="real-esrgan",
            quality_preset=quality,
            memory_limit_mb=4096,
            target_resolution=(0, 0),  # 将根据输入自动计算
            denoise_strength=0.7,
            media_pipeline={
                "controller": "ffmpeg" if use_ffmpeg else "opencv",
                "video_codec": "libx264",
                "audio_mode": "copy" if no_audio else "theater_enhanced"
            },
            audio_enhancement=not no_audio
        )
        
        # 创建并使用集成处理器
        if use_ffmpeg:
            # 使用新的FFmpeg集成处理器
            config_dict = ctx.obj.get('config', {})
            if hasattr(config_dict, 'to_dict'):
                config_dict = config_dict.to_dict()
            
            with create_integrated_processor(config_dict) as processor:
                click.echo("Starting FFmpeg-based processing...")
                
                # 显示进度
                def show_progress():
                    import time
                    while processor.processing_active:
                        progress_info = processor.get_processing_progress()
                        progress = progress_info['overall_progress']
                        stage = progress_info['current_stage']
                        eta = progress_info['eta_seconds']
                        
                        click.echo(f"\r进度: {progress*100:.1f}% | 阶段: {stage} | "
                                 f"预计剩余: {eta:.0f}秒", nl=False)
                        time.sleep(1)
                    click.echo()  # 换行
                
                # 启动进度显示线程
                import threading
                progress_thread = threading.Thread(target=show_progress, daemon=True)
                progress_thread.start()
                
                # 执行处理
                result = processor.process(input_path, str(output_path), strategy)
                
                if result.success:
                    click.echo(f"✅ 处理成功完成!")
                    click.echo(f"输出文件: {result.output_path}")
                    click.echo(f"处理时间: {result.processing_time:.2f}秒")
                    if result.frames_processed:
                        click.echo(f"处理帧数: {result.frames_processed}")
                else:
                    click.echo(f"❌ 处理失败: {result.error}", err=True)
                    sys.exit(1)
        else:
            # 回退到传统OpenCV处理器
            click.echo("Using OpenCV fallback mode...")
            click.echo("Note: OpenCV mode has limited audio support and compatibility")
            
            # 这里可以调用原有的处理器
            # 暂时显示占位信息
            click.echo("OpenCV fallback implementation pending...")
        
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        click.echo(f"Error: Missing required dependencies: {e}", err=True)
        click.echo("Please install FFmpeg and ensure it's available in PATH", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        click.echo(f"Error: Processing failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--detailed', is_flag=True,
              help='Show detailed system information')
@click.pass_context
def info(ctx, detailed):
    """Show system information and compatibility."""
    logger = logging.getLogger(__name__)
    
    try:
        click.echo("huaju4k System Information")
        click.echo("=" * 40)
        
        # Basic system info
        system_info = get_system_info()
        click.echo(f"Platform: {system_info.get('platform', 'Unknown')}")
        click.echo(f"Architecture: {system_info.get('architecture', 'Unknown')}")
        click.echo(f"CPU Cores: {system_info.get('cpu_cores', 'Unknown')}")
        click.echo(f"Total Memory: {system_info.get('total_memory_mb', 0)} MB")
        click.echo(f"Available Memory: {system_info.get('available_memory_mb', 0)} MB")
        
        # GPU information
        if system_info.get('gpu_available', False):
            click.echo(f"GPU: {system_info.get('gpu_name', 'Unknown')}")
            click.echo(f"GPU Memory: {system_info.get('gpu_memory_mb', 0)} MB")
            click.echo(f"CUDA Available: {system_info.get('cuda_available', False)}")
        else:
            click.echo("GPU: Not available")
        
        # Dependencies
        click.echo("\nDependency Status:")
        click.echo("-" * 20)
        deps = check_dependencies()
        for dep, available in deps.items():
            status = "✓" if available else "✗"
            click.echo(f"{status} {dep}")
        
        # System requirements validation
        click.echo("\nSystem Requirements:")
        click.echo("-" * 20)
        req_validation = validate_system_requirements()
        if req_validation['valid']:
            click.echo("✓ System meets minimum requirements")
        else:
            click.echo("✗ System does not meet minimum requirements")
            for error in req_validation['errors']:
                click.echo(f"  - {error}")
        
        for warning in req_validation['warnings']:
            click.echo(f"⚠ {warning}")
        
        if detailed:
            click.echo("\nDetailed Information:")
            click.echo("-" * 20)
            for key, value in system_info.items():
                if key not in ['platform', 'architecture', 'cpu_cores', 
                              'total_memory_mb', 'available_memory_mb', 
                              'gpu_available', 'gpu_name', 'gpu_memory_mb', 'cuda_available']:
                    click.echo(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        click.echo(f"Error: Failed to get system information: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def presets(ctx):
    """List available theater presets."""
    logger = logging.getLogger(__name__)
    
    try:
        from .configs.config_manager import PresetManager
        
        preset_manager = PresetManager()
        available_presets = preset_manager.list_available_presets()
        
        click.echo("Available Theater Presets:")
        click.echo("=" * 40)
        
        for preset_name in available_presets:
            try:
                preset_info = preset_manager.get_preset_info(preset_name)
                click.echo(f"\n{preset_name}:")
                click.echo(f"  Description: {preset_info.get('description', 'N/A')}")
                click.echo(f"  Theater Size: {preset_info.get('theater_size', 'N/A')}")
                click.echo(f"  Quality Level: {preset_info.get('quality_level', 'N/A')}")
                click.echo(f"  Target Resolution: {preset_info.get('target_resolution', 'N/A')}")
            except Exception as e:
                click.echo(f"\n{preset_name}: Error loading preset info - {e}")
        
    except Exception as e:
        logger.error(f"Failed to list presets: {e}")
        click.echo(f"Error: Failed to list presets: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_paths', nargs=-1, type=click.Path(exists=True))
@click.option('--output-dir', '-d', type=click.Path(),
              help='Output directory for batch processing')
@click.option('--preset', '-p', default='theater_medium_balanced',
              help='Theater preset for all videos')
@click.option('--quality', '-q', default='balanced',
              type=click.Choice(['fast', 'balanced', 'high']),
              help='Quality level for all videos')
@click.option('--continue-on-error', is_flag=True,
              help='Continue processing other videos if one fails')
@click.pass_context
def batch(ctx, input_paths, output_dir, preset, quality, continue_on_error):
    """
    Process multiple video files in batch.
    
    INPUT_PATHS: Paths to input video files
    """
    logger = logging.getLogger(__name__)
    
    if not input_paths:
        click.echo("Error: No input files specified", err=True)
        sys.exit(1)
    
    try:
        click.echo(f"Batch processing {len(input_paths)} video(s)")
        click.echo(f"Preset: {preset}")
        click.echo(f"Quality: {quality}")
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            click.echo(f"Output directory: {output_path}")
        
        # Placeholder for actual batch processing
        for i, input_path in enumerate(input_paths, 1):
            click.echo(f"\n[{i}/{len(input_paths)}] Processing: {input_path}")
            
            # This would call the actual processing function
            # For now, just show what would be processed
            input_file = Path(input_path)
            if output_dir:
                output_file = Path(output_dir) / f"{input_file.stem}_4k_enhanced{input_file.suffix}"
            else:
                output_file = input_file.parent / f"{input_file.stem}_4k_enhanced{input_file.suffix}"
            
            click.echo(f"  Output: {output_file}")
            click.echo("  Status: Placeholder - implementation pending")
        
        click.echo("\nNote: Batch processing implementation will be added in subsequent tasks")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        click.echo(f"Error: Batch processing failed: {e}", err=True)
        if not continue_on_error:
            sys.exit(1)


# Add system commands to the main CLI
cli.add_command(system)


def main():
    """Main entry point for the application."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()