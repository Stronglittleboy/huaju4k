#!/usr/bin/env python3
"""
主CLI应用程序 - huaju4k视频增强工具

实现任务12.1的要求：
- 实现基于Click的命令行界面
- 添加命令解析和验证
- 创建输出路径处理系统
- 需求: 2.1, 2.2, 2.5
"""

import os
import sys
import click
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# 添加项目路径以支持导入
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from huaju4k.core.video_enhancement_processor import VideoEnhancementProcessor
from huaju4k.models.data_models import ProcessResult
from huaju4k.utils.system_utils import get_system_info, check_dependencies
from huaju4k.cli.utils import (
    setup_logging, validate_input_file, generate_output_path,
    display_system_info, display_processing_result, handle_processing_error
)

# 配置日志
logger = logging.getLogger(__name__)

# 支持的预设
THEATER_PRESETS = ['theater_small', 'theater_medium', 'theater_large']
QUALITY_LEVELS = ['fast', 'balanced', 'high']

@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='显示版本信息')
@click.option('--system-info', is_flag=True, help='显示系统信息和兼容性状态')
@click.pass_context
def cli(ctx, version, system_info):
    """
    huaju4k - 专业的戏剧视频4K增强工具
    
    将1080p/720p戏剧视频增强至4K分辨率，并优化戏剧音频效果。
    """
    if ctx.invoked_subcommand is None:
        if version:
            display_version()
        elif system_info:
            display_system_info()
        else:
            click.echo(ctx.get_help())

@cli.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True))
@click.option('-o', '--output', 'output_path', 
              type=click.Path(), 
              help='输出文件路径（默认：输入文件目录下的enhanced子目录）')
@click.option('-p', '--preset', 
              type=click.Choice(THEATER_PRESETS, case_sensitive=False),
              default='theater_medium',
              help='剧院预设配置 (默认: theater_medium)')
@click.option('-q', '--quality',
              type=click.Choice(QUALITY_LEVELS, case_sensitive=False),
              default='balanced',
              help='质量级别 (默认: balanced)')
@click.option('-c', '--config',
              type=click.Path(exists=True, readable=True),
              help='自定义配置文件路径')
@click.option('-v', '--verbose', is_flag=True,
              help='显示详细处理信息')
@click.option('--dry-run', is_flag=True,
              help='预览处理参数，不执行实际处理')
@click.option('--force', is_flag=True,
              help='强制覆盖已存在的输出文件')
def enhance(input_file, output_path, preset, quality, config, verbose, dry_run, force):
    """
    增强单个视频文件
    
    实现需求2.1: 处理视频文件与默认theater-medium预设
    实现需求2.2: 指定输出位置保存增强视频
    实现需求2.3: 应用剧院特定配置
    实现需求2.4: 调整处理参数
    
    示例:
        huaju4k enhance video.mp4
        huaju4k enhance video.mp4 -o enhanced_video.mp4 -p theater_large -q high
    """
    # 设置日志级别
    setup_logging(verbose)
    
    try:
        # 验证输入文件
        input_path = validate_input_file(input_file)
        logger.info(f"输入文件: {input_path}")
        
        # 生成输出路径
        if output_path is None:
            output_path = generate_output_path(input_path, preset, quality)
        else:
            output_path = Path(output_path).resolve()
        
        logger.info(f"输出路径: {output_path}")
        
        # 检查输出文件是否存在
        if output_path.exists() and not force:
            click.echo(f"错误: 输出文件已存在: {output_path}")
            click.echo("使用 --force 选项强制覆盖")
            sys.exit(1)
        
        # 显示处理参数
        click.echo(f"输入文件: {input_path}")
        click.echo(f"输出路径: {output_path}")
        click.echo(f"剧院预设: {preset}")
        click.echo(f"质量级别: {quality}")
        if config:
            click.echo(f"配置文件: {config}")
        
        # 预览模式
        if dry_run:
            click.echo("\n预览模式 - 不执行实际处理")
            click.echo("处理参数已验证，可以执行实际处理")
            return
        
        # 确认处理
        if not click.confirm("\n开始处理？"):
            click.echo("处理已取消")
            return
        
        # 初始化处理器
        click.echo("\n初始化视频增强处理器...")
        processor = VideoEnhancementProcessor(config_path=config)
        
        # 执行处理
        click.echo("开始视频增强处理...")
        result = processor.process(
            input_path=str(input_path),
            output_path=str(output_path),
            preset=preset,
            quality=quality
        )
        
        # 显示结果
        display_processing_result(result)
        
        if result.success:
            click.echo(f"\n✅ 处理完成！输出文件: {result.output_path}")
            sys.exit(0)
        else:
            click.echo(f"\n❌ 处理失败: {result.error}")
            sys.exit(1)
            
    except Exception as e:
        handle_processing_error(e, verbose)
        sys.exit(1)

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('-o', '--output-dir', 'output_dir',
              type=click.Path(file_okay=False, dir_okay=True),
              help='输出目录（默认：输入目录下的enhanced子目录）')
@click.option('-p', '--preset',
              type=click.Choice(THEATER_PRESETS, case_sensitive=False),
              default='theater_medium',
              help='剧院预设配置 (默认: theater_medium)')
@click.option('-q', '--quality',
              type=click.Choice(QUALITY_LEVELS, case_sensitive=False),
              default='balanced',
              help='质量级别 (默认: balanced)')
@click.option('-c', '--config',
              type=click.Path(exists=True, readable=True),
              help='自定义配置文件路径')
@click.option('--pattern', default='*.mp4',
              help='文件匹配模式 (默认: *.mp4)')
@click.option('--recursive', is_flag=True,
              help='递归搜索子目录')
@click.option('-v', '--verbose', is_flag=True,
              help='显示详细处理信息')
@click.option('--dry-run', is_flag=True,
              help='预览要处理的文件，不执行实际处理')
@click.option('--force', is_flag=True,
              help='强制覆盖已存在的输出文件')
@click.option('--continue-on-error', is_flag=True,
              help='遇到错误时继续处理其他文件')
def batch(input_dir, output_dir, preset, quality, config, pattern, recursive, 
          verbose, dry_run, force, continue_on_error):
    """
    批量处理视频文件
    
    实现需求2.6: 批量处理多个视频文件
    实现需求12.1, 12.2, 12.3, 12.4: 批量处理功能
    
    示例:
        huaju4k batch /path/to/videos
        huaju4k batch /path/to/videos -o /path/to/output --recursive
    """
    # 设置日志级别
    setup_logging(verbose)
    
    try:
        from huaju4k.cli.batch_processor import BatchProcessor
        
        # 初始化批处理器
        batch_processor = BatchProcessor(
            input_dir=input_dir,
            output_dir=output_dir,
            preset=preset,
            quality=quality,
            config_path=config,
            pattern=pattern,
            recursive=recursive,
            force=force,
            continue_on_error=continue_on_error,
            verbose=verbose
        )
        
        # 预览模式
        if dry_run:
            batch_processor.preview()
            return
        
        # 执行批处理
        batch_processor.process()
        
    except Exception as e:
        handle_processing_error(e, verbose)
        sys.exit(1)

@cli.command()
@click.option('--detailed', is_flag=True, help='显示详细系统信息')
def info(detailed):
    """
    显示系统信息和兼容性状态
    
    实现需求2.5: 显示硬件能力和兼容性状态
    """
    display_system_info(detailed)

@cli.command()
@click.option('--list-presets', is_flag=True, help='列出所有可用预设')
@click.option('--validate', type=click.Path(exists=True, readable=True),
              help='验证配置文件')
@click.option('--create-preset', type=str,
              help='创建新预设（指定预设名称）')
def config(list_presets, validate, create_preset):
    """
    配置和预设管理
    
    实现需求8.1, 8.2, 8.3, 8.5, 8.6: 配置和预设管理
    """
    try:
        from huaju4k.cli.config_manager import ConfigCLI
        
        config_cli = ConfigCLI()
        
        if list_presets:
            config_cli.list_presets()
        elif validate:
            config_cli.validate_config(validate)
        elif create_preset:
            config_cli.create_preset(create_preset)
        else:
            click.echo("请指定配置操作。使用 --help 查看可用选项。")
            
    except Exception as e:
        click.echo(f"配置操作失败: {e}")
        sys.exit(1)

def display_version():
    """显示版本信息"""
    try:
        from huaju4k import __version__
        version = __version__
    except ImportError:
        version = "开发版本"
    
    click.echo(f"huaju4k 视频增强工具 v{version}")
    click.echo("专业的戏剧视频4K增强工具")
    click.echo("Copyright (c) 2025 huaju4k项目")

if __name__ == '__main__':
    cli()