#!/usr/bin/env python3
"""
huaju4k CLI — 话剧视频 4K 增强工具命令行界面
"""

import json
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)

QUALITY_LEVELS = ["fast", "standard", "master"]
CODECS = ["h264", "h265", "prores"]


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="显示版本信息")
@click.pass_context
def cli(ctx, version):
    """huaju4k — 话剧视频 4K 增强工具"""
    if ctx.invoked_subcommand is None:
        if version:
            from huaju4k import __version__
            click.echo(f"huaju4k v{__version__}")
        else:
            click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# enhance 命令
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("input_file", type=click.Path(exists=True, readable=True))
@click.option("-o", "--output", "output_path", type=click.Path(), default=None,
              help="输出文件路径")
@click.option("-q", "--quality", type=click.Choice(QUALITY_LEVELS, case_sensitive=False),
              default="standard", help="质量档位: fast / standard / master (默认 standard)")
@click.option("--codec", type=click.Choice(CODECS, case_sensitive=False),
              default="h264", help="输出编码: h264 / h265 / prores (默认 h264)")
@click.option("--crf", type=int, default=18, help="质量参数 CRF (默认 18)")
@click.option("--preview", is_flag=True, help="预览模式: 只处理前 30 秒")
@click.option("--preview-at", type=str, default=None,
              help="预览模式: 从指定时间点开始处理 30 秒 (格式 HH:MM:SS)")
@click.option("--segment", nargs=2, type=str, default=None,
              help="只处理指定时间段 (格式: HH:MM:SS HH:MM:SS)")
@click.option("--resume", is_flag=True, help="从上次断点续传")
@click.option("-v", "--verbose", is_flag=True, help="详细日志")
@click.option("--force", is_flag=True, help="强制覆盖已有输出")
@click.option("--dry-run", is_flag=True, help="预览策略，不执行处理")
def enhance(input_file, output_path, quality, codec, crf, preview, preview_at,
            segment, resume, verbose, force, dry_run):
    """增强单个话剧视频到 4K

    \b
    示例:
      python -m huaju4k enhance input.mp4
      python -m huaju4k enhance input.mp4 --preview
      python -m huaju4k enhance input.mp4 --quality master --codec h265 --crf 20
      python -m huaju4k enhance input.mp4 --segment 00:15:00 00:25:00
      python -m huaju4k enhance input.mp4 --resume
    """
    _setup_logging(verbose)

    input_path = Path(input_file).resolve()

    if output_path is None:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_4k.mp4"
    else:
        output_path = Path(output_path).resolve()

    if output_path.exists() and not force:
        click.echo(f"输出文件已存在: {output_path}  (使用 --force 覆盖)")
        sys.exit(1)

    # ---- 延迟导入重量级模块 ----
    try:
        from huaju4k.analysis.stage_structure_analyzer import StageStructureAnalyzer
        from huaju4k.strategy.enhancement_planner import EnhancementStrategyPlanner
        from huaju4k.media.ffmpeg_media_controller import FFmpegMediaController
    except ImportError as e:
        click.echo(f"依赖缺失: {e}")
        sys.exit(1)

    # ---- 分析 ----
    click.echo(f"输入: {input_path}")
    click.echo(f"输出: {output_path}")
    click.echo(f"质量: {quality}  编码: {codec}  CRF: {crf}")

    media = FFmpegMediaController()
    video_info = media.analyze_input_video(str(input_path))

    analyzer = StageStructureAnalyzer()
    features = analyzer.analyze_structure(str(input_path))

    planner = EnhancementStrategyPlanner()
    strategy = planner.generate_strategy(features, quality=quality, codec=codec, crf=crf)

    # ---- 预处理确认 ----
    static_count = sum(1 for s in strategy.scenes if s.type == "static")
    gradual_count = sum(1 for s in strategy.scenes if s.type == "gradual")
    dynamic_count = sum(1 for s in strategy.scenes if s.type == "dynamic")

    click.echo(f"\n{'='*50}")
    click.echo(f"输入: {input_path.name} ({video_info.width}x{video_info.height}, "
               f"{video_info.duration:.0f}s, {video_info.file_size/(1024*1024):.1f}MB)")
    click.echo(f"策略: {strategy.model_name} (tile={strategy.tile_size})")
    click.echo(f"场景: {len(strategy.scenes)} 段 "
               f"(static {static_count} / gradual {gradual_count} / dynamic {dynamic_count})")
    click.echo(f"降噪: {strategy.denoise_strength}")

    if preview or preview_at:
        click.echo("模式: 预览 (30 秒)")
    elif segment:
        click.echo(f"模式: 片段 ({segment[0]} ~ {segment[1]})")
    elif resume:
        click.echo("模式: 断点续传")
    click.echo(f"{'='*50}")

    if dry_run:
        click.echo("\n[dry-run] 策略预览完成，未执行处理")
        return

    if not click.confirm("\n是否开始处理?", default=True):
        click.echo("已取消")
        return

    # ---- 执行处理 ----
    start_time = time.time()
    click.echo("\n开始处理...")

    try:
        from huaju4k.core.three_stage_enhancer import ThreeStageEnhancer

        enhancer = ThreeStageEnhancer()
        result = enhancer.process(
            input_path=str(input_path),
            output_path=str(output_path),
            strategy=strategy,
            preview=preview,
            preview_at=preview_at,
            segment=segment,
            resume=resume,
        )
    except Exception as e:
        logger.exception("处理失败")
        click.echo(f"\n处理失败: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time

    # ---- 完成报告 ----
    if result.success:
        report = result.report or {}
        click.echo(f"\n处理完成")
        click.echo(f"  耗时: {elapsed/3600:.1f}h | 平均速度: {result.frames_processed/elapsed:.2f} fps")
        click.echo(f"  输出: {result.output_path}")
        if report:
            report_path = Path(result.output_path).with_suffix(".report.json")
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            click.echo(f"  报告: {report_path}")
    else:
        click.echo(f"\n处理失败: {result.error}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# info 命令
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--detailed", is_flag=True, help="显示详细系统信息")
def info(detailed):
    """显示系统信息和兼容性状态"""
    from huaju4k import __version__
    click.echo(f"huaju4k v{__version__}\n")

    try:
        from huaju4k.utils.system_utils import get_system_info
        sys_info = get_system_info()
        for key, value in sys_info.items():
            click.echo(f"  {key}: {value}")
    except Exception as e:
        click.echo(f"  系统检测失败: {e}")

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            click.echo(f"\n  GPU: {name} ({mem:.1f} GB)")
        else:
            click.echo("\n  GPU: 不可用 (将使用 CPU lanczos)")
    except ImportError:
        click.echo("\n  GPU: PyTorch 未安装")


if __name__ == "__main__":
    cli()
