"""
批处理器 - 批量视频处理功能

实现任务12.3的要求：
- 实现批量视频处理
- 创建批处理进度跟踪
- 添加批处理错误处理和报告
- 需求: 2.6, 12.1, 12.2, 12.3, 12.4
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import click

from ..core.video_enhancement_processor import VideoEnhancementProcessor
from ..models.data_models import ProcessResult
from .utils import validate_input_file, generate_output_path

logger = logging.getLogger(__name__)

@dataclass
class BatchItem:
    """批处理项目"""
    input_path: Path
    output_path: Path
    status: str = "pending"  # pending, processing, completed, failed, skipped
    result: Optional[ProcessResult] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None

@dataclass
class BatchReport:
    """批处理报告"""
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_processing_time: float = 0.0
    total_input_size_mb: float = 0.0
    total_output_size_mb: float = 0.0
    average_processing_speed: float = 0.0
    items: List[BatchItem] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class BatchProcessor:
    """
    批量视频处理器
    
    实现需求2.6: 批量处理多个视频文件
    实现需求12.1, 12.2, 12.3, 12.4: 批量处理功能
    """
    
    def __init__(self, input_dir: str, output_dir: Optional[str] = None,
                 preset: str = "theater_medium", quality: str = "balanced",
                 config_path: Optional[str] = None, pattern: str = "*.mp4",
                 recursive: bool = False, force: bool = False,
                 continue_on_error: bool = False, verbose: bool = False):
        """
        初始化批处理器
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            preset: 剧院预设
            quality: 质量级别
            config_path: 配置文件路径
            pattern: 文件匹配模式
            recursive: 是否递归搜索
            force: 是否强制覆盖
            continue_on_error: 遇到错误时是否继续
            verbose: 是否显示详细信息
        """
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.preset = preset
        self.quality = quality
        self.config_path = config_path
        self.pattern = pattern
        self.recursive = recursive
        self.force = force
        self.continue_on_error = continue_on_error
        self.verbose = verbose
        
        # 初始化处理器
        self.processor = None
        self.report = BatchReport()
        
        # 发现文件
        self._discover_files()
    
    def _discover_files(self) -> None:
        """发现要处理的文件"""
        click.echo(f"搜索文件: {self.input_dir} (模式: {self.pattern})")
        
        if self.recursive:
            files = list(self.input_dir.rglob(self.pattern))
        else:
            files = list(self.input_dir.glob(self.pattern))
        
        # 过滤文件
        video_files = []
        supported_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
        
        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                video_files.append(file_path)
        
        # 创建批处理项目
        for video_file in sorted(video_files):
            # 生成输出路径
            if self.output_dir:
                # 保持相对目录结构
                rel_path = video_file.relative_to(self.input_dir)
                output_path = self.output_dir / rel_path.parent / f"{rel_path.stem}_enhanced_{self.preset}_{self.quality}_4k{rel_path.suffix}"
            else:
                output_path = generate_output_path(video_file, self.preset, self.quality)
            
            # 创建输出目录
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 检查是否跳过已存在的文件
            if output_path.exists() and not self.force:
                item = BatchItem(
                    input_path=video_file,
                    output_path=output_path,
                    status="skipped",
                    error="输出文件已存在"
                )
                self.report.skipped_files += 1
            else:
                item = BatchItem(
                    input_path=video_file,
                    output_path=output_path
                )
            
            self.report.items.append(item)
        
        self.report.total_files = len(self.report.items)
        
        click.echo(f"发现 {len(video_files)} 个视频文件")
        if self.report.skipped_files > 0:
            click.echo(f"跳过 {self.report.skipped_files} 个已存在的文件")
    
    def preview(self) -> None:
        """预览要处理的文件"""
        click.echo("=" * 60)
        click.echo("批处理预览")
        click.echo("=" * 60)
        
        click.echo(f"输入目录: {self.input_dir}")
        if self.output_dir:
            click.echo(f"输出目录: {self.output_dir}")
        click.echo(f"剧院预设: {self.preset}")
        click.echo(f"质量级别: {self.quality}")
        click.echo(f"文件模式: {self.pattern}")
        click.echo(f"递归搜索: {'是' if self.recursive else '否'}")
        click.echo(f"强制覆盖: {'是' if self.force else '否'}")
        
        click.echo(f"\n要处理的文件 ({len([item for item in self.report.items if item.status == 'pending'])}):")
        
        for i, item in enumerate(self.report.items, 1):
            status_icon = {
                "pending": "📝",
                "skipped": "⏭️"
            }.get(item.status, "❓")
            
            click.echo(f"{i:3d}. {status_icon} {item.input_path.name}")
            if item.status == "skipped":
                click.echo(f"     跳过原因: {item.error}")
            else:
                click.echo(f"     输出: {item.output_path}")
        
        # 统计信息
        pending_count = len([item for item in self.report.items if item.status == "pending"])
        if pending_count > 0:
            click.echo(f"\n将处理 {pending_count} 个文件")
        else:
            click.echo("\n没有文件需要处理")
    
    def process(self) -> None:
        """执行批处理"""
        pending_items = [item for item in self.report.items if item.status == "pending"]
        
        if not pending_items:
            click.echo("没有文件需要处理")
            return
        
        click.echo("=" * 60)
        click.echo("开始批处理")
        click.echo("=" * 60)
        
        # 确认处理
        if not click.confirm(f"确定要处理 {len(pending_items)} 个文件？"):
            click.echo("批处理已取消")
            return
        
        # 初始化处理器
        try:
            self.processor = VideoEnhancementProcessor(config_path=self.config_path)
        except Exception as e:
            click.echo(f"初始化处理器失败: {e}")
            return
        
        # 开始批处理
        self.report.start_time = datetime.now()
        
        for i, item in enumerate(pending_items, 1):
            click.echo(f"\n处理文件 {i}/{len(pending_items)}: {item.input_path.name}")
            
            try:
                # 更新状态
                item.status = "processing"
                item.start_time = datetime.now()
                
                # 执行处理
                result = self.processor.process(
                    input_path=str(item.input_path),
                    output_path=str(item.output_path),
                    preset=self.preset,
                    quality=self.quality
                )
                
                # 更新结果
                item.result = result
                item.end_time = datetime.now()
                
                if result.success:
                    item.status = "completed"
                    self.report.completed_files += 1
                    click.echo(f"✅ 完成: {result.processing_time:.1f}秒")
                else:
                    item.status = "failed"
                    item.error = result.error
                    self.report.failed_files += 1
                    click.echo(f"❌ 失败: {result.error}")
                    
                    if not self.continue_on_error:
                        click.echo("遇到错误，停止批处理")
                        break
                
            except Exception as e:
                item.status = "failed"
                item.error = str(e)
                item.end_time = datetime.now()
                self.report.failed_files += 1
                
                click.echo(f"❌ 异常: {e}")
                logger.error(f"处理文件 {item.input_path} 时发生异常: {e}")
                
                if not self.continue_on_error:
                    click.echo("遇到异常，停止批处理")
                    break
        
        # 完成批处理
        self.report.end_time = datetime.now()
        self._generate_report()
    
    def _generate_report(self) -> None:
        """生成批处理报告"""
        click.echo("\n" + "=" * 60)
        click.echo("批处理报告")
        click.echo("=" * 60)
        
        # 基本统计
        click.echo(f"总文件数: {self.report.total_files}")
        click.echo(f"成功处理: {self.report.completed_files}")
        click.echo(f"处理失败: {self.report.failed_files}")
        click.echo(f"跳过文件: {self.report.skipped_files}")
        
        # 时间统计
        if self.report.start_time and self.report.end_time:
            total_time = (self.report.end_time - self.report.start_time).total_seconds()
            click.echo(f"总用时: {total_time:.1f} 秒")
            
            if self.report.completed_files > 0:
                avg_time = total_time / self.report.completed_files
                click.echo(f"平均处理时间: {avg_time:.1f} 秒/文件")
        
        # 文件大小统计
        total_input_size = 0
        total_output_size = 0
        
        for item in self.report.items:
            if item.status == "completed" and item.result and item.result.success:
                # 输入文件大小
                if item.input_path.exists():
                    total_input_size += item.input_path.stat().st_size
                
                # 输出文件大小
                if item.output_path.exists():
                    total_output_size += item.output_path.stat().st_size
        
        if total_input_size > 0:
            click.echo(f"输入总大小: {total_input_size / (1024*1024):.1f} MB")
        if total_output_size > 0:
            click.echo(f"输出总大小: {total_output_size / (1024*1024):.1f} MB")
        
        # 失败文件详情
        failed_items = [item for item in self.report.items if item.status == "failed"]
        if failed_items:
            click.echo(f"\n失败文件详情:")
            for item in failed_items:
                click.echo(f"  ❌ {item.input_path.name}: {item.error}")
        
        # 成功率
        if self.report.total_files > 0:
            success_rate = (self.report.completed_files / self.report.total_files) * 100
            click.echo(f"\n成功率: {success_rate:.1f}%")
        
        # 保存报告到文件
        self._save_report_to_file()
    
    def _save_report_to_file(self) -> None:
        """保存报告到文件"""
        try:
            # 生成报告文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.input_dir / f"batch_report_{timestamp}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("huaju4k 批处理报告\n")
                f.write("=" * 60 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"输入目录: {self.input_dir}\n")
                if self.output_dir:
                    f.write(f"输出目录: {self.output_dir}\n")
                f.write(f"剧院预设: {self.preset}\n")
                f.write(f"质量级别: {self.quality}\n")
                f.write(f"文件模式: {self.pattern}\n")
                f.write(f"递归搜索: {'是' if self.recursive else '否'}\n")
                f.write("\n统计信息:\n")
                f.write(f"总文件数: {self.report.total_files}\n")
                f.write(f"成功处理: {self.report.completed_files}\n")
                f.write(f"处理失败: {self.report.failed_files}\n")
                f.write(f"跳过文件: {self.report.skipped_files}\n")
                
                if self.report.start_time and self.report.end_time:
                    total_time = (self.report.end_time - self.report.start_time).total_seconds()
                    f.write(f"总用时: {total_time:.1f} 秒\n")
                
                f.write("\n详细结果:\n")
                for i, item in enumerate(self.report.items, 1):
                    status_text = {
                        "completed": "✅ 成功",
                        "failed": "❌ 失败",
                        "skipped": "⏭️ 跳过",
                        "pending": "📝 待处理"
                    }.get(item.status, "❓ 未知")
                    
                    f.write(f"{i:3d}. {status_text} - {item.input_path.name}\n")
                    f.write(f"     输入: {item.input_path}\n")
                    f.write(f"     输出: {item.output_path}\n")
                    
                    if item.result and item.result.processing_time:
                        f.write(f"     处理时间: {item.result.processing_time:.1f} 秒\n")
                    
                    if item.error:
                        f.write(f"     错误: {item.error}\n")
                    
                    f.write("\n")
            
            click.echo(f"\n报告已保存到: {report_file}")
            
        except Exception as e:
            click.echo(f"保存报告失败: {e}")
            logger.error(f"保存批处理报告失败: {e}")