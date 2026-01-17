"""
CLI工具函数

提供命令行界面的辅助功能，包括：
- 日志设置
- 文件验证
- 输出路径生成
- 系统信息显示
- 结果显示
- 错误处理
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List

import click

from ..models.data_models import ProcessResult
from ..utils.system_utils import get_system_info, check_dependencies

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False) -> None:
    """
    设置日志配置
    
    Args:
        verbose: 是否启用详细日志
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # 配置根日志器
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 设置第三方库日志级别
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

def validate_input_file(input_file: str) -> Path:
    """
    验证输入文件
    
    Args:
        input_file: 输入文件路径
        
    Returns:
        验证后的Path对象
        
    Raises:
        click.ClickException: 文件验证失败
    """
    input_path = Path(input_file).resolve()
    
    # 检查文件是否存在
    if not input_path.exists():
        raise click.ClickException(f"输入文件不存在: {input_path}")
    
    # 检查是否为文件
    if not input_path.is_file():
        raise click.ClickException(f"输入路径不是文件: {input_path}")
    
    # 检查文件是否可读
    if not os.access(input_path, os.R_OK):
        raise click.ClickException(f"输入文件不可读: {input_path}")
    
    # 检查文件扩展名
    supported_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
    if input_path.suffix.lower() not in supported_extensions:
        click.echo(f"警告: 文件扩展名 '{input_path.suffix}' 可能不受支持")
        click.echo(f"支持的格式: {', '.join(supported_extensions)}")
        if not click.confirm("继续处理？"):
            raise click.Abort()
    
    return input_path

def generate_output_path(input_path: Path, preset: str, quality: str) -> Path:
    """
    生成输出文件路径
    
    实现需求2.2: 指定输出位置保存增强视频
    
    Args:
        input_path: 输入文件路径
        preset: 剧院预设
        quality: 质量级别
        
    Returns:
        生成的输出路径
    """
    # 创建输出目录
    output_dir = input_path.parent / "enhanced"
    output_dir.mkdir(exist_ok=True)
    
    # 生成输出文件名
    stem = input_path.stem
    suffix = input_path.suffix
    
    # 格式: 原文件名_enhanced_预设_质量_4k.扩展名
    output_name = f"{stem}_enhanced_{preset}_{quality}_4k{suffix}"
    
    return output_dir / output_name

def display_system_info(detailed: bool = False) -> None:
    """
    显示系统信息和兼容性状态
    
    实现需求2.5: 显示硬件能力和兼容性状态
    实现需求10.1: 检测系统能力并显示兼容性信息
    
    Args:
        detailed: 是否显示详细信息
    """
    click.echo("=" * 60)
    click.echo("huaju4k 系统信息")
    click.echo("=" * 60)
    
    try:
        # 获取系统信息
        system_info = get_system_info()
        
        # 基本系统信息
        click.echo(f"操作系统: {system_info.get('os', 'Unknown')}")
        click.echo(f"Python版本: {system_info.get('python_version', 'Unknown')}")
        click.echo(f"CPU核心数: {system_info.get('cpu_count', 'Unknown')}")
        click.echo(f"总内存: {system_info.get('total_memory_gb', 'Unknown')} GB")
        click.echo(f"可用内存: {system_info.get('available_memory_gb', 'Unknown')} GB")
        
        # GPU信息
        gpu_info = system_info.get('gpu', {})
        if gpu_info.get('available', False):
            click.echo(f"GPU: ✅ {gpu_info.get('name', 'Unknown')}")
            click.echo(f"GPU内存: {gpu_info.get('memory_gb', 'Unknown')} GB")
            click.echo(f"CUDA支持: {'✅' if gpu_info.get('cuda_available', False) else '❌'}")
        else:
            click.echo("GPU: ❌ 未检测到或不可用")
        
        # 依赖检查
        click.echo("\n依赖检查:")
        dependencies = check_dependencies()
        
        for dep_name, dep_info in dependencies.items():
            status = "✅" if dep_info.get('available', False) else "❌"
            version = dep_info.get('version', '未知版本')
            click.echo(f"  {dep_name}: {status} {version}")
        
        # 详细信息
        if detailed:
            click.echo("\n详细系统信息:")
            for key, value in system_info.items():
                if key not in ['os', 'python_version', 'cpu_count', 'total_memory_gb', 'available_memory_gb', 'gpu']:
                    click.echo(f"  {key}: {value}")
        
        # 兼容性评估
        click.echo("\n兼容性评估:")
        compatibility_score = _calculate_compatibility_score(system_info, dependencies)
        
        if compatibility_score >= 0.8:
            click.echo("🟢 系统完全兼容，推荐使用GPU加速")
        elif compatibility_score >= 0.6:
            click.echo("🟡 系统基本兼容，建议使用CPU模式")
        else:
            click.echo("🔴 系统兼容性较差，可能遇到性能问题")
        
        # 推荐设置
        click.echo("\n推荐设置:")
        recommendations = _get_system_recommendations(system_info, dependencies)
        for rec in recommendations:
            click.echo(f"  • {rec}")
            
    except Exception as e:
        click.echo(f"获取系统信息失败: {e}")
        logger.error(f"系统信息获取错误: {e}")

def display_processing_result(result: ProcessResult) -> None:
    """
    显示处理结果
    
    Args:
        result: 处理结果对象
    """
    click.echo("\n" + "=" * 60)
    click.echo("处理结果")
    click.echo("=" * 60)
    
    if result.success:
        click.echo("状态: ✅ 成功")
        if result.output_path:
            click.echo(f"输出文件: {result.output_path}")
        if result.processing_time:
            click.echo(f"处理时间: {result.processing_time:.1f} 秒")
        if result.frames_processed:
            click.echo(f"处理帧数: {result.frames_processed}")
        if result.processing_speed_fps:
            click.echo(f"处理速度: {result.processing_speed_fps:.2f} FPS")
        if result.memory_peak_mb:
            click.echo(f"峰值内存: {result.memory_peak_mb} MB")
        
        # 质量指标
        if result.quality_metrics:
            click.echo("\n质量指标:")
            for metric, value in result.quality_metrics.items():
                if isinstance(value, float):
                    click.echo(f"  {metric}: {value:.3f}")
                else:
                    click.echo(f"  {metric}: {value}")
        
        # 性能报告
        if result.performance_report:
            click.echo("\n性能报告:")
            perf_metrics = result.performance_report.get('performance_metrics', {})
            resource_util = result.performance_report.get('resource_utilization', {})
            
            if perf_metrics:
                click.echo("  处理性能:")
                for metric, value in perf_metrics.items():
                    if isinstance(value, float):
                        click.echo(f"    {metric}: {value:.3f}")
                    else:
                        click.echo(f"    {metric}: {value}")
            
            if resource_util:
                click.echo("  资源利用率:")
                for metric, value in resource_util.items():
                    if isinstance(value, float):
                        click.echo(f"    {metric}: {value:.1f}")
                    else:
                        click.echo(f"    {metric}: {value}")
    else:
        click.echo("状态: ❌ 失败")
        if result.error:
            click.echo(f"错误: {result.error}")
        if result.processing_time:
            click.echo(f"运行时间: {result.processing_time:.1f} 秒")

def handle_processing_error(error: Exception, verbose: bool = False) -> None:
    """
    处理处理错误
    
    Args:
        error: 异常对象
        verbose: 是否显示详细错误信息
    """
    click.echo(f"\n❌ 处理失败: {error}")
    
    if verbose:
        click.echo("\n详细错误信息:")
        click.echo(traceback.format_exc())
    else:
        click.echo("使用 -v 选项查看详细错误信息")
    
    # 根据错误类型提供建议
    error_type = type(error).__name__
    suggestions = _get_error_suggestions(error_type, str(error))
    
    if suggestions:
        click.echo("\n建议:")
        for suggestion in suggestions:
            click.echo(f"  • {suggestion}")

def _calculate_compatibility_score(system_info: Dict[str, Any], 
                                 dependencies: Dict[str, Any]) -> float:
    """计算系统兼容性评分"""
    score = 0.0
    
    # Python版本检查
    python_version = system_info.get('python_version', '')
    if python_version.startswith('3.8') or python_version.startswith('3.9') or python_version.startswith('3.10'):
        score += 0.2
    
    # 内存检查
    available_memory = system_info.get('available_memory_gb', 0)
    if available_memory >= 8:
        score += 0.3
    elif available_memory >= 4:
        score += 0.2
    elif available_memory >= 2:
        score += 0.1
    
    # GPU检查
    gpu_info = system_info.get('gpu', {})
    if gpu_info.get('available', False):
        score += 0.2
        if gpu_info.get('cuda_available', False):
            score += 0.1
    
    # 依赖检查
    required_deps = ['opencv-python', 'numpy', 'click']
    available_deps = sum(1 for dep in required_deps 
                        if dependencies.get(dep, {}).get('available', False))
    score += (available_deps / len(required_deps)) * 0.2
    
    return min(score, 1.0)

def _get_system_recommendations(system_info: Dict[str, Any], 
                              dependencies: Dict[str, Any]) -> List[str]:
    """获取系统推荐设置"""
    recommendations = []
    
    # 内存建议
    available_memory = system_info.get('available_memory_gb', 0)
    if available_memory < 4:
        recommendations.append("系统内存较少，建议使用 --quality fast 选项")
    elif available_memory >= 8:
        recommendations.append("系统内存充足，可以使用 --quality high 选项")
    
    # GPU建议
    gpu_info = system_info.get('gpu', {})
    if gpu_info.get('available', False):
        if gpu_info.get('cuda_available', False):
            recommendations.append("检测到CUDA支持，将自动启用GPU加速")
        else:
            recommendations.append("检测到GPU但无CUDA支持，将使用CPU模式")
    else:
        recommendations.append("未检测到GPU，将使用CPU模式")
    
    # 依赖建议
    missing_deps = []
    for dep_name, dep_info in dependencies.items():
        if not dep_info.get('available', False):
            missing_deps.append(dep_name)
    
    if missing_deps:
        recommendations.append(f"缺少依赖: {', '.join(missing_deps)}，请先安装")
    
    return recommendations

def _get_error_suggestions(error_type: str, error_message: str) -> List[str]:
    """根据错误类型获取建议"""
    suggestions = []
    
    if error_type == "FileNotFoundError":
        suggestions.append("检查输入文件路径是否正确")
        suggestions.append("确保文件存在且可访问")
    elif error_type == "PermissionError":
        suggestions.append("检查文件权限")
        suggestions.append("尝试以管理员权限运行")
    elif error_type == "MemoryError":
        suggestions.append("系统内存不足，尝试使用 --quality fast 选项")
        suggestions.append("关闭其他应用程序释放内存")
    elif error_type == "ImportError":
        suggestions.append("检查依赖库是否正确安装")
        suggestions.append("尝试重新安装相关依赖")
    elif "CUDA" in error_message or "GPU" in error_message:
        suggestions.append("GPU相关错误，系统将自动回退到CPU模式")
        suggestions.append("检查GPU驱动是否正确安装")
    
    if not suggestions:
        suggestions.append("请检查输入参数和系统环境")
        suggestions.append("使用 --verbose 选项获取更多调试信息")
    
    return suggestions