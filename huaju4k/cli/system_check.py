"""
System compatibility check CLI command for huaju4k.

This module provides a command-line interface for running system compatibility checks
and generating system reports.
"""

import click
import json
import logging
from pathlib import Path

from ..core.compatibility_checker import CompatibilityChecker

logger = logging.getLogger(__name__)


@click.group()
def system():
    """System compatibility and optimization commands."""
    pass


@system.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path for report')
@click.option('--format', 'output_format', type=click.Choice(['json', 'text']), 
              default='text', help='Output format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def check(output, output_format, verbose):
    """Run comprehensive system compatibility check."""
    try:
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        
        click.echo("🔍 Running system compatibility check...")
        
        # Initialize compatibility checker
        checker = CompatibilityChecker()
        
        # Run full compatibility check
        results = checker.run_full_compatibility_check()
        
        if 'error' in results:
            click.echo(f"❌ Compatibility check failed: {results['error']}", err=True)
            return
        
        # Display results
        if output_format == 'text':
            _display_text_results(results)
        else:
            _display_json_results(results)
        
        # Save report if requested
        if output:
            if output_format == 'json':
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, default=str)
            else:
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(_format_text_report(results))
            
            click.echo(f"📄 Report saved to: {output}")
        
    except Exception as e:
        click.echo(f"❌ System check failed: {str(e)}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)


@system.command()
def requirements():
    """Check minimum system requirements."""
    try:
        click.echo("📋 Checking minimum system requirements...")
        
        checker = CompatibilityChecker()
        results = checker.check_minimum_requirements()
        
        if 'error' in results:
            click.echo(f"❌ Requirements check failed: {results['error']}", err=True)
            return
        
        # Display results
        overall_status = "✅ PASSED" if results['overall_passed'] else "❌ FAILED"
        click.echo(f"\n🎯 Overall Status: {overall_status}")
        
        click.echo("\n📊 Requirement Details:")
        for check_name, check_data in results['checks'].items():
            status = "✅" if check_data['passed'] else "❌"
            click.echo(f"  {status} {check_name.title()}: {check_data.get('available', 'N/A')}")
            if not check_data['passed']:
                required = check_data.get('required', 'N/A')
                click.echo(f"      Required: {required}")
        
        if not results['overall_passed']:
            click.echo("\n⚠️  System does not meet minimum requirements for optimal performance.")
        
    except Exception as e:
        click.echo(f"❌ Requirements check failed: {str(e)}", err=True)


@system.command()
def optimize():
    """Get system optimization recommendations."""
    try:
        click.echo("🚀 Analyzing system for optimization opportunities...")
        
        checker = CompatibilityChecker()
        recommendations = checker.get_optimization_recommendations()
        
        if 'error' in recommendations:
            click.echo(f"❌ Optimization analysis failed: {recommendations['error']}", err=True)
            return
        
        # Display recommendations by category
        for category, recs in recommendations['recommendations'].items():
            if recs:
                click.echo(f"\n🔧 {category.title()} Recommendations:")
                for rec in recs:
                    priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec['priority'], "⚪")
                    click.echo(f"  {priority_icon} {rec['description']}")
                    if 'impact' in rec:
                        click.echo(f"      Impact: {rec['impact']}")
        
        # Display priority summary
        if 'priority_summary' in recommendations:
            summary = recommendations['priority_summary']
            click.echo(f"\n📈 Priority Summary:")
            click.echo(f"  🔴 High Priority: {summary.get('high', 0)} items")
            click.echo(f"  🟡 Medium Priority: {summary.get('medium', 0)} items")
            click.echo(f"  🟢 Low Priority: {summary.get('low', 0)} items")
        
    except Exception as e:
        click.echo(f"❌ Optimization analysis failed: {str(e)}", err=True)


@system.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path for report')
def report(output):
    """Generate comprehensive system report."""
    try:
        click.echo("📊 Generating comprehensive system report...")
        
        checker = CompatibilityChecker()
        
        # Generate report
        if output:
            report_path = checker.generate_system_report(output)
        else:
            report_path = checker.generate_system_report()
        
        click.echo(f"✅ System report generated: {report_path}")
        
        # Display summary
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        report_content = report_data.get('huaju4k_system_compatibility_report', {})
        summary = report_content.get('summary', {})
        
        if summary:
            click.echo(f"\n📋 Report Summary:")
            click.echo(f"  Compatibility Score: {summary.get('compatibility_score', 0):.1f}/10")
            click.echo(f"  Compatibility Level: {summary.get('compatibility_level', 'unknown').title()}")
            click.echo(f"  GPU Acceleration: {'✅' if summary.get('gpu_acceleration') else '❌'}")
            click.echo(f"  Missing Dependencies: {summary.get('missing_dependencies', 0)}")
            click.echo(f"  Production Ready: {'✅' if summary.get('ready_for_production') else '❌'}")
        
    except Exception as e:
        click.echo(f"❌ Report generation failed: {str(e)}", err=True)


@system.command()
def validate():
    """Validate huaju4k installation."""
    try:
        click.echo("🔍 Validating huaju4k installation...")
        
        checker = CompatibilityChecker()
        results = checker.validate_installation()
        
        if 'error' in results:
            click.echo(f"❌ Installation validation failed: {results['error']}", err=True)
            return
        
        # Display overall status
        overall_status = "✅ VALID" if results['overall_status'] == 'passed' else "❌ INVALID"
        click.echo(f"\n🎯 Installation Status: {overall_status}")
        
        # Display detailed results
        for category, result in results.items():
            if category == 'overall_status':
                continue
            
            if isinstance(result, dict) and 'passed' in result:
                status = "✅" if result['passed'] else "❌"
                click.echo(f"\n{status} {category.title()}:")
                
                if 'issues' in result and result['issues']:
                    for issue in result['issues']:
                        click.echo(f"    ⚠️  {issue}")
                
                if category == 'core_modules' and 'modules' in result:
                    for module, available in result['modules'].items():
                        module_status = "✅" if available else "❌"
                        click.echo(f"    {module_status} {module}")
        
    except Exception as e:
        click.echo(f"❌ Installation validation failed: {str(e)}", err=True)


def _display_text_results(results):
    """Display results in text format."""
    summary = results.get('summary', {})
    
    click.echo(f"\n🎯 System Compatibility Results")
    click.echo("=" * 50)
    
    # Compatibility score
    score = summary.get('compatibility_score', 0)
    level = summary.get('compatibility_level', 'unknown')
    click.echo(f"📊 Compatibility Score: {score:.1f}/10 ({level.title()})")
    
    # System info
    system_info = results.get('system_info', {})
    click.echo(f"💻 Platform: {system_info.get('platform', 'Unknown')}")
    click.echo(f"🧠 CPU Cores: {system_info.get('cpu_count', 0)}")
    click.echo(f"💾 Memory: {system_info.get('total_memory_gb', 0):.1f} GB")
    
    # GPU info
    gpu_analysis = results.get('gpu_analysis', {})
    gpu_status = "✅ Available" if gpu_analysis.get('nvidia_driver_available') else "❌ Not Available"
    click.echo(f"🎮 NVIDIA GPU: {gpu_status}")
    
    if gpu_analysis.get('cuda_available'):
        click.echo(f"⚡ CUDA: ✅ Version {gpu_analysis.get('cuda_version', 'Unknown')}")
    else:
        click.echo(f"⚡ CUDA: ❌ Not Available")
    
    # Dependencies
    dependencies = results.get('dependencies', {})
    installed_count = sum(1 for dep in dependencies.values() if dep.get('installed', False))
    total_count = len(dependencies)
    click.echo(f"📦 Dependencies: {installed_count}/{total_count} installed")
    
    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        click.echo(f"\n💡 Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
            click.echo(f"  {i}. {rec}")
        
        if len(recommendations) > 5:
            click.echo(f"  ... and {len(recommendations) - 5} more")


def _display_json_results(results):
    """Display results in JSON format."""
    click.echo(json.dumps(results, indent=2, default=str))


def _format_text_report(results):
    """Format results as text report."""
    lines = []
    lines.append("HUAJU4K SYSTEM COMPATIBILITY REPORT")
    lines.append("=" * 50)
    lines.append(f"Generated: {results.get('timestamp', 'Unknown')}")
    lines.append("")
    
    # Summary
    summary = results.get('summary', {})
    lines.append("SUMMARY")
    lines.append("-" * 20)
    lines.append(f"Compatibility Score: {summary.get('compatibility_score', 0):.1f}/10")
    lines.append(f"Compatibility Level: {summary.get('compatibility_level', 'unknown').title()}")
    lines.append(f"GPU Acceleration: {'Yes' if summary.get('gpu_acceleration') else 'No'}")
    lines.append(f"Production Ready: {'Yes' if summary.get('ready_for_production') else 'No'}")
    lines.append("")
    
    # System Information
    system_info = results.get('system_info', {})
    lines.append("SYSTEM INFORMATION")
    lines.append("-" * 20)
    lines.append(f"Platform: {system_info.get('platform', 'Unknown')}")
    lines.append(f"Architecture: {system_info.get('architecture', 'Unknown')}")
    lines.append(f"CPU Cores: {system_info.get('cpu_count', 0)}")
    lines.append(f"Memory: {system_info.get('total_memory_gb', 0):.1f} GB")
    lines.append("")
    
    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 20)
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")
    
    return "\n".join(lines)


if __name__ == '__main__':
    system()