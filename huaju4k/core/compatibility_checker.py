"""
Comprehensive system compatibility checker for huaju4k.

This module provides a unified interface for system detection, compatibility checking,
and platform optimization. It integrates all Task 13 components.

Implements Task 13 (Complete):
- System detection and hardware capability detection
- Cross-platform consistency validation
- Platform-specific optimizations
- Comprehensive compatibility reporting
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from .system_detector import SystemDetector, SystemInfo
from .platform_optimizer import CrossPlatformManager

logger = logging.getLogger(__name__)


class CompatibilityChecker:
    """
    Comprehensive system compatibility checker.
    
    Provides unified interface for system detection, compatibility validation,
    and platform optimization for huaju4k video enhancement system.
    """
    
    def __init__(self):
        """Initialize compatibility checker."""
        self.system_detector = SystemDetector()
        self.platform_manager = CrossPlatformManager()
        
        logger.info("Compatibility checker initialized")
    
    def run_full_compatibility_check(self) -> Dict[str, Any]:
        """
        Run comprehensive compatibility check.
        
        Returns:
            Dictionary with complete compatibility analysis
        """
        try:
            logger.info("Starting full compatibility check")
            
            # System detection
            system_info = self.system_detector.detect_system_info()
            
            # Platform optimization
            platform_config = self.platform_manager.get_optimized_configuration()
            
            # Cross-platform validation
            validation_results = self.platform_manager.validate_cross_platform_consistency()
            
            # GPU analysis
            gpu_info = self.system_detector.check_nvidia_gpu_support()
            
            # Dependency verification
            dependencies = self.system_detector.verify_dependencies()
            
            # Generate compatibility report
            compatibility_report = self.system_detector.generate_compatibility_report(system_info)
            
            # Compile comprehensive results
            results = {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'system_info': self._serialize_system_info(system_info),
                'platform_config': platform_config,
                'validation_results': validation_results,
                'gpu_analysis': gpu_info,
                'dependencies': self._serialize_dependencies(dependencies),
                'compatibility_report': compatibility_report,
                'recommendations': self._generate_comprehensive_recommendations(
                    system_info, gpu_info, dependencies, validation_results
                ),
                'summary': self._generate_compatibility_summary(
                    system_info, validation_results, gpu_info
                )
            }
            
            logger.info(f"Compatibility check completed - Score: {system_info.compatibility_score:.1f}/10")
            return results
            
        except Exception as e:
            logger.error(f"Compatibility check failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            }
    
    def check_minimum_requirements(self) -> Dict[str, Any]:
        """
        Check if system meets minimum requirements.
        
        Returns:
            Dictionary with requirement check results
        """
        try:
            system_info = self.system_detector.detect_system_info()
            requirements = self.system_detector.get_system_requirements()
            
            min_req = requirements['minimum']
            
            checks = {
                'memory': {
                    'required': min_req['memory_gb'],
                    'available': system_info.total_memory_gb,
                    'passed': system_info.total_memory_gb >= min_req['memory_gb']
                },
                'storage': {
                    'required': min_req['storage_gb'],
                    'available': system_info.disk_space_gb,
                    'passed': system_info.disk_space_gb >= min_req['storage_gb']
                },
                'python_version': {
                    'required': min_req['python'],
                    'available': system_info.python_version,
                    'passed': self._check_python_version(system_info.python_version, min_req['python'])
                },
                'platform': {
                    'supported': min_req['os'],
                    'current': system_info.platform,
                    'passed': any(os_name.lower() in system_info.platform.lower() 
                                for os_name in ['windows', 'linux', 'darwin'])
                }
            }
            
            # Overall pass/fail
            overall_passed = all(check['passed'] for check in checks.values())
            
            return {
                'overall_passed': overall_passed,
                'checks': checks,
                'compatibility_level': 'minimum' if overall_passed else 'insufficient'
            }
            
        except Exception as e:
            logger.error(f"Minimum requirements check failed: {e}")
            return {
                'overall_passed': False,
                'error': str(e)
            }
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Get system-specific optimization recommendations.
        
        Returns:
            Dictionary with optimization recommendations
        """
        try:
            system_info = self.system_detector.detect_system_info()
            gpu_info = self.system_detector.check_nvidia_gpu_support()
            platform_config = self.platform_manager.get_optimized_configuration()
            
            recommendations = {
                'hardware': [],
                'software': [],
                'configuration': [],
                'performance': []
            }
            
            # Hardware recommendations
            if system_info.total_memory_gb < 16:
                recommendations['hardware'].append({
                    'type': 'memory_upgrade',
                    'priority': 'high',
                    'description': f'Upgrade RAM from {system_info.total_memory_gb:.1f}GB to 16GB+ for better performance',
                    'impact': 'Significantly improves processing speed and enables larger tile sizes'
                })
            
            if not system_info.gpu_available:
                recommendations['hardware'].append({
                    'type': 'gpu_upgrade',
                    'priority': 'medium',
                    'description': 'Add NVIDIA GPU with 4GB+ VRAM for AI acceleration',
                    'impact': 'Enables GPU-accelerated AI upscaling (10-50x faster)'
                })
            elif system_info.gpu_available and system_info.gpu_memory_gb:
                max_gpu_memory = max(system_info.gpu_memory_gb)
                if max_gpu_memory < 6:
                    recommendations['hardware'].append({
                        'type': 'gpu_memory',
                        'priority': 'medium',
                        'description': f'Upgrade GPU from {max_gpu_memory:.1f}GB to 6GB+ VRAM',
                        'impact': 'Allows larger tile sizes and batch processing'
                    })
            
            # Software recommendations
            if not system_info.cuda_available and system_info.gpu_available:
                recommendations['software'].append({
                    'type': 'cuda_installation',
                    'priority': 'high',
                    'description': 'Install CUDA toolkit for GPU acceleration',
                    'impact': 'Enables GPU processing capabilities'
                })
            
            for dep_name in system_info.missing_dependencies:
                recommendations['software'].append({
                    'type': 'dependency_installation',
                    'priority': 'high',
                    'description': f'Install missing dependency: {dep_name}',
                    'impact': 'Required for core functionality'
                })
            
            # Configuration recommendations
            if 'performance' in platform_config:
                perf_config = platform_config['performance']
                
                if system_info.cpu_count > perf_config.get('max_workers', 1):
                    recommendations['configuration'].append({
                        'type': 'worker_optimization',
                        'priority': 'medium',
                        'description': f'Increase worker threads to {system_info.cpu_count} for better CPU utilization',
                        'impact': 'Improves parallel processing performance'
                    })
            
            # Performance recommendations
            if system_info.compatibility_score < 7.0:
                recommendations['performance'].append({
                    'type': 'general_optimization',
                    'priority': 'medium',
                    'description': 'Consider system optimization for better performance',
                    'impact': 'Overall system performance improvement'
                })
            
            return {
                'recommendations': recommendations,
                'priority_summary': self._summarize_recommendation_priorities(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Optimization recommendations failed: {e}")
            return {'error': str(e)}
    
    def generate_system_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive system compatibility report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Path to generated report file
        """
        try:
            # Run full compatibility check
            results = self.run_full_compatibility_check()
            
            # Generate report content
            report_content = self._format_system_report(results)
            
            # Determine output path
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f'huaju4k_system_report_{timestamp}.json'
            
            # Save report
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_content, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"System report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"System report generation failed: {e}")
            raise
    
    def validate_installation(self) -> Dict[str, Any]:
        """
        Validate huaju4k installation completeness.
        
        Returns:
            Dictionary with installation validation results
        """
        try:
            validation = {
                'core_modules': self._validate_core_modules(),
                'dependencies': self._validate_installation_dependencies(),
                'configuration': self._validate_configuration_files(),
                'permissions': self._validate_file_permissions(),
                'overall_status': 'unknown'
            }
            
            # Determine overall status
            all_passed = all(
                result.get('passed', False) 
                for result in validation.values() 
                if isinstance(result, dict) and 'passed' in result
            )
            
            validation['overall_status'] = 'passed' if all_passed else 'failed'
            
            return validation
            
        except Exception as e:
            logger.error(f"Installation validation failed: {e}")
            return {
                'overall_status': 'error',
                'error': str(e)
            }
    
    def _serialize_system_info(self, system_info: SystemInfo) -> Dict[str, Any]:
        """Serialize SystemInfo object to dictionary."""
        return asdict(system_info)
    
    def _serialize_dependencies(self, dependencies: Dict) -> Dict[str, Any]:
        """Serialize dependencies to dictionary."""
        return {
            name: asdict(dep_info) 
            for name, dep_info in dependencies.items()
        }
    
    def _generate_comprehensive_recommendations(self, system_info: SystemInfo, 
                                             gpu_info: Dict[str, Any],
                                             dependencies: Dict,
                                             validation_results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations."""
        recommendations = []
        
        # System-based recommendations
        recommendations.extend(system_info.recommendations)
        
        # GPU-based recommendations
        if gpu_info.get('nvidia_driver_available') and not gpu_info.get('cuda_available'):
            recommendations.append("Install CUDA toolkit for GPU acceleration")
        
        if not gpu_info.get('opencv_cuda_support'):
            recommendations.append("Install OpenCV with CUDA support for better GPU performance")
        
        # Dependency-based recommendations
        missing_required = [
            name for name, dep in dependencies.items() 
            if dep.required and not dep.installed
        ]
        
        if missing_required:
            recommendations.append(f"Install missing required dependencies: {', '.join(missing_required)}")
        
        # Validation-based recommendations
        if validation_results.get('recommendations'):
            recommendations.extend(validation_results['recommendations'])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_compatibility_summary(self, system_info: SystemInfo,
                                      validation_results: Dict[str, Any],
                                      gpu_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate compatibility summary."""
        return {
            'compatibility_score': system_info.compatibility_score,
            'compatibility_level': self._get_compatibility_level(system_info.compatibility_score),
            'platform_supported': validation_results.get('overall_status') == 'passed',
            'gpu_acceleration': gpu_info.get('cuda_available', False),
            'missing_dependencies': len(system_info.missing_dependencies),
            'major_issues': len(system_info.compatibility_issues),
            'ready_for_production': (
                system_info.compatibility_score >= 7.0 and
                len(system_info.missing_dependencies) == 0 and
                validation_results.get('overall_status') == 'passed'
            )
        }
    
    def _get_compatibility_level(self, score: float) -> str:
        """Get compatibility level from score."""
        if score >= 9.0:
            return 'excellent'
        elif score >= 7.0:
            return 'good'
        elif score >= 5.0:
            return 'adequate'
        elif score >= 3.0:
            return 'poor'
        else:
            return 'insufficient'
    
    def _check_python_version(self, current: str, required: str) -> bool:
        """Check if Python version meets requirements."""
        try:
            current_parts = [int(x) for x in current.split('.')]
            required_parts = [int(x) for x in required.replace('+', '').split('.')]
            
            # Pad shorter version with zeros
            max_len = max(len(current_parts), len(required_parts))
            current_parts.extend([0] * (max_len - len(current_parts)))
            required_parts.extend([0] * (max_len - len(required_parts)))
            
            return current_parts >= required_parts
        except (ValueError, AttributeError):
            return False
    
    def _summarize_recommendation_priorities(self, recommendations: Dict[str, List]) -> Dict[str, int]:
        """Summarize recommendation priorities."""
        priority_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for category in recommendations.values():
            for rec in category:
                priority = rec.get('priority', 'low')
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        return priority_counts
    
    def _format_system_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format system report for output."""
        return {
            'huaju4k_system_compatibility_report': {
                'generated_at': results.get('timestamp'),
                'version': results.get('version'),
                'summary': results.get('summary', {}),
                'system_information': results.get('system_info', {}),
                'platform_configuration': results.get('platform_config', {}),
                'gpu_analysis': results.get('gpu_analysis', {}),
                'dependency_status': results.get('dependencies', {}),
                'validation_results': results.get('validation_results', {}),
                'recommendations': results.get('recommendations', []),
                'compatibility_report': results.get('compatibility_report', {})
            }
        }
    
    def _validate_core_modules(self) -> Dict[str, Any]:
        """Validate core huaju4k modules."""
        result = {'passed': True, 'modules': {}, 'issues': []}
        
        core_modules = [
            'huaju4k.core.video_enhancement_processor',
            'huaju4k.core.ai_model_manager',
            'huaju4k.core.theater_audio_enhancer',
            'huaju4k.core.memory_manager',
            'huaju4k.core.progress_tracker'
        ]
        
        for module_name in core_modules:
            try:
                __import__(module_name)
                result['modules'][module_name] = True
            except ImportError as e:
                result['modules'][module_name] = False
                result['issues'].append(f"Core module missing: {module_name} - {str(e)}")
                result['passed'] = False
        
        return result
    
    def _validate_installation_dependencies(self) -> Dict[str, Any]:
        """Validate installation dependencies."""
        dependencies = self.system_detector.verify_dependencies()
        
        required_missing = [
            name for name, dep in dependencies.items()
            if dep.required and not dep.installed
        ]
        
        return {
            'passed': len(required_missing) == 0,
            'missing_required': required_missing,
            'total_dependencies': len(dependencies),
            'installed_dependencies': len([d for d in dependencies.values() if d.installed])
        }
    
    def _validate_configuration_files(self) -> Dict[str, Any]:
        """Validate configuration files."""
        result = {'passed': True, 'files': {}, 'issues': []}
        
        # Check for default configuration files
        config_files = [
            'huaju4k/configs/default_config.yaml',
            'huaju4k/configs/presets/theater_presets.yaml'
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                result['files'][config_file] = True
            else:
                result['files'][config_file] = False
                result['issues'].append(f"Configuration file missing: {config_file}")
        
        # Configuration files are optional, so don't fail validation
        return result
    
    def _validate_file_permissions(self) -> Dict[str, Any]:
        """Validate file permissions."""
        result = {'passed': True, 'permissions': {}, 'issues': []}
        
        # Check write permissions for common directories
        test_dirs = [
            os.path.expanduser('~'),
            '/tmp' if os.name != 'nt' else os.environ.get('TEMP', 'C:\\temp')
        ]
        
        for test_dir in test_dirs:
            try:
                if os.path.exists(test_dir) and os.access(test_dir, os.W_OK):
                    result['permissions'][test_dir] = True
                else:
                    result['permissions'][test_dir] = False
                    result['issues'].append(f"No write permission for: {test_dir}")
                    result['passed'] = False
            except Exception as e:
                result['permissions'][test_dir] = False
                result['issues'].append(f"Permission check failed for {test_dir}: {str(e)}")
        
        return result