"""
Cross-platform optimization and consistency for huaju4k.

This module provides platform-specific optimizations, ensures cross-platform consistency,
and implements platform-aware performance tuning including NVIDIA GPU optimizations.

Implements Task 13.3:
- Cross-platform consistency testing and validation
- Platform-specific optimizations (including NVIDIA GPU optimization)
- Requirements: 10.5
"""

import os
import sys
import platform
import logging
import multiprocessing
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .system_detector import SystemDetector, SystemInfo

logger = logging.getLogger(__name__)


@dataclass
class PlatformConfig:
    """Platform-specific configuration."""
    
    platform_name: str
    file_separator: str
    path_separator: str
    temp_dir: str
    max_path_length: int
    supports_symlinks: bool
    case_sensitive_fs: bool
    
    # Performance settings
    default_workers: int
    memory_limit_factor: float
    gpu_memory_factor: float
    
    # Platform-specific paths
    config_dir: str
    cache_dir: str
    log_dir: str


class PlatformOptimizer(ABC):
    """Abstract base class for platform-specific optimizers."""
    
    @abstractmethod
    def optimize_performance_settings(self, system_info: SystemInfo) -> Dict[str, Any]:
        """Optimize performance settings for the platform."""
        pass
    
    @abstractmethod
    def get_optimal_paths(self) -> Dict[str, str]:
        """Get optimal paths for the platform."""
        pass
    
    @abstractmethod
    def configure_gpu_settings(self, gpu_info: Dict[str, Any]) -> Dict[str, Any]:
        """Configure GPU settings for the platform."""
        pass
    
    @abstractmethod
    def get_platform_limitations(self) -> List[str]:
        """Get platform-specific limitations."""
        pass


class WindowsOptimizer(PlatformOptimizer):
    """Windows-specific optimizer."""
    
    def optimize_performance_settings(self, system_info: SystemInfo) -> Dict[str, Any]:
        """Optimize performance settings for Windows."""
        settings = {
            'max_workers': min(system_info.cpu_count, 8),  # Windows handles threading well
            'memory_limit_gb': system_info.total_memory_gb * 0.7,  # Leave 30% for system
            'use_memory_mapping': True,  # Windows supports memory mapping well
            'chunk_size_mb': 64,  # Larger chunks work well on Windows
            'enable_gpu_scheduling': True,  # Windows GPU scheduling
            'priority_class': 'normal'  # Process priority
        }
        
        # Adjust for Windows version
        if 'Windows 11' in system_info.platform_version:
            settings['enable_gpu_scheduling'] = True
            settings['use_hardware_acceleration'] = True
        elif 'Windows 10' in system_info.platform_version:
            # Check build number for GPU scheduling support
            settings['enable_gpu_scheduling'] = True
        
        # Memory optimization for Windows
        if system_info.total_memory_gb >= 32:
            settings['large_page_support'] = True
            settings['memory_limit_gb'] = system_info.total_memory_gb * 0.8
        
        logger.debug(f"Windows optimization: {settings['max_workers']} workers, "
                    f"{settings['memory_limit_gb']:.1f}GB memory limit")
        
        return settings
    
    def get_optimal_paths(self) -> Dict[str, str]:
        """Get optimal paths for Windows."""
        import os
        
        # Use Windows-specific directories
        appdata = os.environ.get('APPDATA', os.path.expanduser('~\\AppData\\Roaming'))
        localappdata = os.environ.get('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local'))
        temp = os.environ.get('TEMP', os.path.expanduser('~\\AppData\\Local\\Temp'))
        
        return {
            'config_dir': os.path.join(appdata, 'huaju4k'),
            'cache_dir': os.path.join(localappdata, 'huaju4k', 'cache'),
            'log_dir': os.path.join(localappdata, 'huaju4k', 'logs'),
            'temp_dir': os.path.join(temp, 'huaju4k'),
            'models_dir': os.path.join(localappdata, 'huaju4k', 'models')
        }
    
    def configure_gpu_settings(self, gpu_info: Dict[str, Any]) -> Dict[str, Any]:
        """Configure GPU settings for Windows."""
        settings = {
            'enable_gpu_acceleration': gpu_info.get('nvidia_driver_available', False),
            'cuda_device_order': 'PCI_BUS_ID',  # Consistent device ordering
            'gpu_memory_growth': True,  # Allow dynamic memory allocation
            'mixed_precision': False,  # Disable by default for stability
            'gpu_memory_limit_mb': 0  # No limit by default
        }
        
        # Windows-specific NVIDIA optimizations
        if gpu_info.get('nvidia_driver_available'):
            settings.update({
                'enable_tcc_mode': False,  # Use WDDM mode on Windows
                'enable_mps': False,  # MPS not available on Windows
                'cuda_visible_devices': 'all',
                'gpu_scheduling_mode': 'compute'
            })
            
            # Memory management for Windows
            gpu_details = gpu_info.get('gpu_details', [])
            if gpu_details:
                total_gpu_memory = max(gpu['memory_total'] for gpu in gpu_details)
                # Reserve 500MB for Windows display
                settings['gpu_memory_limit_mb'] = total_gpu_memory - 500
        
        logger.debug(f"Windows GPU settings: acceleration={settings['enable_gpu_acceleration']}")
        return settings
    
    def get_platform_limitations(self) -> List[str]:
        """Get Windows-specific limitations."""
        return [
            "Path length limited to 260 characters (unless long path support enabled)",
            "Case-insensitive filesystem may cause issues with some models",
            "Windows Defender may slow down file operations",
            "GPU scheduling requires Windows 10 version 2004 or later"
        ]


class LinuxOptimizer(PlatformOptimizer):
    """Linux-specific optimizer."""
    
    def optimize_performance_settings(self, system_info: SystemInfo) -> Dict[str, Any]:
        """Optimize performance settings for Linux."""
        settings = {
            'max_workers': system_info.cpu_count,  # Linux handles many threads well
            'memory_limit_gb': system_info.total_memory_gb * 0.8,  # More aggressive memory use
            'use_memory_mapping': True,
            'chunk_size_mb': 128,  # Larger chunks for better I/O
            'enable_numa_optimization': True,  # NUMA awareness
            'use_transparent_hugepages': True,
            'priority_nice': 0  # Normal priority
        }
        
        # Check for specific Linux distributions
        try:
            with open('/etc/os-release', 'r') as f:
                os_info = f.read().lower()
                
            if 'ubuntu' in os_info:
                settings['distribution'] = 'ubuntu'
                settings['package_manager'] = 'apt'
            elif 'centos' in os_info or 'rhel' in os_info:
                settings['distribution'] = 'rhel'
                settings['package_manager'] = 'yum'
            elif 'arch' in os_info:
                settings['distribution'] = 'arch'
                settings['package_manager'] = 'pacman'
        except FileNotFoundError:
            settings['distribution'] = 'unknown'
        
        # Optimize for high-memory systems
        if system_info.total_memory_gb >= 64:
            settings['use_huge_pages'] = True
            settings['memory_limit_gb'] = system_info.total_memory_gb * 0.9
        
        logger.debug(f"Linux optimization: {settings['max_workers']} workers, "
                    f"{settings['memory_limit_gb']:.1f}GB memory limit")
        
        return settings
    
    def get_optimal_paths(self) -> Dict[str, str]:
        """Get optimal paths for Linux."""
        home = os.path.expanduser('~')
        
        # Follow XDG Base Directory Specification
        xdg_config_home = os.environ.get('XDG_CONFIG_HOME', os.path.join(home, '.config'))
        xdg_cache_home = os.environ.get('XDG_CACHE_HOME', os.path.join(home, '.cache'))
        xdg_data_home = os.environ.get('XDG_DATA_HOME', os.path.join(home, '.local', 'share'))
        
        return {
            'config_dir': os.path.join(xdg_config_home, 'huaju4k'),
            'cache_dir': os.path.join(xdg_cache_home, 'huaju4k'),
            'log_dir': os.path.join(xdg_cache_home, 'huaju4k', 'logs'),
            'temp_dir': '/tmp/huaju4k',
            'models_dir': os.path.join(xdg_data_home, 'huaju4k', 'models')
        }
    
    def configure_gpu_settings(self, gpu_info: Dict[str, Any]) -> Dict[str, Any]:
        """Configure GPU settings for Linux."""
        settings = {
            'enable_gpu_acceleration': gpu_info.get('nvidia_driver_available', False),
            'cuda_device_order': 'PCI_BUS_ID',
            'gpu_memory_growth': True,
            'mixed_precision': True,  # Better support on Linux
            'gpu_memory_limit_mb': 0
        }
        
        # Linux-specific NVIDIA optimizations
        if gpu_info.get('nvidia_driver_available'):
            settings.update({
                'enable_persistence_mode': True,  # Better for long-running processes
                'enable_mps': True,  # Multi-Process Service available on Linux
                'cuda_visible_devices': 'all',
                'gpu_scheduling_mode': 'compute',
                'enable_ecc': True  # Error correction if available
            })
            
            # Check for multiple GPUs
            gpu_count = gpu_info.get('gpu_count', 0)
            if gpu_count > 1:
                settings['multi_gpu_strategy'] = 'data_parallel'
                settings['gpu_memory_fraction'] = 0.9  # Use more memory per GPU
            
            # Memory optimization for Linux
            gpu_details = gpu_info.get('gpu_details', [])
            if gpu_details:
                total_gpu_memory = max(gpu['memory_total'] for gpu in gpu_details)
                # Linux can use more GPU memory
                settings['gpu_memory_limit_mb'] = int(total_gpu_memory * 0.95)
        
        logger.debug(f"Linux GPU settings: acceleration={settings['enable_gpu_acceleration']}")
        return settings
    
    def get_platform_limitations(self) -> List[str]:
        """Get Linux-specific limitations."""
        return [
            "May require manual installation of NVIDIA drivers",
            "Some distributions may have outdated packages",
            "Root access may be required for some optimizations",
            "Display server (X11/Wayland) may interfere with GPU compute"
        ]


class MacOSOptimizer(PlatformOptimizer):
    """macOS-specific optimizer."""
    
    def optimize_performance_settings(self, system_info: SystemInfo) -> Dict[str, Any]:
        """Optimize performance settings for macOS."""
        settings = {
            'max_workers': min(system_info.cpu_count, 6),  # Conservative for macOS
            'memory_limit_gb': system_info.total_memory_gb * 0.6,  # macOS uses more system memory
            'use_memory_mapping': True,
            'chunk_size_mb': 32,  # Smaller chunks for better responsiveness
            'enable_metal_acceleration': True,  # Use Metal for GPU compute
            'priority_nice': 0
        }
        
        # Check for Apple Silicon
        if platform.machine() == 'arm64':
            settings.update({
                'apple_silicon': True,
                'unified_memory': True,
                'memory_limit_gb': system_info.total_memory_gb * 0.7,  # Unified memory architecture
                'enable_neural_engine': True
            })
        else:
            settings['apple_silicon'] = False
            settings['unified_memory'] = False
        
        # Optimize for macOS version
        mac_version = platform.mac_ver()[0]
        if mac_version and float('.'.join(mac_version.split('.')[:2])) >= 12.0:
            settings['enable_async_io'] = True
            settings['use_grand_central_dispatch'] = True
        
        logger.debug(f"macOS optimization: {settings['max_workers']} workers, "
                    f"{settings['memory_limit_gb']:.1f}GB memory limit")
        
        return settings
    
    def get_optimal_paths(self) -> Dict[str, str]:
        """Get optimal paths for macOS."""
        home = os.path.expanduser('~')
        
        return {
            'config_dir': os.path.join(home, 'Library', 'Application Support', 'huaju4k'),
            'cache_dir': os.path.join(home, 'Library', 'Caches', 'huaju4k'),
            'log_dir': os.path.join(home, 'Library', 'Logs', 'huaju4k'),
            'temp_dir': '/tmp/huaju4k',
            'models_dir': os.path.join(home, 'Library', 'Application Support', 'huaju4k', 'models')
        }
    
    def configure_gpu_settings(self, gpu_info: Dict[str, Any]) -> Dict[str, Any]:
        """Configure GPU settings for macOS."""
        settings = {
            'enable_gpu_acceleration': False,  # CUDA not available on modern macOS
            'enable_metal_acceleration': True,  # Use Metal instead
            'unified_memory_optimization': platform.machine() == 'arm64',
            'gpu_memory_limit_mb': 0
        }
        
        # Apple Silicon optimizations
        if platform.machine() == 'arm64':
            settings.update({
                'enable_neural_engine': True,
                'metal_performance_shaders': True,
                'unified_memory_pool': True,
                'memory_bandwidth_optimization': True
            })
        
        # Intel Mac with discrete GPU
        elif gpu_info.get('gpu_count', 0) > 0:
            settings.update({
                'enable_opencl': True,
                'discrete_gpu_available': True
            })
        
        logger.debug(f"macOS GPU settings: Metal={settings['enable_metal_acceleration']}")
        return settings
    
    def get_platform_limitations(self) -> List[str]:
        """Get macOS-specific limitations."""
        limitations = [
            "CUDA not supported on macOS 10.14+ (use Metal/OpenCL instead)",
            "Gatekeeper may block unsigned binaries",
            "System Integrity Protection may limit some optimizations"
        ]
        
        if platform.machine() == 'arm64':
            limitations.extend([
                "Some x86-64 libraries may not be available",
                "Rosetta 2 translation may impact performance for x86 code"
            ])
        
        return limitations


class CrossPlatformManager:
    """
    Cross-platform consistency manager.
    
    Ensures consistent behavior across different platforms while applying
    platform-specific optimizations for best performance.
    """
    
    def __init__(self):
        """Initialize cross-platform manager."""
        self.system_detector = SystemDetector()
        self.current_platform = platform.system()
        
        # Initialize platform-specific optimizer
        self.optimizer = self._create_platform_optimizer()
        
        # Platform configuration
        self.platform_config = self._create_platform_config()
        
        logger.info(f"Cross-platform manager initialized for {self.current_platform}")
    
    def get_optimized_configuration(self) -> Dict[str, Any]:
        """
        Get optimized configuration for the current platform.
        
        Returns:
            Dictionary with platform-optimized configuration
        """
        try:
            # Detect system information
            system_info = self.system_detector.detect_system_info()
            
            # Get platform-specific performance settings
            performance_settings = self.optimizer.optimize_performance_settings(system_info)
            
            # Get optimal paths
            optimal_paths = self.optimizer.get_optimal_paths()
            
            # Get GPU configuration
            gpu_info = self.system_detector.check_nvidia_gpu_support()
            gpu_settings = self.optimizer.configure_gpu_settings(gpu_info)
            
            # Combine all settings
            config = {
                'platform': {
                    'name': self.current_platform,
                    'version': platform.version(),
                    'architecture': platform.machine()
                },
                'performance': performance_settings,
                'paths': optimal_paths,
                'gpu': gpu_settings,
                'system_info': {
                    'cpu_count': system_info.cpu_count,
                    'memory_gb': system_info.total_memory_gb,
                    'gpu_available': system_info.gpu_available,
                    'cuda_available': system_info.cuda_available
                },
                'limitations': self.optimizer.get_platform_limitations()
            }
            
            logger.info(f"Generated optimized configuration for {self.current_platform}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to generate optimized configuration: {e}")
            return self._get_fallback_configuration()
    
    def validate_cross_platform_consistency(self) -> Dict[str, Any]:
        """
        Validate cross-platform consistency.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'timestamp': self.system_detector.detect_system_info().dependencies_status,
            'platform': self.current_platform,
            'consistency_checks': {},
            'issues_found': [],
            'recommendations': []
        }
        
        try:
            # Check file path handling
            validation_results['consistency_checks']['file_paths'] = self._validate_file_paths()
            
            # Check dependency availability
            validation_results['consistency_checks']['dependencies'] = self._validate_dependencies()
            
            # Check GPU compatibility
            validation_results['consistency_checks']['gpu_support'] = self._validate_gpu_support()
            
            # Check performance consistency
            validation_results['consistency_checks']['performance'] = self._validate_performance_settings()
            
            # Collect issues and recommendations
            for check_name, check_result in validation_results['consistency_checks'].items():
                if not check_result.get('passed', False):
                    validation_results['issues_found'].extend(check_result.get('issues', []))
                    validation_results['recommendations'].extend(check_result.get('recommendations', []))
            
            # Overall validation status
            validation_results['overall_status'] = 'passed' if not validation_results['issues_found'] else 'failed'
            
            logger.info(f"Cross-platform validation: {validation_results['overall_status']}")
            
        except Exception as e:
            logger.error(f"Cross-platform validation failed: {e}")
            validation_results['overall_status'] = 'error'
            validation_results['issues_found'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def apply_platform_optimizations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply platform-specific optimizations to configuration.
        
        Args:
            config: Base configuration
            
        Returns:
            Optimized configuration
        """
        try:
            optimized_config = config.copy()
            
            # Get platform-specific optimizations
            platform_config = self.get_optimized_configuration()
            
            # Merge platform optimizations
            if 'performance' in platform_config:
                optimized_config.setdefault('performance', {}).update(platform_config['performance'])
            
            if 'gpu' in platform_config:
                optimized_config.setdefault('gpu', {}).update(platform_config['gpu'])
            
            if 'paths' in platform_config:
                optimized_config.setdefault('paths', {}).update(platform_config['paths'])
            
            # Apply platform-specific adjustments
            optimized_config = self._apply_platform_specific_adjustments(optimized_config)
            
            logger.info("Applied platform-specific optimizations")
            return optimized_config
            
        except Exception as e:
            logger.error(f"Failed to apply platform optimizations: {e}")
            return config
    
    def _create_platform_optimizer(self) -> PlatformOptimizer:
        """Create platform-specific optimizer."""
        if self.current_platform == 'Windows':
            return WindowsOptimizer()
        elif self.current_platform == 'Linux':
            return LinuxOptimizer()
        elif self.current_platform == 'Darwin':
            return MacOSOptimizer()
        else:
            logger.warning(f"Unsupported platform: {self.current_platform}, using Linux optimizer")
            return LinuxOptimizer()
    
    def _create_platform_config(self) -> PlatformConfig:
        """Create platform configuration."""
        if self.current_platform == 'Windows':
            return PlatformConfig(
                platform_name='Windows',
                file_separator='\\',
                path_separator=';',
                temp_dir=os.environ.get('TEMP', 'C:\\temp'),
                max_path_length=260,
                supports_symlinks=False,
                case_sensitive_fs=False,
                default_workers=multiprocessing.cpu_count(),
                memory_limit_factor=0.7,
                gpu_memory_factor=0.9,
                config_dir=os.path.join(os.environ.get('APPDATA', ''), 'huaju4k'),
                cache_dir=os.path.join(os.environ.get('LOCALAPPDATA', ''), 'huaju4k'),
                log_dir=os.path.join(os.environ.get('LOCALAPPDATA', ''), 'huaju4k', 'logs')
            )
        elif self.current_platform == 'Linux':
            return PlatformConfig(
                platform_name='Linux',
                file_separator='/',
                path_separator=':',
                temp_dir='/tmp',
                max_path_length=4096,
                supports_symlinks=True,
                case_sensitive_fs=True,
                default_workers=multiprocessing.cpu_count(),
                memory_limit_factor=0.8,
                gpu_memory_factor=0.95,
                config_dir=os.path.expanduser('~/.config/huaju4k'),
                cache_dir=os.path.expanduser('~/.cache/huaju4k'),
                log_dir=os.path.expanduser('~/.cache/huaju4k/logs')
            )
        else:  # Darwin (macOS)
            return PlatformConfig(
                platform_name='macOS',
                file_separator='/',
                path_separator=':',
                temp_dir='/tmp',
                max_path_length=1024,
                supports_symlinks=True,
                case_sensitive_fs=False,
                default_workers=min(multiprocessing.cpu_count(), 6),
                memory_limit_factor=0.6,
                gpu_memory_factor=0.8,
                config_dir=os.path.expanduser('~/Library/Application Support/huaju4k'),
                cache_dir=os.path.expanduser('~/Library/Caches/huaju4k'),
                log_dir=os.path.expanduser('~/Library/Logs/huaju4k')
            )
    
    def _validate_file_paths(self) -> Dict[str, Any]:
        """Validate file path handling consistency."""
        result = {'passed': True, 'issues': [], 'recommendations': []}
        
        try:
            # Test path operations
            test_paths = [
                'simple_file.txt',
                'path/with/subdirs/file.txt',
                'file with spaces.txt',
                'file-with-dashes.txt',
                'file_with_underscores.txt'
            ]
            
            for test_path in test_paths:
                # Test path normalization
                normalized = os.path.normpath(test_path)
                if not normalized:
                    result['issues'].append(f"Path normalization failed for: {test_path}")
                    result['passed'] = False
            
            # Test long path handling
            long_path = 'a' * (self.platform_config.max_path_length + 10)
            if len(long_path) > self.platform_config.max_path_length:
                if self.current_platform == 'Windows':
                    result['recommendations'].append("Enable long path support on Windows")
            
        except Exception as e:
            result['issues'].append(f"File path validation error: {str(e)}")
            result['passed'] = False
        
        return result
    
    def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate dependency availability."""
        result = {'passed': True, 'issues': [], 'recommendations': []}
        
        try:
            dependencies = self.system_detector.verify_dependencies()
            
            for dep_name, dep_info in dependencies.items():
                if dep_info.required and not dep_info.installed:
                    result['issues'].append(f"Required dependency missing: {dep_name}")
                    result['passed'] = False
                    if dep_info.install_command:
                        result['recommendations'].append(f"Install {dep_name}: {dep_info.install_command}")
        
        except Exception as e:
            result['issues'].append(f"Dependency validation error: {str(e)}")
            result['passed'] = False
        
        return result
    
    def _validate_gpu_support(self) -> Dict[str, Any]:
        """Validate GPU support consistency."""
        result = {'passed': True, 'issues': [], 'recommendations': []}
        
        try:
            gpu_info = self.system_detector.check_nvidia_gpu_support()
            
            if self.current_platform == 'Darwin' and gpu_info['cuda_available']:
                result['issues'].append("CUDA not supported on modern macOS")
                result['recommendations'].append("Use Metal acceleration instead of CUDA on macOS")
                result['passed'] = False
            
            if gpu_info['nvidia_driver_available'] and not gpu_info['cuda_available']:
                result['recommendations'].append("Install CUDA toolkit for GPU acceleration")
            
        except Exception as e:
            result['issues'].append(f"GPU validation error: {str(e)}")
            result['passed'] = False
        
        return result
    
    def _validate_performance_settings(self) -> Dict[str, Any]:
        """Validate performance settings consistency."""
        result = {'passed': True, 'issues': [], 'recommendations': []}
        
        try:
            system_info = self.system_detector.detect_system_info()
            
            # Check memory settings
            if system_info.total_memory_gb < 4:
                result['issues'].append("Insufficient memory for optimal performance")
                result['recommendations'].append("Upgrade to at least 8GB RAM")
                result['passed'] = False
            
            # Check CPU settings
            if system_info.cpu_count < 2:
                result['issues'].append("Insufficient CPU cores for parallel processing")
                result['recommendations'].append("Upgrade to multi-core processor")
                result['passed'] = False
        
        except Exception as e:
            result['issues'].append(f"Performance validation error: {str(e)}")
            result['passed'] = False
        
        return result
    
    def _apply_platform_specific_adjustments(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply platform-specific configuration adjustments."""
        if self.current_platform == 'Windows':
            # Windows-specific adjustments
            config.setdefault('file_handling', {})['use_long_paths'] = True
            config.setdefault('process', {})['priority'] = 'normal'
        
        elif self.current_platform == 'Linux':
            # Linux-specific adjustments
            config.setdefault('process', {})['nice_value'] = 0
            config.setdefault('memory', {})['use_huge_pages'] = True
        
        elif self.current_platform == 'Darwin':
            # macOS-specific adjustments
            config.setdefault('gpu', {})['prefer_metal'] = True
            config.setdefault('memory', {})['unified_memory'] = platform.machine() == 'arm64'
        
        return config
    
    def _get_fallback_configuration(self) -> Dict[str, Any]:
        """Get fallback configuration for error cases."""
        return {
            'platform': {
                'name': self.current_platform,
                'version': 'unknown',
                'architecture': platform.machine()
            },
            'performance': {
                'max_workers': 2,
                'memory_limit_gb': 4.0,
                'chunk_size_mb': 32
            },
            'paths': {
                'config_dir': os.path.expanduser('~/.huaju4k'),
                'cache_dir': os.path.expanduser('~/.huaju4k/cache'),
                'temp_dir': '/tmp/huaju4k'
            },
            'gpu': {
                'enable_gpu_acceleration': False
            },
            'limitations': ['Fallback configuration - limited functionality']
        }