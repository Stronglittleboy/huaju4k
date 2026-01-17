"""
System detection and hardware capability detection for huaju4k.

This module provides comprehensive system detection including hardware capabilities,
NVIDIA GPU detection, dependency checking, and compatibility reporting.

Implements Task 13.1:
- Hardware capability detection (including NVIDIA GPU detection)
- Dependency checking and verification
- Compatibility reporting
- Requirements: 10.1, 10.4
"""

import os
import sys
import platform
import subprocess
import logging
import psutil
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """System information container."""
    
    # Basic system info
    platform: str
    platform_version: str
    architecture: str
    python_version: str
    
    # Hardware info
    cpu_count: int
    cpu_model: str
    total_memory_gb: float
    available_memory_gb: float
    
    # GPU info
    gpu_available: bool
    gpu_count: int
    
    # Storage info
    disk_space_gb: float
    temp_space_gb: float
    
    # Fields with defaults (must come after non-default fields)
    gpu_models: List[str] = field(default_factory=list)
    gpu_memory_gb: List[float] = field(default_factory=list)
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    
    # Dependencies
    dependencies_status: Dict[str, bool] = field(default_factory=dict)
    missing_dependencies: List[str] = field(default_factory=list)
    
    # Compatibility
    compatibility_score: float = 0.0
    compatibility_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DependencyInfo:
    """Dependency information."""
    
    name: str
    required: bool
    installed: bool
    version: Optional[str] = None
    min_version: Optional[str] = None
    install_command: Optional[str] = None
    description: str = ""


class SystemDetector:
    """
    System detection and hardware capability detection.
    
    Provides comprehensive system analysis including hardware detection,
    NVIDIA GPU capabilities, dependency verification, and compatibility assessment.
    """
    
    def __init__(self):
        """Initialize system detector."""
        self.required_dependencies = self._define_required_dependencies()
        self.optional_dependencies = self._define_optional_dependencies()
        
        logger.info("System detector initialized")
    
    def detect_system_info(self) -> SystemInfo:
        """
        Detect comprehensive system information.
        
        Returns:
            SystemInfo object with complete system details
        """
        try:
            logger.info("Starting comprehensive system detection")
            
            # Basic system info
            system_info = SystemInfo(
                platform=platform.system(),
                platform_version=platform.version(),
                architecture=platform.machine(),
                python_version=platform.python_version()
            )
            
            # Hardware detection
            self._detect_hardware_info(system_info)
            
            # GPU detection (including NVIDIA)
            self._detect_gpu_capabilities(system_info)
            
            # Storage detection
            self._detect_storage_info(system_info)
            
            # Dependency checking
            self._check_dependencies(system_info)
            
            # Compatibility assessment
            self._assess_compatibility(system_info)
            
            logger.info(f"System detection completed - Compatibility score: {system_info.compatibility_score:.1f}/10")
            return system_info
            
        except Exception as e:
            logger.error(f"System detection failed: {e}")
            # Return minimal system info
            return SystemInfo(
                platform=platform.system(),
                platform_version="unknown",
                architecture=platform.machine(),
                python_version=platform.python_version(),
                cpu_count=1,
                cpu_model="unknown",
                total_memory_gb=0.0,
                available_memory_gb=0.0,
                gpu_available=False,
                gpu_count=0,
                disk_space_gb=0.0,
                temp_space_gb=0.0
            )
    
    def check_nvidia_gpu_support(self) -> Dict[str, Any]:
        """
        Comprehensive NVIDIA GPU support detection.
        
        Returns:
            Dictionary with detailed NVIDIA GPU information
        """
        gpu_info = {
            'nvidia_driver_available': False,
            'cuda_available': False,
            'cuda_version': None,
            'gpu_count': 0,
            'gpu_details': [],
            'opencv_cuda_support': False,
            'pytorch_cuda_support': False,
            'memory_total_mb': 0,
            'memory_available_mb': 0,
            'compute_capability': None,
            'driver_version': None
        }
        
        try:
            # Check NVIDIA driver
            gpu_info.update(self._check_nvidia_driver())
            
            # Check CUDA availability
            gpu_info.update(self._check_cuda_support())
            
            # Check OpenCV CUDA support
            gpu_info['opencv_cuda_support'] = self._check_opencv_cuda_support()
            
            # Check PyTorch CUDA support
            gpu_info['pytorch_cuda_support'] = self._check_pytorch_cuda_support()
            
            # Get detailed GPU information
            if gpu_info['nvidia_driver_available']:
                gpu_info['gpu_details'] = self._get_detailed_gpu_info()
            
            logger.info(f"NVIDIA GPU detection completed: {gpu_info['gpu_count']} GPUs found")
            
        except Exception as e:
            logger.error(f"NVIDIA GPU detection failed: {e}")
        
        return gpu_info
    
    def verify_dependencies(self) -> Dict[str, DependencyInfo]:
        """
        Verify all required and optional dependencies.
        
        Returns:
            Dictionary mapping dependency names to DependencyInfo objects
        """
        dependencies = {}
        
        # Check required dependencies
        for dep_name, dep_config in self.required_dependencies.items():
            dep_info = self._check_single_dependency(dep_name, dep_config, required=True)
            dependencies[dep_name] = dep_info
        
        # Check optional dependencies
        for dep_name, dep_config in self.optional_dependencies.items():
            dep_info = self._check_single_dependency(dep_name, dep_config, required=False)
            dependencies[dep_name] = dep_info
        
        # Log summary
        installed_count = sum(1 for dep in dependencies.values() if dep.installed)
        total_count = len(dependencies)
        logger.info(f"Dependency check completed: {installed_count}/{total_count} dependencies available")
        
        return dependencies
    
    def generate_compatibility_report(self, system_info: SystemInfo) -> Dict[str, Any]:
        """
        Generate comprehensive compatibility report.
        
        Args:
            system_info: System information
            
        Returns:
            Dictionary with compatibility report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_summary': {
                'platform': f"{system_info.platform} {system_info.platform_version}",
                'architecture': system_info.architecture,
                'python_version': system_info.python_version,
                'compatibility_score': system_info.compatibility_score
            },
            'hardware_capabilities': {
                'cpu_cores': system_info.cpu_count,
                'memory_gb': system_info.total_memory_gb,
                'gpu_available': system_info.gpu_available,
                'gpu_count': system_info.gpu_count,
                'cuda_support': system_info.cuda_available
            },
            'feature_support': self._assess_feature_support(system_info),
            'performance_expectations': self._estimate_performance(system_info),
            'recommendations': system_info.recommendations,
            'compatibility_issues': system_info.compatibility_issues,
            'installation_guide': self._generate_installation_guide(system_info)
        }
        
        return report
    
    def get_system_requirements(self) -> Dict[str, Any]:
        """
        Get system requirements for huaju4k.
        
        Returns:
            Dictionary with minimum and recommended system requirements
        """
        return {
            'minimum': {
                'os': ['Windows 10', 'Ubuntu 18.04+', 'macOS 10.15+'],
                'cpu': 'Dual-core processor',
                'memory_gb': 4,
                'storage_gb': 10,
                'python': '3.8+',
                'gpu': 'Optional (CPU processing supported)'
            },
            'recommended': {
                'os': ['Windows 11', 'Ubuntu 20.04+', 'macOS 12+'],
                'cpu': 'Quad-core processor or better',
                'memory_gb': 16,
                'storage_gb': 50,
                'python': '3.10+',
                'gpu': 'NVIDIA GPU with 4GB+ VRAM and CUDA support'
            },
            'optimal': {
                'os': ['Windows 11', 'Ubuntu 22.04+', 'macOS 13+'],
                'cpu': '8+ core processor',
                'memory_gb': 32,
                'storage_gb': 100,
                'python': '3.11+',
                'gpu': 'NVIDIA RTX series with 8GB+ VRAM'
            }
        }
    
    def _detect_hardware_info(self, system_info: SystemInfo) -> None:
        """Detect hardware information."""
        try:
            # CPU information
            system_info.cpu_count = psutil.cpu_count(logical=True)
            system_info.cpu_model = self._get_cpu_model()
            
            # Memory information
            memory = psutil.virtual_memory()
            system_info.total_memory_gb = memory.total / (1024**3)
            system_info.available_memory_gb = memory.available / (1024**3)
            
            logger.debug(f"Hardware detected: {system_info.cpu_count} cores, "
                        f"{system_info.total_memory_gb:.1f}GB RAM")
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            system_info.cpu_count = 1
            system_info.cpu_model = "unknown"
            system_info.total_memory_gb = 0.0
            system_info.available_memory_gb = 0.0
    
    def _detect_gpu_capabilities(self, system_info: SystemInfo) -> None:
        """Detect GPU capabilities including NVIDIA support."""
        try:
            nvidia_info = self.check_nvidia_gpu_support()
            
            system_info.gpu_available = nvidia_info['nvidia_driver_available']
            system_info.gpu_count = nvidia_info['gpu_count']
            system_info.cuda_available = nvidia_info['cuda_available']
            system_info.cuda_version = nvidia_info['cuda_version']
            
            # Extract GPU models and memory
            for gpu_detail in nvidia_info['gpu_details']:
                system_info.gpu_models.append(gpu_detail.get('name', 'Unknown GPU'))
                memory_mb = gpu_detail.get('memory_total', 0)
                system_info.gpu_memory_gb.append(memory_mb / 1024)
            
            logger.debug(f"GPU detection: {system_info.gpu_count} GPUs, CUDA: {system_info.cuda_available}")
            
        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
            system_info.gpu_available = False
            system_info.gpu_count = 0
            system_info.cuda_available = False
    
    def _detect_storage_info(self, system_info: SystemInfo) -> None:
        """Detect storage information."""
        try:
            # Current directory disk space
            current_disk = psutil.disk_usage('.')
            system_info.disk_space_gb = current_disk.free / (1024**3)
            
            # Temporary directory space
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_disk = psutil.disk_usage(temp_dir)
            system_info.temp_space_gb = temp_disk.free / (1024**3)
            
            logger.debug(f"Storage: {system_info.disk_space_gb:.1f}GB free, "
                        f"temp: {system_info.temp_space_gb:.1f}GB")
            
        except Exception as e:
            logger.error(f"Storage detection failed: {e}")
            system_info.disk_space_gb = 0.0
            system_info.temp_space_gb = 0.0
    
    def _check_dependencies(self, system_info: SystemInfo) -> None:
        """Check all dependencies."""
        try:
            dependencies = self.verify_dependencies()
            
            for dep_name, dep_info in dependencies.items():
                system_info.dependencies_status[dep_name] = dep_info.installed
                if dep_info.required and not dep_info.installed:
                    system_info.missing_dependencies.append(dep_name)
            
            logger.debug(f"Dependencies: {len(system_info.missing_dependencies)} missing")
            
        except Exception as e:
            logger.error(f"Dependency checking failed: {e}")
    
    def _assess_compatibility(self, system_info: SystemInfo) -> None:
        """Assess overall system compatibility."""
        try:
            score = 0.0
            max_score = 10.0
            
            # Platform compatibility (2 points)
            if system_info.platform in ['Windows', 'Linux', 'Darwin']:
                score += 2.0
            else:
                system_info.compatibility_issues.append(f"Unsupported platform: {system_info.platform}")
            
            # Memory adequacy (2 points)
            if system_info.total_memory_gb >= 16:
                score += 2.0
            elif system_info.total_memory_gb >= 8:
                score += 1.5
            elif system_info.total_memory_gb >= 4:
                score += 1.0
            else:
                system_info.compatibility_issues.append("Insufficient memory (< 4GB)")
            
            # CPU capability (2 points)
            if system_info.cpu_count >= 8:
                score += 2.0
            elif system_info.cpu_count >= 4:
                score += 1.5
            elif system_info.cpu_count >= 2:
                score += 1.0
            else:
                system_info.compatibility_issues.append("Insufficient CPU cores (< 2)")
            
            # GPU support (2 points)
            if system_info.gpu_available and system_info.cuda_available:
                score += 2.0
            elif system_info.gpu_available:
                score += 1.0
                system_info.recommendations.append("Install CUDA for better GPU performance")
            else:
                system_info.recommendations.append("Consider GPU upgrade for better performance")
            
            # Dependencies (2 points)
            required_deps = len([d for d in self.required_dependencies.keys()])
            available_deps = len([d for d, status in system_info.dependencies_status.items() 
                                if status and d in self.required_dependencies])
            
            if required_deps > 0:
                dep_score = (available_deps / required_deps) * 2.0
                score += dep_score
                
                if available_deps < required_deps:
                    missing_count = required_deps - available_deps
                    system_info.compatibility_issues.append(f"{missing_count} required dependencies missing")
            
            system_info.compatibility_score = min(score, max_score)
            
            # Generate recommendations based on score
            if system_info.compatibility_score < 5.0:
                system_info.recommendations.append("System may not meet minimum requirements")
            elif system_info.compatibility_score < 7.0:
                system_info.recommendations.append("Consider hardware upgrades for better performance")
            
            logger.info(f"Compatibility assessment: {system_info.compatibility_score:.1f}/10")
            
        except Exception as e:
            logger.error(f"Compatibility assessment failed: {e}")
            system_info.compatibility_score = 0.0
    
    def _check_nvidia_driver(self) -> Dict[str, Any]:
        """Check NVIDIA driver availability."""
        driver_info = {
            'nvidia_driver_available': False,
            'driver_version': None
        }
        
        try:
            # Try nvidia-smi command
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                driver_info['nvidia_driver_available'] = True
                driver_info['driver_version'] = result.stdout.strip().split('\n')[0]
                logger.debug(f"NVIDIA driver detected: {driver_info['driver_version']}")
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("NVIDIA driver not detected")
        
        return driver_info
    
    def _check_cuda_support(self) -> Dict[str, Any]:
        """Check CUDA support."""
        cuda_info = {
            'cuda_available': False,
            'cuda_version': None
        }
        
        try:
            # Try PyTorch CUDA check
            import torch
            if torch.cuda.is_available():
                cuda_info['cuda_available'] = True
                cuda_info['cuda_version'] = torch.version.cuda
                logger.debug(f"CUDA detected via PyTorch: {cuda_info['cuda_version']}")
                return cuda_info
        except ImportError:
            pass
        
        try:
            # Try nvcc command
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse CUDA version from nvcc output
                output = result.stdout
                for line in output.split('\n'):
                    if 'release' in line.lower():
                        # Extract version number
                        import re
                        version_match = re.search(r'release (\d+\.\d+)', line)
                        if version_match:
                            cuda_info['cuda_available'] = True
                            cuda_info['cuda_version'] = version_match.group(1)
                            logger.debug(f"CUDA detected via nvcc: {cuda_info['cuda_version']}")
                            break
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("CUDA not detected")
        
        return cuda_info
    
    def _check_opencv_cuda_support(self) -> bool:
        """Check if OpenCV has CUDA support."""
        try:
            import cv2
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except (ImportError, AttributeError):
            return False
    
    def _check_pytorch_cuda_support(self) -> bool:
        """Check if PyTorch has CUDA support."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _get_detailed_gpu_info(self) -> List[Dict[str, Any]]:
        """Get detailed GPU information."""
        gpu_details = []
        
        try:
            # Try nvidia-ml-py
            try:
                import pynvml
                pynvml.nvmlInit()
                
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpu_details.append({
                        'index': i,
                        'name': name,
                        'memory_total': memory_info.total // (1024 * 1024),  # MB
                        'memory_free': memory_info.free // (1024 * 1024),   # MB
                        'memory_used': memory_info.used // (1024 * 1024)    # MB
                    })
                
                pynvml.nvmlShutdown()
                return gpu_details
                
            except ImportError:
                pass
            
            # Fallback to nvidia-smi
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total,memory.free,memory.used',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 5:
                            gpu_details.append({
                                'index': int(parts[0]),
                                'name': parts[1],
                                'memory_total': int(parts[2]),
                                'memory_free': int(parts[3]),
                                'memory_used': int(parts[4])
                            })
        
        except Exception as e:
            logger.debug(f"Detailed GPU info failed: {e}")
        
        return gpu_details
    
    def _get_cpu_model(self) -> str:
        """Get CPU model name."""
        try:
            if platform.system() == "Windows":
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                   r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                winreg.CloseKey(key)
                return cpu_name.strip()
            
            elif platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            return line.split(':')[1].strip()
            
            elif platform.system() == "Darwin":  # macOS
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
        
        except Exception:
            pass
        
        return f"{platform.processor()} ({platform.machine()})"
    
    def _define_required_dependencies(self) -> Dict[str, Dict[str, Any]]:
        """Define required dependencies."""
        return {
            'numpy': {
                'min_version': '1.19.0',
                'install_command': 'pip install numpy',
                'description': 'Numerical computing library'
            },
            'opencv-python': {
                'min_version': '4.5.0',
                'install_command': 'pip install opencv-python',
                'description': 'Computer vision library'
            },
            'pillow': {
                'min_version': '8.0.0',
                'install_command': 'pip install pillow',
                'description': 'Image processing library'
            },
            'psutil': {
                'min_version': '5.7.0',
                'install_command': 'pip install psutil',
                'description': 'System monitoring library'
            },
            'click': {
                'min_version': '7.0',
                'install_command': 'pip install click',
                'description': 'Command line interface library'
            }
        }
    
    def _define_optional_dependencies(self) -> Dict[str, Dict[str, Any]]:
        """Define optional dependencies."""
        return {
            'torch': {
                'min_version': '1.9.0',
                'install_command': 'pip install torch',
                'description': 'Deep learning framework (for AI models)'
            },
            'librosa': {
                'min_version': '0.8.0',
                'install_command': 'pip install librosa',
                'description': 'Audio processing library'
            },
            'soundfile': {
                'min_version': '0.10.0',
                'install_command': 'pip install soundfile',
                'description': 'Audio file I/O library'
            },
            'ffmpeg-python': {
                'min_version': '0.2.0',
                'install_command': 'pip install ffmpeg-python',
                'description': 'Video processing library'
            },
            'realesrgan': {
                'min_version': '0.2.0',
                'install_command': 'pip install realesrgan',
                'description': 'AI upscaling library'
            }
        }
    
    def _check_single_dependency(self, dep_name: str, dep_config: Dict[str, Any], 
                                required: bool) -> DependencyInfo:
        """Check a single dependency."""
        dep_info = DependencyInfo(
            name=dep_name,
            required=required,
            installed=False,
            min_version=dep_config.get('min_version'),
            install_command=dep_config.get('install_command'),
            description=dep_config.get('description', '')
        )
        
        try:
            # Try to import the module
            if dep_name == 'opencv-python':
                import cv2
                dep_info.version = cv2.__version__
            elif dep_name == 'ffmpeg-python':
                import ffmpeg
                dep_info.version = getattr(ffmpeg, '__version__', 'unknown')
            else:
                module = __import__(dep_name.replace('-', '_'))
                dep_info.version = getattr(module, '__version__', 'unknown')
            
            dep_info.installed = True
            
            # Check version if specified
            if dep_info.min_version and dep_info.version != 'unknown':
                if self._compare_versions(dep_info.version, dep_info.min_version) < 0:
                    logger.warning(f"{dep_name} version {dep_info.version} is below minimum {dep_info.min_version}")
            
        except ImportError:
            logger.debug(f"Dependency {dep_name} not installed")
        
        return dep_info
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings. Returns -1, 0, or 1."""
        try:
            from packaging import version
            v1 = version.parse(version1)
            v2 = version.parse(version2)
            
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
            else:
                return 0
        except ImportError:
            # Fallback to simple string comparison
            v1_parts = [int(x) for x in version1.split('.') if x.isdigit()]
            v2_parts = [int(x) for x in version2.split('.') if x.isdigit()]
            
            # Pad shorter version with zeros
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for i in range(max_len):
                if v1_parts[i] < v2_parts[i]:
                    return -1
                elif v1_parts[i] > v2_parts[i]:
                    return 1
            
            return 0
    
    def _assess_feature_support(self, system_info: SystemInfo) -> Dict[str, str]:
        """Assess feature support based on system capabilities."""
        features = {}
        
        # Video processing
        if system_info.total_memory_gb >= 8:
            features['video_processing'] = 'full'
        elif system_info.total_memory_gb >= 4:
            features['video_processing'] = 'limited'
        else:
            features['video_processing'] = 'basic'
        
        # AI upscaling
        if system_info.gpu_available and system_info.cuda_available:
            features['ai_upscaling'] = 'gpu_accelerated'
        elif system_info.total_memory_gb >= 16:
            features['ai_upscaling'] = 'cpu_only'
        else:
            features['ai_upscaling'] = 'not_recommended'
        
        # Audio processing
        if 'librosa' in system_info.dependencies_status and system_info.dependencies_status['librosa']:
            features['audio_processing'] = 'full'
        else:
            features['audio_processing'] = 'basic'
        
        # Batch processing
        if system_info.cpu_count >= 4 and system_info.total_memory_gb >= 8:
            features['batch_processing'] = 'efficient'
        else:
            features['batch_processing'] = 'sequential'
        
        return features
    
    def _estimate_performance(self, system_info: SystemInfo) -> Dict[str, str]:
        """Estimate performance expectations."""
        performance = {}
        
        # Processing speed
        if system_info.gpu_available and system_info.cuda_available:
            if system_info.gpu_memory_gb and max(system_info.gpu_memory_gb) >= 8:
                performance['speed'] = 'very_fast'
            elif system_info.gpu_memory_gb and max(system_info.gpu_memory_gb) >= 4:
                performance['speed'] = 'fast'
            else:
                performance['speed'] = 'moderate'
        elif system_info.cpu_count >= 8:
            performance['speed'] = 'moderate'
        else:
            performance['speed'] = 'slow'
        
        # Memory usage
        if system_info.total_memory_gb >= 32:
            performance['memory'] = 'excellent'
        elif system_info.total_memory_gb >= 16:
            performance['memory'] = 'good'
        elif system_info.total_memory_gb >= 8:
            performance['memory'] = 'adequate'
        else:
            performance['memory'] = 'limited'
        
        # Stability
        missing_deps = len(system_info.missing_dependencies)
        if missing_deps == 0:
            performance['stability'] = 'high'
        elif missing_deps <= 2:
            performance['stability'] = 'medium'
        else:
            performance['stability'] = 'low'
        
        return performance
    
    def _generate_installation_guide(self, system_info: SystemInfo) -> List[str]:
        """Generate installation guide based on system info."""
        guide = []
        
        # Platform-specific instructions
        if system_info.platform == 'Windows':
            guide.append("1. Install Python 3.8+ from python.org")
            guide.append("2. Install Visual Studio Build Tools for C++ compilation")
        elif system_info.platform == 'Linux':
            guide.append("1. Update package manager: sudo apt update")
            guide.append("2. Install Python development headers: sudo apt install python3-dev")
        elif system_info.platform == 'Darwin':
            guide.append("1. Install Xcode Command Line Tools: xcode-select --install")
            guide.append("2. Install Homebrew for package management")
        
        # Missing dependencies
        if system_info.missing_dependencies:
            guide.append("3. Install missing dependencies:")
            for dep in system_info.missing_dependencies:
                if dep in self.required_dependencies:
                    install_cmd = self.required_dependencies[dep].get('install_command')
                    if install_cmd:
                        guide.append(f"   {install_cmd}")
        
        # GPU setup
        if system_info.gpu_available and not system_info.cuda_available:
            guide.append("4. Install CUDA toolkit for GPU acceleration:")
            guide.append("   Download from developer.nvidia.com/cuda-downloads")
        
        # Performance optimization
        if system_info.total_memory_gb < 16:
            guide.append("5. Consider increasing virtual memory/swap space")
        
        return guide