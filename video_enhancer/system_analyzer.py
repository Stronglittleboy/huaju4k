"""
System Analysis Module for Video Enhancement

This module analyzes the current Windows system specifications, GPU capabilities,
WSL environment, and available disk space for video processing.
"""

import os
import platform
import subprocess
import json
import shutil
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class SystemSpecs:
    """System specifications data model"""
    total_ram_gb: float
    cpu_cores: int
    cpu_model: str
    gpu_model: Optional[str]
    gpu_memory_gb: Optional[float]
    cuda_version: Optional[str]
    wsl_available: bool
    storage_available_gb: float
    os_info: str
    python_version: str


class SystemAnalyzer:
    """Analyzes system capabilities for video enhancement processing"""
    
    def __init__(self):
        self.specs = None
    
    def analyze_system(self) -> SystemSpecs:
        """
        Perform comprehensive system analysis
        
        Returns:
            SystemSpecs: Complete system specification data
        """
        print("Starting system analysis...")
        
        # Analyze basic system info
        ram_gb = self._get_ram_info()
        cpu_cores, cpu_model = self._get_cpu_info()
        os_info = self._get_os_info()
        python_version = self._get_python_version()
        
        # Analyze GPU capabilities
        gpu_model, gpu_memory_gb, cuda_version = self._get_gpu_info()
        
        # Check WSL availability
        wsl_available = self._check_wsl_availability()
        
        # Check available disk space
        storage_available_gb = self._get_available_storage()
        
        self.specs = SystemSpecs(
            total_ram_gb=ram_gb,
            cpu_cores=cpu_cores,
            cpu_model=cpu_model,
            gpu_model=gpu_model,
            gpu_memory_gb=gpu_memory_gb,
            cuda_version=cuda_version,
            wsl_available=wsl_available,
            storage_available_gb=storage_available_gb,
            os_info=os_info,
            python_version=python_version
        )
        
        return self.specs
    
    def _get_ram_info(self) -> float:
        """Get total RAM in GB"""
        try:
            if platform.system() == "Windows":
                # Use wmic command on Windows
                result = subprocess.run(
                    ["wmic", "computersystem", "get", "TotalPhysicalMemory", "/value"],
                    capture_output=True, text=True, check=True
                )
                for line in result.stdout.split('\n'):
                    if 'TotalPhysicalMemory=' in line:
                        bytes_ram = int(line.split('=')[1].strip())
                        return round(bytes_ram / (1024**3), 2)  # Convert to GB
            else:
                # Linux/WSL method
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            kb_ram = int(line.split()[1])
                            return round(kb_ram / (1024**2), 2)  # Convert to GB
        except Exception as e:
            print(f"Warning: Could not determine RAM size: {e}")
            return 0.0
        
        return 0.0
    
    def _get_cpu_info(self) -> Tuple[int, str]:
        """Get CPU core count and model"""
        try:
            cpu_cores = os.cpu_count() or 0
            
            if platform.system() == "Windows":
                # Get CPU model on Windows
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name", "/value"],
                    capture_output=True, text=True, check=True
                )
                for line in result.stdout.split('\n'):
                    if 'Name=' in line:
                        cpu_model = line.split('=', 1)[1].strip()
                        return cpu_cores, cpu_model
            else:
                # Linux method
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            cpu_model = line.split(':', 1)[1].strip()
                            return cpu_cores, cpu_model
        except Exception as e:
            print(f"Warning: Could not determine CPU info: {e}")
            return os.cpu_count() or 0, "Unknown CPU"
        
        return os.cpu_count() or 0, "Unknown CPU"
    
    def _get_os_info(self) -> str:
        """Get operating system information"""
        try:
            return f"{platform.system()} {platform.release()} {platform.version()}"
        except Exception:
            return "Unknown OS"
    
    def _get_python_version(self) -> str:
        """Get Python version"""
        return platform.python_version()
    
    def _get_gpu_info(self) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        """Get GPU model, memory, and CUDA version"""
        gpu_model = None
        gpu_memory_gb = None
        cuda_version = None
        
        try:
            # Try nvidia-smi command
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            
            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        gpu_model = parts[0].strip()
                        gpu_memory_mb = float(parts[1].strip())
                        gpu_memory_gb = round(gpu_memory_mb / 1024, 2)
                        break
            
            # Try to get CUDA version
            try:
                cuda_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True, check=True
                )
                if cuda_result.stdout.strip():
                    cuda_version = cuda_result.stdout.strip().split('\n')[0]
            except Exception:
                pass
                
        except Exception as e:
            print(f"Warning: Could not detect NVIDIA GPU: {e}")
            
            # Try alternative methods for GPU detection
            try:
                if platform.system() == "Windows":
                    # Try wmic for GPU info on Windows
                    result = subprocess.run(
                        ["wmic", "path", "win32_VideoController", "get", "name", "/value"],
                        capture_output=True, text=True, check=True
                    )
                    for line in result.stdout.split('\n'):
                        if 'Name=' in line and line.split('=')[1].strip():
                            gpu_model = line.split('=')[1].strip()
                            break
            except Exception:
                pass
        
        return gpu_model, gpu_memory_gb, cuda_version
    
    def _check_wsl_availability(self) -> bool:
        """Check if WSL is available and functional"""
        try:
            # Try to get WSL version first
            result = subprocess.run(
                ["wsl", "--version"],
                capture_output=True, text=True, encoding='utf-16le', errors='ignore'
            )
            if result.returncode == 0:
                # WSL is available, now check for distributions
                list_result = subprocess.run(
                    ["wsl", "--list", "--verbose"],
                    capture_output=True, text=True, encoding='utf-16le', errors='ignore'
                )
                if list_result.returncode == 0:
                    # Check if Ubuntu is available
                    return "Ubuntu" in list_result.stdout
            return False
        except Exception:
            # Fallback to UTF-8 encoding
            try:
                result = subprocess.run(
                    ["wsl", "--list", "--verbose"],
                    capture_output=True, text=True, encoding='utf-8', errors='ignore'
                )
                return result.returncode == 0 and "Ubuntu" in result.stdout
            except Exception:
                return False
    
    def _get_available_storage(self) -> float:
        """Get available disk space in GB"""
        try:
            # Get current directory disk usage
            total, used, free = shutil.disk_usage(".")
            return round(free / (1024**3), 2)  # Convert to GB
        except Exception as e:
            print(f"Warning: Could not determine available storage: {e}")
            return 0.0
    
    def generate_report(self) -> str:
        """Generate a comprehensive system analysis report"""
        if not self.specs:
            self.analyze_system()
        
        report = []
        report.append("=" * 60)
        report.append("SYSTEM ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic System Info
        report.append("BASIC SYSTEM INFORMATION:")
        report.append(f"  Operating System: {self.specs.os_info}")
        report.append(f"  Python Version: {self.specs.python_version}")
        report.append("")
        
        # CPU Information
        report.append("CPU INFORMATION:")
        report.append(f"  Model: {self.specs.cpu_model}")
        report.append(f"  Cores: {self.specs.cpu_cores}")
        report.append("")
        
        # Memory Information
        report.append("MEMORY INFORMATION:")
        report.append(f"  Total RAM: {self.specs.total_ram_gb} GB")
        report.append("")
        
        # GPU Information
        report.append("GPU INFORMATION:")
        if self.specs.gpu_model:
            report.append(f"  Model: {self.specs.gpu_model}")
            if self.specs.gpu_memory_gb:
                report.append(f"  VRAM: {self.specs.gpu_memory_gb} GB")
            if self.specs.cuda_version:
                report.append(f"  Driver Version: {self.specs.cuda_version}")
        else:
            report.append("  No NVIDIA GPU detected")
        report.append("")
        
        # WSL Information
        report.append("WSL ENVIRONMENT:")
        report.append(f"  WSL Available: {'Yes' if self.specs.wsl_available else 'No'}")
        report.append("")
        
        # Storage Information
        report.append("STORAGE INFORMATION:")
        report.append(f"  Available Space: {self.specs.storage_available_gb} GB")
        report.append("")
        
        # Processing Recommendations
        report.append("PROCESSING RECOMMENDATIONS:")
        self._add_processing_recommendations(report)
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _add_processing_recommendations(self, report: list):
        """Add processing recommendations based on system specs"""
        if not self.specs:
            return
        
        # GPU recommendations
        if self.specs.gpu_model and "GTX 1650" in self.specs.gpu_model:
            report.append("  ✓ NVIDIA GTX 1650 detected - Good for AI upscaling")
            if self.specs.gpu_memory_gb and self.specs.gpu_memory_gb >= 4:
                report.append("  ✓ 4GB+ VRAM available - Suitable for Real-ESRGAN")
                report.append("  → Recommended tile size: 512-1024 pixels")
            else:
                report.append("  ⚠ Limited VRAM - Use smaller tile sizes (256-512 pixels)")
        elif self.specs.gpu_model:
            report.append(f"  ✓ GPU detected: {self.specs.gpu_model}")
            report.append("  → Check compatibility with CUDA/Real-ESRGAN")
        else:
            report.append("  ⚠ No NVIDIA GPU detected - CPU processing only")
            report.append("  → Processing will be significantly slower")
        
        # RAM recommendations
        if self.specs.total_ram_gb >= 16:
            report.append("  ✓ Sufficient RAM for video processing")
        else:
            report.append("  ⚠ Limited RAM - Monitor memory usage during processing")
        
        # Storage recommendations
        if self.specs.storage_available_gb >= 50:
            report.append("  ✓ Sufficient storage for 4K video processing")
        elif self.specs.storage_available_gb >= 20:
            report.append("  ⚠ Limited storage - Monitor disk space during processing")
        else:
            report.append("  ❌ Insufficient storage - Free up disk space before processing")
        
        # WSL recommendations
        if self.specs.wsl_available:
            report.append("  ✓ WSL available - Can use Linux-based AI tools")
            report.append("  → Recommended: Install Real-ESRGAN in WSL Ubuntu")
        else:
            report.append("  ⚠ WSL not available - Limited to Windows-native tools")
    
    def save_specs_to_file(self, filepath: str = "system_specs.json"):
        """Save system specifications to JSON file"""
        if not self.specs:
            self.analyze_system()
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self.specs), f, indent=2)
        
        print(f"System specifications saved to {filepath}")


def main():
    """Main function to run system analysis"""
    analyzer = SystemAnalyzer()
    
    print("Analyzing system for video enhancement capabilities...")
    print()
    
    # Perform analysis
    specs = analyzer.analyze_system()
    
    # Generate and display report
    report = analyzer.generate_report()
    print(report)
    
    # Save specifications to file
    analyzer.save_specs_to_file()
    
    return specs


if __name__ == "__main__":
    main()