"""
Processing Strategy Module for Video Enhancement

This module designs optimal processing strategies based on video analysis results
and system capabilities, specifically for theater drama content.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path

from .models import ProcessingTool, ProcessingEnvironment


@dataclass
class ProcessingStrategy:
    """Complete processing strategy for video enhancement."""
    
    # Tool and model selection
    recommended_tool: ProcessingTool
    ai_model: str
    backup_model: str
    
    # Processing parameters
    scale_factor: int
    tile_size: int
    batch_size: int
    denoise_level: int
    
    # System optimization
    gpu_acceleration: bool
    memory_optimization: bool
    processing_environment: ProcessingEnvironment
    
    # Workflow configuration
    frame_extraction_format: str
    intermediate_cleanup: bool
    quality_check_enabled: bool
    
    # Performance estimates
    estimated_processing_time_hours: float
    estimated_memory_usage_gb: float
    estimated_output_size_gb: float
    
    # Processing steps
    processing_steps: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary format."""
        result = asdict(self)
        # Convert enums to strings for JSON serialization
        result['recommended_tool'] = self.recommended_tool.value
        result['processing_environment'] = self.processing_environment.value
        return result
    
    def save_to_file(self, filepath: str) -> None:
        """Save strategy to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class ProcessingStrategyDesigner:
    """Designs optimal processing strategies based on video analysis and system capabilities."""
    
    def __init__(self):
        """Initialize the strategy designer."""
        self.gpu_memory_gb = 4  # NVIDIA GTX 1650
        self.system_ram_gb = 24  # Available system RAM
        
    def design_strategy(self, video_metadata: Dict[str, Any], 
                       theater_analysis: Dict[str, Any],
                       system_specs: Optional[Dict[str, Any]] = None) -> ProcessingStrategy:
        """
        Design optimal processing strategy based on analysis results.
        
        Args:
            video_metadata: Basic video specifications
            theater_analysis: Theater-specific analysis results
            system_specs: System capabilities (optional)
            
        Returns:
            ProcessingStrategy with optimized parameters
        """
        
        # Extract key parameters
        resolution = video_metadata['resolution']
        duration = video_metadata['duration']
        lighting_type = theater_analysis['lighting_type']
        scene_complexity = theater_analysis['scene_complexity']
        
        # Tool selection based on content characteristics
        recommended_tool = self._select_optimal_tool(lighting_type, scene_complexity)
        
        # AI model selection
        ai_model, backup_model = self._select_ai_models(lighting_type, scene_complexity)
        
        # Calculate optimal processing parameters
        scale_factor = self._calculate_scale_factor(resolution)
        tile_size = self._calculate_optimal_tile_size(resolution, scene_complexity)
        batch_size = self._calculate_batch_size(tile_size)
        denoise_level = self._determine_denoise_level(lighting_type, scene_complexity)
        
        # System optimization settings
        gpu_acceleration = True  # GTX 1650 supports CUDA
        memory_optimization = self._requires_memory_optimization(resolution, duration)
        processing_environment = ProcessingEnvironment.WSL  # Real-ESRGAN works best in WSL
        
        # Workflow configuration
        frame_extraction_format = "png"  # Best quality for AI processing
        intermediate_cleanup = True  # Save disk space during processing
        quality_check_enabled = True  # Important for theater content
        
        # Performance estimates
        estimates = self._calculate_performance_estimates(
            resolution, duration, scale_factor, tile_size, scene_complexity
        )
        
        # Generate processing steps
        processing_steps = self._generate_processing_steps(
            recommended_tool, ai_model, scale_factor, tile_size
        )
        
        return ProcessingStrategy(
            recommended_tool=recommended_tool,
            ai_model=ai_model,
            backup_model=backup_model,
            scale_factor=scale_factor,
            tile_size=tile_size,
            batch_size=batch_size,
            denoise_level=denoise_level,
            gpu_acceleration=gpu_acceleration,
            memory_optimization=memory_optimization,
            processing_environment=processing_environment,
            frame_extraction_format=frame_extraction_format,
            intermediate_cleanup=intermediate_cleanup,
            quality_check_enabled=quality_check_enabled,
            estimated_processing_time_hours=estimates['processing_time_hours'],
            estimated_memory_usage_gb=estimates['memory_usage_gb'],
            estimated_output_size_gb=estimates['output_size_gb'],
            processing_steps=processing_steps
        )
    
    def _select_optimal_tool(self, lighting_type: str, scene_complexity: str) -> ProcessingTool:
        """Select the best tool based on content characteristics."""
        # For theater content with stage lighting, Real-ESRGAN is optimal
        if lighting_type == "stage" or scene_complexity == "complex":
            return ProcessingTool.REAL_ESRGAN
        elif lighting_type == "natural" and scene_complexity == "simple":
            return ProcessingTool.WAIFU2X_GUI  # Faster for simple content
        else:
            return ProcessingTool.REAL_ESRGAN  # Default robust choice
    
    def _select_ai_models(self, lighting_type: str, scene_complexity: str) -> tuple[str, str]:
        """Select primary and backup AI models."""
        if lighting_type == "stage":
            # Stage lighting benefits from Real-ESRGAN's robust training
            primary = "RealESRGAN_x4plus"
            backup = "RealESRNet_x4plus"
        elif scene_complexity == "complex":
            # Complex scenes need the most capable model
            primary = "RealESRGAN_x4plus"
            backup = "RealESRGAN_x4plus_anime_6B"
        else:
            # Default high-quality choice
            primary = "RealESRGAN_x4plus"
            backup = "RealESRNet_x4plus"
        
        return primary, backup
    
    def _calculate_scale_factor(self, resolution: tuple[int, int]) -> int:
        """Calculate required scale factor to reach 4K."""
        width, height = resolution
        
        # Target 4K resolution (3840x2160)
        target_width = 3840
        scale_factor = target_width / width
        
        # Round to nearest integer scale factor
        if scale_factor <= 1.5:
            return 1  # Already high resolution
        elif scale_factor <= 2.5:
            return 2
        elif scale_factor <= 3.5:
            return 3
        else:
            return 4
    
    def _calculate_optimal_tile_size(self, resolution: tuple[int, int], 
                                   scene_complexity: str) -> int:
        """Calculate optimal tile size based on GPU memory and content complexity."""
        width, height = resolution
        
        # Base tile size for 4GB GPU
        base_tile_size = 512
        
        # Adjust based on input resolution
        if width * height > 1920 * 1080:  # Higher than 1080p
            base_tile_size = 400
        elif width * height < 1280 * 720:  # Lower than 720p
            base_tile_size = 640
        
        # Adjust based on scene complexity
        if scene_complexity == "complex":
            base_tile_size = min(base_tile_size, 400)  # Reduce for complex scenes
        elif scene_complexity == "simple":
            base_tile_size = min(base_tile_size + 128, 640)  # Increase for simple scenes
        
        return base_tile_size
    
    def _calculate_batch_size(self, tile_size: int) -> int:
        """Calculate optimal batch size based on tile size and GPU memory."""
        # Conservative batch size for 4GB GPU
        if tile_size >= 512:
            return 1  # Process one tile at a time for large tiles
        elif tile_size >= 400:
            return 2
        else:
            return 4
    
    def _determine_denoise_level(self, lighting_type: str, scene_complexity: str) -> int:
        """Determine optimal denoising level."""
        if lighting_type == "stage":
            return 2  # Stage lighting often has more noise
        elif scene_complexity == "complex":
            return 1  # Moderate denoising for complex scenes
        else:
            return 1  # Default moderate denoising
    
    def _requires_memory_optimization(self, resolution: tuple[int, int], 
                                    duration: float) -> bool:
        """Determine if memory optimization is needed."""
        width, height = resolution
        total_pixels = width * height
        
        # Enable memory optimization for large videos or long duration
        return total_pixels > 1920 * 1080 or duration > 1800  # 30 minutes
    
    def _calculate_performance_estimates(self, resolution: tuple[int, int], 
                                       duration: float, scale_factor: int,
                                       tile_size: int, scene_complexity: str) -> Dict[str, float]:
        """Calculate performance estimates."""
        width, height = resolution
        total_frames = duration * 25  # 25 fps
        
        # Processing time estimation (very rough)
        # Base: ~0.5 seconds per frame for 1080p on GTX 1650
        base_time_per_frame = 0.5
        
        # Adjust for resolution
        resolution_factor = (width * height) / (1920 * 1080)
        
        # Adjust for scale factor
        scale_time_factor = scale_factor ** 1.5
        
        # Adjust for complexity
        complexity_factor = 1.0
        if scene_complexity == "complex":
            complexity_factor = 1.3
        elif scene_complexity == "simple":
            complexity_factor = 0.8
        
        # Adjust for tile size (smaller tiles = more overhead)
        tile_factor = 512 / tile_size
        
        time_per_frame = (base_time_per_frame * resolution_factor * 
                         scale_time_factor * complexity_factor * tile_factor)
        
        total_processing_time = (total_frames * time_per_frame) / 3600  # Convert to hours
        
        # Memory usage estimation
        # Base memory for processing + model loading
        base_memory = 2.0  # GB
        frame_memory = (width * height * 3 * 4) / (1024**3)  # RGB float32
        tile_memory = (tile_size * tile_size * 3 * 4 * 2) / (1024**3)  # Input + output
        
        memory_usage = base_memory + frame_memory + tile_memory
        
        # Output size estimation
        # Assume similar compression ratio but 4x pixels
        input_size_gb = 0.76  # From analysis: 762.6 MB
        output_size_gb = input_size_gb * (scale_factor ** 2) * 1.2  # 20% overhead
        
        return {
            'processing_time_hours': round(total_processing_time, 1),
            'memory_usage_gb': round(memory_usage, 1),
            'output_size_gb': round(output_size_gb, 1)
        }
    
    def _generate_processing_steps(self, tool: ProcessingTool, model: str,
                                 scale_factor: int, tile_size: int) -> List[str]:
        """Generate detailed processing steps."""
        steps = [
            "1. Prepare WSL Ubuntu environment with CUDA support",
            "2. Install Real-ESRGAN and download required AI models",
            f"3. Extract video frames using FFmpeg (PNG format)",
            "4. Create processing workspace and organize frames",
            f"5. Configure Real-ESRGAN with model: {model}",
            f"6. Set processing parameters: scale={scale_factor}x, tile_size={tile_size}",
            "7. Begin AI upscaling process with GPU acceleration",
            "8. Monitor progress and handle any processing errors",
            "9. Perform quality checks on sample enhanced frames",
            "10. Reassemble enhanced frames into 4K video",
            "11. Merge original audio track with enhanced video",
            "12. Generate final 4K output file",
            "13. Cleanup intermediate files and validate output",
            "14. Create processing report with settings and results"
        ]
        
        return steps
    
    def print_strategy_report(self, strategy: ProcessingStrategy, 
                            video_metadata: Dict[str, Any]) -> None:
        """Print a comprehensive strategy report."""
        print("\n" + "="*70)
        print("OPTIMAL PROCESSING STRATEGY")
        print("="*70)
        
        print(f"\nTOOL SELECTION:")
        print(f"  Recommended Tool: {strategy.recommended_tool.value}")
        print(f"  AI Model: {strategy.ai_model}")
        print(f"  Backup Model: {strategy.backup_model}")
        print(f"  Processing Environment: {strategy.processing_environment.value}")
        
        print(f"\nPROCESSING PARAMETERS:")
        print(f"  Scale Factor: {strategy.scale_factor}x")
        print(f"  Tile Size: {strategy.tile_size}px")
        print(f"  Batch Size: {strategy.batch_size}")
        print(f"  Denoise Level: {strategy.denoise_level}")
        print(f"  GPU Acceleration: {'Enabled' if strategy.gpu_acceleration else 'Disabled'}")
        print(f"  Memory Optimization: {'Enabled' if strategy.memory_optimization else 'Disabled'}")
        
        print(f"\nWORKFLOW CONFIGURATION:")
        print(f"  Frame Format: {strategy.frame_extraction_format.upper()}")
        print(f"  Intermediate Cleanup: {'Enabled' if strategy.intermediate_cleanup else 'Disabled'}")
        print(f"  Quality Checks: {'Enabled' if strategy.quality_check_enabled else 'Disabled'}")
        
        print(f"\nPERFORMANCE ESTIMATES:")
        print(f"  Processing Time: ~{strategy.estimated_processing_time_hours} hours")
        print(f"  Memory Usage: ~{strategy.estimated_memory_usage_gb} GB")
        print(f"  Output File Size: ~{strategy.estimated_output_size_gb} GB")
        
        # Calculate final resolution
        input_width, input_height = video_metadata['resolution']
        output_width = input_width * strategy.scale_factor
        output_height = input_height * strategy.scale_factor
        
        print(f"\nOUTPUT SPECIFICATIONS:")
        print(f"  Input Resolution: {input_width}x{input_height}")
        print(f"  Output Resolution: {output_width}x{output_height}")
        print(f"  Quality Improvement: {strategy.scale_factor**2}x pixel count")
        
        print(f"\nPROCESSING STEPS:")
        for step in strategy.processing_steps:
            print(f"  {step}")
        
        print("\n" + "="*70)


def main():
    """Example usage of the processing strategy designer."""
    # Example video metadata (from our analysis)
    video_metadata = {
        'resolution': (1920, 1080),
        'frame_rate': 25.0,
        'duration': 2319.96,
        'codec': 'h264',
        'bitrate': 2627731,
        'audio_tracks': 1,
        'file_size': 799604178
    }
    
    # Example theater analysis (from our analysis)
    theater_analysis = {
        'lighting_type': 'stage',
        'scene_complexity': 'simple',
        'avg_brightness': 23.0,
        'avg_complexity': 0.012,
        'frames_analyzed': 10,
        'recommended_model': 'realesrgan-x4plus'
    }
    
    # Design strategy
    designer = ProcessingStrategyDesigner()
    strategy = designer.design_strategy(video_metadata, theater_analysis)
    
    # Print report
    designer.print_strategy_report(strategy, video_metadata)
    
    # Save strategy to file
    strategy.save_to_file('processing_strategy.json')
    print(f"\n✓ Processing strategy saved to processing_strategy.json")


if __name__ == "__main__":
    main()