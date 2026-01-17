#!/usr/bin/env python3
"""
Execute AI Upscaling Process

This script implements task 4.2: Execute AI upscaling process
- Process video frames using selected AI model
- Monitor GPU memory usage and processing progress
- Handle batch processing with appropriate tile sizes
- Implement error handling and recovery for failed frames
"""

import os
import sys
import logging
import json
import time
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add video_enhancer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'video_enhancer'))

try:
    from config import ConfigManager
    from models import ProcessingConfig, ProcessingTool, ProcessingEnvironment
    from ai_upscaler import AIUpscaler
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback to direct imports
    sys.path.append('video_enhancer')
    from config import ConfigManager
    from models import ProcessingConfig, ProcessingTool, ProcessingEnvironment
    from ai_upscaler import AIUpscaler


class AIUpscalingExecutor:
    """Executes AI upscaling process with monitoring and error handling."""
    
    def __init__(self):
        """Initialize the AI upscaling executor."""
        self.logger = self._setup_logging()
        self.config_manager = ConfigManager()
        self.ai_upscaler = AIUpscaler(self.config_manager)
        
        # Processing state
        self.processing_stats = {}
        self.failed_frames = []
        self.recovery_attempts = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ai_upscaling_execution.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def load_processing_strategy(self, strategy_file: str = "theater_video_processing_strategy.json") -> Dict[str, Any]:
        """Load processing strategy from JSON file."""
        try:
            if os.path.exists(strategy_file):
                with open(strategy_file, 'r', encoding='utf-8') as f:
                    strategy = json.load(f)
                self.logger.info(f"Loaded processing strategy from {strategy_file}")
                return strategy
            else:
                self.logger.warning(f"Strategy file not found: {strategy_file}")
                return self._get_default_strategy()
        except Exception as e:
            self.logger.error(f"Failed to load strategy: {str(e)}")
            return self._get_default_strategy()
    
    def _get_default_strategy(self) -> Dict[str, Any]:
        """Get default processing strategy for theater video."""
        return {
            "recommended_tool": "real-esrgan",
            "ai_model": "RealESRGAN_x4plus",
            "scale_factor": 2,
            "tile_size": 640,
            "batch_size": 1,
            "denoise_level": 2,
            "gpu_acceleration": True,
            "memory_optimization": True
        }
    
    def create_processing_config(self, strategy: Dict[str, Any]) -> ProcessingConfig:
        """Create processing configuration from strategy."""
        return ProcessingConfig(
            tool=ProcessingTool.REAL_ESRGAN,
            model=strategy.get("ai_model", "RealESRGAN_x4plus"),
            scale_factor=strategy.get("scale_factor", 2),
            tile_size=strategy.get("tile_size", 640),
            gpu_acceleration=strategy.get("gpu_acceleration", True),
            denoise_level=strategy.get("denoise_level", 2),
            environment=ProcessingEnvironment.WSL,
            color_correction=True
        )
    
    def monitor_system_resources(self) -> Dict[str, float]:
        """Monitor current system resource usage."""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # GPU memory usage (if available)
            gpu_info = {}
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                    gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    gpu_info = {
                        "gpu_memory_allocated_gb": gpu_memory_allocated,
                        "gpu_memory_reserved_gb": gpu_memory_reserved,
                        "gpu_memory_total_gb": gpu_memory_total,
                        "gpu_memory_usage_percent": (gpu_memory_reserved / gpu_memory_total) * 100
                    }
            except ImportError:
                pass
            
            resources = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_available_gb": memory_available_gb,
                **gpu_info
            }
            
            return resources
            
        except Exception as e:
            self.logger.warning(f"Could not monitor system resources: {str(e)}")
            return {}
    
    def validate_input_frames(self, frames_dir: str) -> Tuple[bool, int, list]:
        """
        Validate input frames directory and return frame information.
        
        Returns:
            Tuple of (success, frame_count, frame_files)
        """
        try:
            if not os.path.exists(frames_dir):
                self.logger.error(f"Frames directory not found: {frames_dir}")
                return False, 0, []
            
            # Get list of frame files
            frame_files = sorted([f for f in os.listdir(frames_dir) 
                                if f.startswith('frame_') and f.endswith('.png')])
            
            if not frame_files:
                self.logger.error("No frame files found in input directory")
                return False, 0, []
            
            self.logger.info(f"Found {len(frame_files)} frames for processing")
            
            # Validate a sample of frames
            sample_indices = [0, len(frame_files)//2, len(frame_files)-1]
            for idx in sample_indices:
                frame_path = os.path.join(frames_dir, frame_files[idx])
                try:
                    import cv2
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        self.logger.error(f"Corrupted frame detected: {frame_files[idx]}")
                        return False, 0, []
                except Exception as e:
                    self.logger.error(f"Frame validation failed for {frame_files[idx]}: {str(e)}")
                    return False, 0, []
            
            self.logger.info("Frame validation passed")
            return True, len(frame_files), frame_files
            
        except Exception as e:
            self.logger.error(f"Frame validation failed: {str(e)}")
            return False, 0, []
    
    def execute_upscaling_process(self, input_frames_dir: str, output_frames_dir: str,
                                processing_config: ProcessingConfig) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the complete AI upscaling process with monitoring and error handling.
        
        Args:
            input_frames_dir: Directory containing input frames
            output_frames_dir: Directory for upscaled frames
            processing_config: Processing configuration
            
        Returns:
            Tuple of (success, processing_stats)
        """
        try:
            self.logger.info("="*70)
            self.logger.info("EXECUTING AI UPSCALING PROCESS")
            self.logger.info("="*70)
            
            # Step 1: Validate input frames
            self.logger.info("Step 1: Validating input frames...")
            success, frame_count, frame_files = self.validate_input_frames(input_frames_dir)
            if not success:
                return False, {}
            
            # Step 2: Create output directory
            os.makedirs(output_frames_dir, exist_ok=True)
            self.logger.info(f"Output directory: {output_frames_dir}")
            
            # Step 3: Initialize AI model
            self.logger.info("Step 2: Initializing AI upscaling model...")
            self.logger.info(f"Model: {processing_config.model}")
            self.logger.info(f"Scale factor: {processing_config.scale_factor}x")
            self.logger.info(f"Tile size: {processing_config.tile_size}")
            self.logger.info(f"GPU acceleration: {processing_config.gpu_acceleration}")
            
            model_success = self.ai_upscaler.initialize_model(
                processing_config.model, 
                processing_config.scale_factor
            )
            
            if not model_success:
                self.logger.error("AI model initialization failed")
                return False, {}
            
            # Get model information
            model_info = self.ai_upscaler.get_model_info()
            self.logger.info(f"Model initialized successfully: {model_info}")
            
            # Step 4: Monitor initial system resources
            self.logger.info("Step 3: Monitoring system resources...")
            initial_resources = self.monitor_system_resources()
            if initial_resources:
                self.logger.info(f"Initial CPU usage: {initial_resources.get('cpu_percent', 0):.1f}%")
                self.logger.info(f"Initial memory usage: {initial_resources.get('memory_percent', 0):.1f}%")
                if 'gpu_memory_usage_percent' in initial_resources:
                    self.logger.info(f"Initial GPU memory usage: {initial_resources['gpu_memory_usage_percent']:.1f}%")
            
            # Step 5: Execute upscaling with progress monitoring
            self.logger.info("Step 4: Starting AI upscaling process...")
            start_time = time.time()
            
            success, upscaling_stats = self.ai_upscaler.upscale_frames_batch(
                input_frames_dir, output_frames_dir, processing_config
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if not success:
                self.logger.error("AI upscaling process failed")
                # Try fallback method
                self.logger.info("Attempting fallback upscaling method...")
                success, upscaling_stats = self.ai_upscaler.upscale_with_fallback(
                    input_frames_dir, output_frames_dir, processing_config
                )
                
                if not success:
                    self.logger.error("Both primary and fallback upscaling methods failed")
                    return False, {}
            
            # Step 6: Validate output and handle failed frames
            self.logger.info("Step 5: Validating output and handling failed frames...")
            validation_success, final_stats = self._validate_and_recover_frames(
                input_frames_dir, output_frames_dir, frame_files, processing_config
            )
            
            # Merge statistics
            final_stats.update(upscaling_stats)
            final_stats['total_processing_time'] = total_time
            final_stats['processing_rate_fps'] = frame_count / total_time if total_time > 0 else 0
            
            # Step 7: Final resource monitoring
            final_resources = self.monitor_system_resources()
            if final_resources:
                final_stats['final_resources'] = final_resources
            
            # Step 8: Generate processing report
            self.logger.info("Step 6: Generating processing report...")
            self._generate_upscaling_report(final_stats, processing_config)
            
            self.logger.info("="*70)
            self.logger.info("AI UPSCALING PROCESS COMPLETED")
            self.logger.info("="*70)
            self.logger.info(f"Total frames processed: {final_stats.get('processed_frames', 0)}")
            self.logger.info(f"Failed frames: {final_stats.get('failed_frames', 0)}")
            self.logger.info(f"Processing time: {total_time/60:.1f} minutes")
            self.logger.info(f"Processing rate: {final_stats['processing_rate_fps']:.2f} fps")
            
            return True, final_stats
            
        except Exception as e:
            self.logger.error(f"AI upscaling execution failed: {str(e)}")
            return False, {}
    
    def _validate_and_recover_frames(self, input_dir: str, output_dir: str, 
                                   frame_files: list, processing_config: ProcessingConfig) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate output frames and attempt recovery for failed frames.
        
        Returns:
            Tuple of (success, validation_stats)
        """
        try:
            import cv2
            
            stats = {
                'validated_frames': 0,
                'corrupted_frames': 0,
                'missing_frames': 0,
                'recovered_frames': 0,
                'final_failed_frames': 0
            }
            
            missing_frames = []
            corrupted_frames = []
            
            # Check each frame
            for frame_file in frame_files:
                output_path = os.path.join(output_dir, frame_file)
                
                if not os.path.exists(output_path):
                    missing_frames.append(frame_file)
                    stats['missing_frames'] += 1
                    continue
                
                # Validate frame integrity
                frame = cv2.imread(output_path)
                if frame is None:
                    corrupted_frames.append(frame_file)
                    stats['corrupted_frames'] += 1
                    continue
                
                stats['validated_frames'] += 1
            
            self.logger.info(f"Validation results:")
            self.logger.info(f"  Valid frames: {stats['validated_frames']}")
            self.logger.info(f"  Missing frames: {stats['missing_frames']}")
            self.logger.info(f"  Corrupted frames: {stats['corrupted_frames']}")
            
            # Attempt recovery for failed frames
            failed_frames = missing_frames + corrupted_frames
            if failed_frames:
                self.logger.info(f"Attempting recovery for {len(failed_frames)} failed frames...")
                
                for frame_file in failed_frames:
                    input_path = os.path.join(input_dir, frame_file)
                    output_path = os.path.join(output_dir, frame_file)
                    
                    # Try single frame upscaling
                    success = self.ai_upscaler._upscale_single_frame(input_path, output_path)
                    
                    if success:
                        # Validate recovered frame
                        frame = cv2.imread(output_path)
                        if frame is not None:
                            stats['recovered_frames'] += 1
                            self.logger.debug(f"Recovered frame: {frame_file}")
                        else:
                            stats['final_failed_frames'] += 1
                    else:
                        stats['final_failed_frames'] += 1
                
                self.logger.info(f"Recovery results:")
                self.logger.info(f"  Recovered frames: {stats['recovered_frames']}")
                self.logger.info(f"  Final failed frames: {stats['final_failed_frames']}")
            
            # Final validation
            total_expected = len(frame_files)
            total_valid = stats['validated_frames'] + stats['recovered_frames']
            success_rate = (total_valid / total_expected) * 100 if total_expected > 0 else 0
            
            self.logger.info(f"Final success rate: {success_rate:.1f}% ({total_valid}/{total_expected})")
            
            # Consider successful if > 95% of frames processed
            validation_success = success_rate > 95.0
            
            return validation_success, stats
            
        except Exception as e:
            self.logger.error(f"Frame validation and recovery failed: {str(e)}")
            return False, {}
    
    def _generate_upscaling_report(self, stats: Dict[str, Any], config: ProcessingConfig) -> None:
        """Generate detailed upscaling process report."""
        try:
            report = {
                "process": "ai_upscaling_execution",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "configuration": {
                    "tool": config.tool.value,
                    "model": config.model,
                    "scale_factor": config.scale_factor,
                    "tile_size": config.tile_size,
                    "gpu_acceleration": config.gpu_acceleration,
                    "denoise_level": config.denoise_level
                },
                "processing_statistics": stats,
                "success": stats.get('final_failed_frames', 0) == 0
            }
            
            report_path = "ai_upscaling_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Upscaling report saved to: {report_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not generate upscaling report: {str(e)}")


def main():
    """Main function to execute AI upscaling process."""
    
    # Configuration
    input_frames_dir = "workspace/test/frames"  # From task 4.1
    output_frames_dir = "workspace/enhanced"
    
    # Verify input frames exist
    if not os.path.exists(input_frames_dir):
        print(f"Error: Input frames directory not found: {input_frames_dir}")
        print("Please ensure task 4.1 (frame extraction) has been completed first.")
        return False
    
    try:
        # Initialize executor
        executor = AIUpscalingExecutor()
        
        # Load processing strategy
        strategy = executor.load_processing_strategy()
        processing_config = executor.create_processing_config(strategy)
        
        # Execute upscaling process
        success, stats = executor.execute_upscaling_process(
            input_frames_dir, output_frames_dir, processing_config
        )
        
        if success:
            print("\n" + "="*70)
            print("🎉 AI UPSCALING PROCESS COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"Input frames: {input_frames_dir}")
            print(f"Output frames: {output_frames_dir}")
            print(f"Processed frames: {stats.get('processed_frames', 0)}")
            print(f"Processing time: {stats.get('total_processing_time', 0)/60:.1f} minutes")
            print(f"Processing rate: {stats.get('processing_rate_fps', 0):.2f} fps")
            
            return True
        else:
            print("\n" + "="*70)
            print("❌ AI UPSCALING PROCESS FAILED")
            print("="*70)
            print("Check the logs for detailed error information.")
            return False
            
    except KeyboardInterrupt:
        print("\n\nUpscaling process interrupted by user.")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)