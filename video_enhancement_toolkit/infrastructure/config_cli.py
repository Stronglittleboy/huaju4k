"""
Configuration CLI Utilities

Command-line utilities for configuration management.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path

from .configuration_manager import ConfigurationManager
from .config_validator import ConfigValidator


class ConfigCLI:
    """Command-line interface for configuration management."""
    
    def __init__(self, config_manager: ConfigurationManager):
        """Initialize configuration CLI.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
    
    def create_preset_interactive(self) -> None:
        """Create a preset interactively."""
        print("Creating new configuration preset...")
        
        preset_name = input("Enter preset name: ").strip()
        if not preset_name:
            print("Error: Preset name cannot be empty")
            return
        
        # Check if preset already exists
        if self.config_manager.get_preset(preset_name):
            overwrite = input(f"Preset '{preset_name}' already exists. Overwrite? (y/N): ").strip().lower()
            if overwrite != 'y':
                print("Preset creation cancelled.")
                return
        
        preset_config = {}
        
        # Video configuration
        if self._confirm("Configure video settings?"):
            video_config = self._get_video_config_interactive()
            if video_config:
                preset_config["video_config"] = video_config
        
        # Audio configuration
        if self._confirm("Configure audio settings?"):
            audio_config = self._get_audio_config_interactive()
            if audio_config:
                preset_config["audio_config"] = audio_config
        
        # Performance configuration
        if self._confirm("Configure performance settings?"):
            performance_config = self._get_performance_config_interactive()
            if performance_config:
                preset_config["performance_config"] = performance_config
        
        if not preset_config:
            print("No configuration provided. Preset creation cancelled.")
            return
        
        try:
            self.config_manager.save_preset(preset_name, preset_config)
            print(f"Preset '{preset_name}' created successfully!")
        except Exception as e:
            print(f"Error creating preset: {e}")
    
    def list_presets(self) -> None:
        """List all available presets."""
        presets = self.config_manager.get_available_presets()
        
        if not presets:
            print("No presets available.")
            return
        
        print("Available presets:")
        for preset_name in presets:
            preset_config = self.config_manager.get_preset(preset_name)
            sections = list(preset_config.keys()) if preset_config else []
            print(f"  - {preset_name} ({', '.join(sections)})")
    
    def show_preset(self, preset_name: str) -> None:
        """Show preset configuration.
        
        Args:
            preset_name: Name of preset to show
        """
        preset_config = self.config_manager.get_preset(preset_name)
        
        if not preset_config:
            print(f"Preset '{preset_name}' not found.")
            return
        
        print(f"Preset '{preset_name}':")
        print(json.dumps(preset_config, indent=2))
    
    def validate_config_file(self, config_path: str) -> None:
        """Validate a configuration file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if self.config_manager.validate_configuration(config):
                print(f"Configuration file '{config_path}' is valid.")
            else:
                print(f"Configuration file '{config_path}' is invalid.")
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading configuration file: {e}")
    
    def _get_video_config_interactive(self) -> Optional[Dict[str, Any]]:
        """Get video configuration interactively."""
        config = {}
        
        # AI Model
        model = input("AI model (realesrgan-x4plus): ").strip() or "realesrgan-x4plus"
        is_valid, error = ConfigValidator.validate_ai_model(model)
        if not is_valid:
            print(f"Warning: {error}")
        config["ai_model"] = model
        
        # Tile size
        try:
            tile_size = int(input("Tile size (512): ").strip() or "512")
            if not 64 <= tile_size <= 2048:
                print("Warning: Tile size should be between 64 and 2048")
            config["tile_size"] = tile_size
        except ValueError:
            config["tile_size"] = 512
        
        # Batch size
        try:
            batch_size = int(input("Batch size (4): ").strip() or "4")
            if not 1 <= batch_size <= 32:
                print("Warning: Batch size should be between 1 and 32")
            config["batch_size"] = batch_size
        except ValueError:
            config["batch_size"] = 4
        
        # GPU acceleration
        gpu_accel = input("GPU acceleration (y/N): ").strip().lower()
        config["gpu_acceleration"] = gpu_accel == 'y'
        
        return config
    
    def _get_audio_config_interactive(self) -> Optional[Dict[str, Any]]:
        """Get audio configuration interactively."""
        config = {}
        
        # Theater preset
        preset = input("Theater preset (small/medium/large) [medium]: ").strip() or "medium"
        is_valid, error = ConfigValidator.validate_theater_preset(preset)
        if not is_valid:
            print(f"Warning: {error}")
        config["theater_preset"] = preset
        
        # Noise reduction
        try:
            noise_reduction = float(input("Noise reduction strength (0.0-1.0) [0.3]: ").strip() or "0.3")
            if not 0.0 <= noise_reduction <= 1.0:
                print("Warning: Noise reduction should be between 0.0 and 1.0")
            config["noise_reduction_strength"] = noise_reduction
        except ValueError:
            config["noise_reduction_strength"] = 0.3
        
        # Dialogue enhancement
        try:
            dialogue_enhancement = float(input("Dialogue enhancement (0.0-1.0) [0.4]: ").strip() or "0.4")
            if not 0.0 <= dialogue_enhancement <= 1.0:
                print("Warning: Dialogue enhancement should be between 0.0 and 1.0")
            config["dialogue_enhancement"] = dialogue_enhancement
        except ValueError:
            config["dialogue_enhancement"] = 0.4
        
        # Preserve naturalness
        preserve = input("Preserve naturalness (Y/n): ").strip().lower()
        config["preserve_naturalness"] = preserve != 'n'
        
        return config
    
    def _get_performance_config_interactive(self) -> Optional[Dict[str, Any]]:
        """Get performance configuration interactively."""
        config = {}
        
        # CPU threads
        import os
        max_threads = os.cpu_count() or 1
        try:
            cpu_threads = int(input(f"CPU threads (1-{max_threads * 2}) [4]: ").strip() or "4")
            if not 1 <= cpu_threads <= max_threads * 2:
                print(f"Warning: CPU threads should be between 1 and {max_threads * 2}")
            config["cpu_threads"] = cpu_threads
        except ValueError:
            config["cpu_threads"] = 4
        
        # GPU memory limit
        try:
            gpu_memory = float(input("GPU memory limit (GB) [4.0]: ").strip() or "4.0")
            if gpu_memory <= 0:
                print("Warning: GPU memory limit should be positive")
            config["gpu_memory_limit"] = gpu_memory
        except ValueError:
            config["gpu_memory_limit"] = 4.0
        
        # Memory optimization
        mem_opt = input("Memory optimization (Y/n): ").strip().lower()
        config["memory_optimization"] = mem_opt != 'n'
        
        # Parallel processing
        parallel = input("Parallel processing (Y/n): ").strip().lower()
        config["parallel_processing"] = parallel != 'n'
        
        return config
    
    def _confirm(self, message: str) -> bool:
        """Get user confirmation.
        
        Args:
            message: Confirmation message
            
        Returns:
            True if user confirms, False otherwise
        """
        response = input(f"{message} (y/N): ").strip().lower()
        return response == 'y'