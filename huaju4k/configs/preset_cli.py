"""
Command-line interface for preset management.

This module provides CLI commands for managing theater presets,
including listing, creating, editing, and validating presets.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from .enhanced_preset_manager import PresetManager, PresetValidationError
from .preset_templates import (
    generate_preset_template, 
    get_recommended_presets,
    validate_preset_parameters,
    get_preset_compatibility_info,
    create_custom_preset_wizard
)
from ..models.config_models import PresetConfig

logger = logging.getLogger(__name__)


class PresetCLI:
    """
    Command-line interface for preset management.
    
    Provides interactive commands for managing theater presets including
    creation, validation, and customization.
    """
    
    def __init__(self, presets_dir: str = "./presets"):
        """
        Initialize preset CLI.
        
        Args:
            presets_dir: Directory for preset files
        """
        self.preset_manager = PresetManager(presets_dir)
        self.presets_dir = Path(presets_dir)
    
    def list_presets(self, detailed: bool = False) -> None:
        """
        List all available presets.
        
        Args:
            detailed: Whether to show detailed information
        """
        try:
            presets = self.preset_manager.list_available_presets()
            
            if not presets:
                print("No presets available.")
                return
            
            print(f"\n📋 Available Presets ({len(presets)}):")
            print("=" * 50)
            
            for preset_name in presets:
                try:
                    if detailed:
                        info = self.preset_manager.get_preset_info(preset_name)
                        print(f"\n🎭 {preset_name}")
                        print(f"   Description: {info.get('description', 'N/A')}")
                        print(f"   Theater Size: {info.get('theater_size', 'N/A')}")
                        print(f"   Quality Level: {info.get('quality_level', 'N/A')}")
                        print(f"   Created By: {info.get('created_by', 'N/A')}")
                        
                        if 'compatibility' in info:
                            comp = info['compatibility']
                            status = "✅" if comp.get('compatible', True) else "⚠️"
                            print(f"   Compatibility: {status}")
                    else:
                        print(f"  • {preset_name}")
                        
                except Exception as e:
                    print(f"  • {preset_name} (error: {e})")
            
            print()
            
        except Exception as e:
            print(f"❌ Error listing presets: {e}")
    
    def show_preset_details(self, preset_name: str) -> None:
        """
        Show detailed information about a preset.
        
        Args:
            preset_name: Name of preset to show
        """
        try:
            preset = self.preset_manager.load_preset(preset_name)
            info = self.preset_manager.get_preset_info(preset_name)
            
            print(f"\n🎭 Preset Details: {preset_name}")
            print("=" * 50)
            
            # Basic information
            print(f"Name: {preset.name}")
            print(f"Description: {preset.description}")
            print(f"Theater Size: {preset.theater_size}")
            print(f"Quality Level: {preset.quality_level}")
            print(f"Target Resolution: {preset.target_resolution}")
            
            # Video parameters
            print(f"\n📹 Video Parameters:")
            print(f"  Denoise Strength: {preset.denoise_strength}")
            print(f"  Tile Size: {preset.tile_size}")
            print(f"  Batch Size: {preset.batch_size}")
            print(f"  Memory Usage: {preset.memory_usage}")
            
            # Audio parameters
            print(f"\n🎵 Audio Parameters:")
            print(f"  Dialogue Boost: {preset.dialogue_boost} dB")
            print(f"  Noise Reduction: {preset.noise_reduction}")
            print(f"  Reverb Reduction: {preset.reverb_reduction}")
            print(f"  Preserve Naturalness: {preset.preserve_naturalness}")
            
            # Metadata
            print(f"\n📊 Metadata:")
            print(f"  Created By: {preset.created_by}")
            print(f"  Created At: {preset.created_at or 'N/A'}")
            print(f"  Version: {preset.version}")
            print(f"  File Exists: {info.get('file_exists', False)}")
            print(f"  Is Default: {info.get('is_default', False)}")
            
            # Compatibility information
            if 'compatibility' in info:
                comp = info['compatibility']
                print(f"\n🔍 Compatibility:")
                status = "✅ Compatible" if comp.get('compatible', True) else "⚠️ Issues Found"
                print(f"  Status: {status}")
                
                if comp.get('warnings'):
                    print(f"  Warnings:")
                    for warning in comp['warnings']:
                        print(f"    • {warning}")
                
                if comp.get('recommendations'):
                    print(f"  Recommendations:")
                    for rec in comp['recommendations']:
                        print(f"    • {rec}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error showing preset details: {e}")
    
    def create_preset_interactive(self) -> None:
        """Create preset through interactive wizard."""
        try:
            print("\n🎨 Create Custom Preset")
            print("=" * 30)
            
            # Get basic information
            name = input("Preset name: ").strip()
            if not name:
                print("❌ Preset name is required.")
                return
            
            description = input("Description: ").strip()
            if not description:
                description = f"Custom preset: {name}"
            
            # Get theater size
            print("\nTheater sizes:")
            print("  1. Small (up to 150 seats)")
            print("  2. Medium (150-500 seats)")
            print("  3. Large (500+ seats)")
            
            theater_choice = input("Select theater size (1-3): ").strip()
            theater_map = {'1': 'small', '2': 'medium', '3': 'large'}
            theater_size = theater_map.get(theater_choice, 'medium')
            
            # Get quality level
            print("\nQuality levels:")
            print("  1. Fast (quick processing)")
            print("  2. Balanced (good quality/speed balance)")
            print("  3. High (maximum quality)")
            
            quality_choice = input("Select quality level (1-3): ").strip()
            quality_map = {'1': 'fast', '2': 'balanced', '3': 'high'}
            quality_level = quality_map.get(quality_choice, 'balanced')
            
            # Get base preset
            base_preset = f"theater_{theater_size}_{quality_level}"
            
            # Get customizations
            print("\n🔧 Customizations (press Enter to skip):")
            
            overrides = {}
            
            # Dialogue boost
            dialogue_input = input(f"Dialogue boost (0.0-20.0, default: auto): ").strip()
            if dialogue_input:
                try:
                    overrides['dialogue_boost'] = float(dialogue_input)
                except ValueError:
                    print("⚠️ Invalid dialogue boost value, using default")
            
            # Noise reduction
            noise_input = input(f"Noise reduction (0.0-1.0, default: auto): ").strip()
            if noise_input:
                try:
                    overrides['noise_reduction'] = float(noise_input)
                except ValueError:
                    print("⚠️ Invalid noise reduction value, using default")
            
            # Tile size
            tile_input = input(f"Tile size (64-2048, default: auto): ").strip()
            if tile_input:
                try:
                    overrides['tile_size'] = int(tile_input)
                except ValueError:
                    print("⚠️ Invalid tile size value, using default")
            
            # Create preset
            print(f"\n🔨 Creating preset '{name}'...")
            
            try:
                preset = self.preset_manager.create_custom_preset(
                    name=name,
                    description=description,
                    base_preset=base_preset,
                    overrides=overrides
                )
                
                # Save preset
                saved_path = self.preset_manager.save_preset(preset, overwrite=True)
                
                print(f"✅ Preset '{name}' created successfully!")
                print(f"📁 Saved to: {Path(saved_path).name}")
                
                # Show validation results
                validation = self.preset_manager.validate_preset(preset)
                if not validation['valid']:
                    print(f"⚠️ Validation warnings:")
                    for error in validation['errors']:
                        print(f"   • {error}")
                
            except Exception as e:
                print(f"❌ Error creating preset: {e}")
            
        except KeyboardInterrupt:
            print("\n\n❌ Preset creation cancelled.")
        except Exception as e:
            print(f"❌ Error in interactive preset creation: {e}")
    
    def validate_preset_command(self, preset_name: str) -> None:
        """
        Validate preset and show results.
        
        Args:
            preset_name: Name of preset to validate
        """
        try:
            preset = self.preset_manager.load_preset(preset_name)
            
            print(f"\n🔍 Validating Preset: {preset_name}")
            print("=" * 40)
            
            # Basic validation
            validation = self.preset_manager.validate_preset(preset)
            
            if validation['valid']:
                print("✅ Preset validation passed")
            else:
                print("❌ Preset validation failed")
                print("\nErrors:")
                for error in validation['errors']:
                    print(f"  • {error}")
            
            if validation['warnings']:
                print("\n⚠️ Warnings:")
                for warning in validation['warnings']:
                    print(f"  • {warning}")
            
            # Theater-specific validation
            preset_dict = preset.__dict__
            theater_validation = validate_preset_parameters(preset_dict)
            
            if theater_validation['recommendations']:
                print("\n💡 Recommendations:")
                for rec in theater_validation['recommendations']:
                    print(f"  • {rec}")
            
            # Compatibility information
            compatibility_info = get_preset_compatibility_info(preset_dict)
            
            if 'theater_info' in compatibility_info and compatibility_info['theater_info']:
                theater_info = compatibility_info['theater_info']
                print(f"\n🎭 Theater Information:")
                print(f"  Size: {theater_info['size'].title()}")
                print(f"  Description: {theater_info['description']}")
                print(f"  Typical Capacity: {theater_info['typical_capacity']} seats")
            
            if 'memory_requirements' in compatibility_info and compatibility_info['memory_requirements']:
                mem_info = compatibility_info['memory_requirements']
                print(f"\n💾 Memory Requirements:")
                print(f"  Estimated Usage: {mem_info['estimated_mb']} MB")
                print(f"  Recommended System Memory: {mem_info['recommended_system_memory_mb']} MB")
            
            print()
            
        except Exception as e:
            print(f"❌ Error validating preset: {e}")
    
    def export_preset_command(self, preset_name: str, export_path: str) -> None:
        """
        Export preset to file.
        
        Args:
            preset_name: Name of preset to export
            export_path: Path to export to
        """
        try:
            exported_path = self.preset_manager.export_preset(preset_name, export_path)
            print(f"✅ Preset '{preset_name}' exported to: {exported_path}")
        except Exception as e:
            print(f"❌ Error exporting preset: {e}")
    
    def import_preset_command(self, import_path: str, preset_name: str = None) -> None:
        """
        Import preset from file.
        
        Args:
            import_path: Path to import from
            preset_name: Optional name for imported preset
        """
        try:
            imported_name = self.preset_manager.import_preset(
                import_path, 
                preset_name, 
                overwrite=True
            )
            print(f"✅ Preset imported as: {imported_name}")
        except Exception as e:
            print(f"❌ Error importing preset: {e}")
    
    def delete_preset_command(self, preset_name: str, force: bool = False) -> None:
        """
        Delete preset.
        
        Args:
            preset_name: Name of preset to delete
            force: Whether to force deletion
        """
        try:
            if not force:
                confirm = input(f"Delete preset '{preset_name}'? (y/N): ").strip().lower()
                if confirm != 'y':
                    print("❌ Deletion cancelled.")
                    return
            
            success = self.preset_manager.delete_preset(preset_name, force=force)
            
            if success:
                print(f"✅ Preset '{preset_name}' deleted successfully.")
            else:
                print(f"⚠️ Preset '{preset_name}' not found or could not be deleted.")
                
        except Exception as e:
            print(f"❌ Error deleting preset: {e}")
    
    def show_recommended_presets(self) -> None:
        """Show recommended preset configurations."""
        try:
            recommended = get_recommended_presets()
            
            print(f"\n💡 Recommended Presets ({len(recommended)}):")
            print("=" * 50)
            
            for i, preset in enumerate(recommended, 1):
                print(f"\n{i}. {preset['name']}")
                print(f"   Description: {preset['description']}")
                print(f"   Theater: {preset['theater_size'].title()}")
                print(f"   Quality: {preset['quality_level'].title()}")
                print(f"   Dialogue Boost: {preset['dialogue_boost']} dB")
                print(f"   Tile Size: {preset['tile_size']}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error showing recommended presets: {e}")
    
    def run_command(self, command: str, args: List[str]) -> None:
        """
        Run preset management command.
        
        Args:
            command: Command to run
            args: Command arguments
        """
        try:
            if command == "list":
                detailed = "--detailed" in args or "-d" in args
                self.list_presets(detailed=detailed)
                
            elif command == "show":
                if not args:
                    print("❌ Preset name required for 'show' command")
                    return
                self.show_preset_details(args[0])
                
            elif command == "create":
                self.create_preset_interactive()
                
            elif command == "validate":
                if not args:
                    print("❌ Preset name required for 'validate' command")
                    return
                self.validate_preset_command(args[0])
                
            elif command == "export":
                if len(args) < 2:
                    print("❌ Preset name and export path required for 'export' command")
                    return
                self.export_preset_command(args[0], args[1])
                
            elif command == "import":
                if not args:
                    print("❌ Import path required for 'import' command")
                    return
                preset_name = args[1] if len(args) > 1 else None
                self.import_preset_command(args[0], preset_name)
                
            elif command == "delete":
                if not args:
                    print("❌ Preset name required for 'delete' command")
                    return
                force = "--force" in args or "-f" in args
                self.delete_preset_command(args[0], force=force)
                
            elif command == "recommended":
                self.show_recommended_presets()
                
            else:
                self.show_help()
                
        except Exception as e:
            print(f"❌ Error running command '{command}': {e}")
    
    def show_help(self) -> None:
        """Show help information."""
        help_text = """
🎭 Preset Management Commands:

  list [--detailed|-d]     List all available presets
  show <preset_name>       Show detailed preset information
  create                   Create new preset interactively
  validate <preset_name>   Validate preset configuration
  export <preset> <path>   Export preset to file
  import <path> [name]     Import preset from file
  delete <preset> [--force|-f]  Delete preset
  recommended              Show recommended preset configurations
  help                     Show this help message

Examples:
  preset list --detailed
  preset show theater_medium_balanced
  preset create
  preset validate my_custom_preset
  preset export my_preset ./backup/my_preset.yaml
  preset import ./backup/my_preset.yaml
  preset delete old_preset --force
"""
        print(help_text)


def main():
    """Main entry point for preset CLI."""
    if len(sys.argv) < 2:
        PresetCLI().show_help()
        return
    
    command = sys.argv[1]
    args = sys.argv[2:]
    
    cli = PresetCLI()
    cli.run_command(command, args)


if __name__ == "__main__":
    main()