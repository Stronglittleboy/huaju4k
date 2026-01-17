"""
简化的配置管理器 - 用于VideoEnhancementProcessor

提供基本的配置加载和管理功能。
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SimpleConfigManager:
    """简化的配置管理器"""
    
    def __init__(self):
        """初始化配置管理器"""
        self.default_config = self._get_default_config()
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载配置
        
        Args:
            config_path: 配置文件路径（可选）
            
        Returns:
            配置字典
        """
        config = self.default_config.copy()
        
        try:
            if config_path and os.path.exists(config_path):
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                
                # 合并配置
                config.update(file_config)
                logger.info(f"配置文件加载成功: {config_path}")
            
            return config
            
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认配置: {e}")
            return self.default_config.copy()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'models_dir': './models',
            'model_cache_size': 2,
            'theater_preset': 'medium',
            'memory_safety_margin': 0.7,
            'temp_dir': None,
            'progress_update_interval': 0.5,
            'video': {
                'ai_model': 'real_esrgan_x4',
                'quality_presets': {
                    'fast': {'tile_size': 512, 'batch_size': 4, 'denoise_strength': 0.5},
                    'balanced': {'tile_size': 768, 'batch_size': 2, 'denoise_strength': 0.7},
                    'high': {'tile_size': 1024, 'batch_size': 1, 'denoise_strength': 0.9}
                }
            },
            'audio': {
                'theater_presets': {
                    'small': {'reverb_reduction': 0.8, 'dialogue_boost': 6.0, 'noise_reduction': 0.7},
                    'medium': {'reverb_reduction': 0.6, 'dialogue_boost': 4.0, 'noise_reduction': 0.5},
                    'large': {'reverb_reduction': 0.4, 'dialogue_boost': 2.0, 'noise_reduction': 0.3}
                }
            },
            'performance': {
                'use_gpu': True,
                'gpu_id': 0,
                'max_memory_usage': 0.7,
                'tile_overlap': 32,
                'num_workers': 2,
                'prefetch_factor': 2
            }
        }