"""
GPU Model Manager - Real-ESRGAN 模型管理

负责：
- 模型下载和加载
- GPU 显存管理
- 模型生命周期控制
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class GPUModelManager:
    """
    GPU 模型管理器
    
    约束：
    - 同时只加载一个模型
    - 6GB 显存安全约束
    - 进程生命周期内模型常驻
    """
    
    # 模型配置
    MODEL_CONFIGS = {
        "RealESRGAN_x4plus": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "filename": "RealESRGAN_x4plus.pth",
            "scale": 4,
            "num_block": 23,
            "num_feat": 64,
            "num_grow_ch": 32,
        },
        "RealESRGAN_x2plus": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "filename": "RealESRGAN_x2plus.pth",
            "scale": 2,
            "num_block": 23,
            "num_feat": 64,
            "num_grow_ch": 32,
        },
    }
    
    def __init__(self, models_dir: str = "./models"):
        """
        初始化模型管理器
        
        Args:
            models_dir: 模型存储目录
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_model = None
        self.current_model_name = None
        self.device = None
        
        # 检查 GPU 可用性
        self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """检查 GPU 是否可用"""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory
                total_memory_gb = total_memory / (1024**3)
                
                logger.info(f"GPU 可用: {device_name}")
                logger.info(f"显存: {total_memory_gb:.1f} GB")
                
                self.device = "cuda"
                return True
            else:
                logger.warning("CUDA 不可用，将使用 CPU")
                self.device = "cpu"
                return False
        except ImportError:
            logger.error("PyTorch 未安装")
            self.device = "cpu"
            return False
    
    def download_model(self, model_name: str) -> Optional[str]:
        """
        下载模型文件
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型文件路径，失败返回 None
        """
        if model_name not in self.MODEL_CONFIGS:
            logger.error(f"未知模型: {model_name}")
            return None
        
        config = self.MODEL_CONFIGS[model_name]
        model_path = self.models_dir / config["filename"]
        
        if model_path.exists():
            logger.info(f"模型已存在: {model_path}")
            return str(model_path)
        
        logger.info(f"下载模型: {config['url']}")
        
        try:
            import urllib.request
            urllib.request.urlretrieve(config["url"], str(model_path))
            logger.info(f"模型下载完成: {model_path}")
            return str(model_path)
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            return None
    
    def load_model(self, model_name: str = "RealESRGAN_x4plus", 
                   tile_size: int = 384,
                   half: bool = True) -> bool:
        """
        加载 Real-ESRGAN 模型到 GPU
        
        Args:
            model_name: 模型名称
            tile_size: 瓦片大小 (影响显存占用)
            half: 是否使用 FP16 (节省显存)
            
        Returns:
            加载成功返回 True
        """
        if self.current_model_name == model_name and self.current_model is not None:
            logger.info(f"模型已加载: {model_name}")
            return True
        
        # 卸载当前模型
        self.unload_model()
        
        try:
            # 检查依赖
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            import torch
        except ImportError as e:
            logger.error(f"依赖缺失: {e}")
            logger.error("请安装: pip install realesrgan basicsr")
            return False
        
        # 下载模型
        model_path = self.download_model(model_name)
        if model_path is None:
            return False
        
        config = self.MODEL_CONFIGS[model_name]
        
        try:
            # 创建网络架构
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=config["num_feat"],
                num_block=config["num_block"],
                num_grow_ch=config["num_grow_ch"],
                scale=config["scale"]
            )
            
            # 创建 Real-ESRGAN 推理器
            self.current_model = RealESRGANer(
                scale=config["scale"],
                model_path=model_path,
                model=model,
                tile=tile_size,
                tile_pad=10,
                pre_pad=0,
                half=half and self.device == "cuda",
                device=self.device
            )
            
            self.current_model_name = model_name
            
            # 验证 GPU 加载
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated(0) / (1024**2)
                logger.info(f"模型加载到 GPU，显存占用: {allocated:.0f} MB")
            
            logger.info(f"✅ 模型加载成功: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.current_model = None
            self.current_model_name = None
            return False
    
    def unload_model(self) -> None:
        """卸载当前模型，释放显存"""
        if self.current_model is not None:
            try:
                import torch
                
                self.current_model = None
                self.current_model_name = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                logger.info("模型已卸载，显存已释放")
            except Exception as e:
                logger.warning(f"模型卸载时出错: {e}")
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """获取 GPU 状态信息"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "device": torch.cuda.get_device_name(0),
                    "total_memory_mb": torch.cuda.get_device_properties(0).total_memory // (1024**2),
                    "allocated_mb": torch.cuda.memory_allocated(0) // (1024**2),
                    "cached_mb": torch.cuda.memory_reserved(0) // (1024**2),
                    "available": True
                }
            return {"available": False, "reason": "CUDA not available"}
        except ImportError:
            return {"available": False, "reason": "PyTorch not installed"}
    
    def enhance_frame(self, frame) -> Any:
        """
        增强单帧图像
        
        Args:
            frame: BGR 格式的 numpy 数组
            
        Returns:
            增强后的帧
        """
        if self.current_model is None:
            raise RuntimeError("模型未加载")
        
        output, _ = self.current_model.enhance(frame, outscale=self.MODEL_CONFIGS[self.current_model_name]["scale"])
        return output
