"""
GPU Super Resolver - 帧级超分辨率处理器

提供单帧/区域级别的 Real-ESRGAN 超分辨率接口。
视频级循环（读取/写入帧、进度显示、音频合并）由 ThreeStageEnhancer 负责。
"""

import logging
from typing import Optional

import cv2
import numpy as np

from .model_manager import GPUModelManager

logger = logging.getLogger(__name__)

MIN_TILE_SIZE = 96


class GPUSuperResolver:
    """帧级 GPU 超分辨率处理器，基于 Real-ESRGAN"""

    def __init__(
        self,
        model_name: str = "RealESRGAN_x4plus",
        tile_size: int = 384,
        device: str = "cuda",
        models_dir: str = "./models",
    ):
        self.model_name = model_name
        self.tile_size = tile_size
        self.device = device
        self.model_manager = GPUModelManager(models_dir=models_dir)
        self._model_loaded = False
        logger.info(
            "GPUSuperResolver 初始化: model=%s, tile=%d, device=%s",
            model_name, tile_size, device,
        )

    def _ensure_model_loaded(self) -> bool:
        """确保模型已加载到 GPU"""
        if self._model_loaded:
            return True

        success = self.model_manager.load_model(
            model_name=self.model_name,
            tile_size=self.tile_size,
            half=True,
        )

        if success:
            self._model_loaded = True
            stats = self.model_manager.get_gpu_stats()
            if stats.get("available"):
                logger.info(
                    "GPU 就绪: %s, 显存已分配 %d MB",
                    stats["device"], stats["allocated_mb"],
                )

        return success

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        对单帧进行超分辨率增强。

        内部处理 CUDA OOM：自动将 tile_size 减半并重试，
        直至 tile_size 降到 MIN_TILE_SIZE 仍失败则向上抛出。

        Args:
            frame: BGR uint8 numpy 数组，原始分辨率

        Returns:
            BGR uint8 numpy 数组，放大后的分辨率
        """
        if not self._ensure_model_loaded():
            raise RuntimeError("模型加载失败，无法进行 GPU 超分")

        import torch

        while True:
            try:
                return self.model_manager.enhance_frame(frame)
            except torch.cuda.OutOfMemoryError:
                if self.tile_size <= MIN_TILE_SIZE:
                    logger.error(
                        "tile_size 已降至最小值 %d 仍然 OOM，放弃重试",
                        MIN_TILE_SIZE,
                    )
                    raise
                old = self.tile_size
                self.update_tile_size(self.tile_size // 2)
                logger.warning(
                    "CUDA OOM, tile_size %d -> %d 后重试", old, self.tile_size,
                )

    def enhance_region(self, region: np.ndarray) -> np.ndarray:
        """
        对裁剪区域进行超分辨率增强。

        语义上用于对 bbox 裁剪出的子图做超分，
        实际调用与 enhance_frame 相同。

        Args:
            region: BGR uint8 numpy 数组

        Returns:
            BGR uint8 numpy 数组，放大后的区域
        """
        return self.enhance_frame(region)

    def get_scale_factor(self) -> int:
        """返回当前模型的放大倍数（4 或 2）"""
        return 4 if "x4" in self.model_name else 2

    def update_tile_size(self, new_tile_size: int) -> None:
        """动态调整 tile 大小（用于 OOM 恢复等场景）"""
        new_tile_size = max(new_tile_size, MIN_TILE_SIZE)
        if new_tile_size == self.tile_size:
            return
        logger.info("tile_size 更新: %d -> %d", self.tile_size, new_tile_size)
        self.tile_size = new_tile_size
        if self._model_loaded:
            self._model_loaded = False
            self.model_manager.unload_model()

    def cleanup(self) -> None:
        """释放 GPU 显存"""
        self.model_manager.unload_model()
        self._model_loaded = False
        logger.info("GPU 资源已释放")


GPUVideoSuperResolver = GPUSuperResolver
