"""
ThreeStageEnhancer — 话剧视频 4K 增强三阶段管线

Stage 1: 预处理（原始分辨率）— 音频提取、降噪、场景分段
Stage 2: AI 超分（1080p → 4K）— 背景复用 + 逐帧/区域超分
Stage 3: 后处理（4K）— 分区域时序修复、音频增强、最终编码
"""

import atexit
import hashlib
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..models.data_models import (
    CheckpointData,
    EnhancementStrategy,
    ProcessResult,
    SceneSegment,
)
from ..core.checkpoint_system import CheckpointSystem
from ..core.progress_tracker import MultiStageProgressTracker

logger = logging.getLogger(__name__)


class ThreeStageEnhancer:
    """话剧视频三阶段 4K 增强管线。"""

    def __init__(self, checkpoint_dir: str = "./checkpoints",
                 temp_dir: str = "./temp"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_system = CheckpointSystem(str(self.checkpoint_dir))
        self._gpu_resolver = None
        self._interrupted = False
        self._temp_files: List[Path] = []

        self._register_signal_handlers()
        atexit.register(self._cleanup_on_exit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, input_path: str, output_path: str,
                strategy: EnhancementStrategy,
                preview: bool = False,
                preview_at: Optional[str] = None,
                segment: Optional[Tuple[str, str]] = None,
                resume: bool = False) -> ProcessResult:
        """执行完整的三阶段增强管线。"""
        start_time = time.time()
        frames_processed = 0

        try:
            # -- 断点续传检查 --
            checkpoint: Optional[CheckpointData] = None
            if resume:
                checkpoint = self.checkpoint_system.find_latest_by_input(input_path)
                if checkpoint:
                    logger.info("找到断点: frame=%d, scene=%d",
                                checkpoint.frame_index, checkpoint.scene_index)
                else:
                    logger.info("未找到匹配的断点，从头开始")

            # -- Stage 1: 预处理 --
            logger.info("=== Stage 1: 预处理 ===")
            stage1_video = self.temp_dir / "stage1_clean.mp4"
            audio_path = self.temp_dir / "audio.wav"
            self._temp_files.extend([stage1_video, audio_path])

            if not (resume and checkpoint and stage1_video.exists()):
                self._stage1_preprocess(input_path, str(stage1_video),
                                        str(audio_path), strategy)

            # -- Stage 2: AI 超分 --
            logger.info("=== Stage 2: AI 超分 ===")
            stage2_video = self.temp_dir / "stage2_upscaled.mp4"
            self._temp_files.append(stage2_video)

            frames_processed = self._stage2_super_resolve(
                str(stage1_video), str(stage2_video), strategy,
                checkpoint=checkpoint,
                preview=preview,
                preview_at=preview_at,
                segment=segment,
            )

            if self._interrupted:
                return ProcessResult(success=False, error="用户中断",
                                     frames_processed=frames_processed)

            # -- Stage 3: 后处理 --
            logger.info("=== Stage 3: 后处理 ===")
            self._stage3_postprocess(
                str(stage2_video), str(audio_path), output_path,
                strategy,
            )

            elapsed = time.time() - start_time
            report = self._build_report(strategy, frames_processed, elapsed)

            return ProcessResult(
                success=True,
                output_path=output_path,
                processing_time=elapsed,
                frames_processed=frames_processed,
                report=report,
            )

        except Exception as e:
            logger.exception("管线执行失败")
            return ProcessResult(
                success=False,
                output_path=output_path,
                processing_time=time.time() - start_time,
                frames_processed=frames_processed,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Stage 1: 预处理
    # ------------------------------------------------------------------

    def _stage1_preprocess(self, input_path: str, output_video: str,
                           audio_path: str, strategy: EnhancementStrategy) -> None:
        """提取音频 + 降噪（保留灯光意图，不做曝光校正）。"""

        # 提取音频
        audio_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2",
            audio_path,
        ]
        subprocess.run(audio_cmd, capture_output=True, check=False)

        # 降噪
        denoise_map = {
            "low": "hqdn3d=1:1:4:4",
            "medium": "hqdn3d=2:2:6:6",
            "high": "hqdn3d=3:3:10:10",
        }
        denoise_filter = denoise_map.get(strategy.denoise_strength, "hqdn3d=2:2:6:6")

        video_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-an",
            "-vf", denoise_filter,
            "-c:v", "libx264", "-preset", "fast", "-crf", "16",
            output_video,
        ]
        subprocess.run(video_cmd, capture_output=True, check=True)
        logger.info("Stage 1 完成: %s", output_video)

    # ------------------------------------------------------------------
    # Stage 2: AI 超分
    # ------------------------------------------------------------------

    def _stage2_super_resolve(self, input_video: str, output_video: str,
                              strategy: EnhancementStrategy,
                              checkpoint: Optional[CheckpointData] = None,
                              preview: bool = False,
                              preview_at: Optional[str] = None,
                              segment: Optional[Tuple[str, str]] = None) -> int:
        """逐场景、逐帧 AI 超分，支持背景复用和断点续传。"""

        if strategy.model_name == "lanczos":
            return self._stage2_lanczos(input_video, output_video)

        resolver = self._get_gpu_resolver(strategy)
        if resolver is None:
            logger.warning("GPU 不可用，回退到 lanczos")
            return self._stage2_lanczos(input_video, output_video)

        scale = resolver.get_scale_factor()

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {input_video}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_w, out_h = width * scale, height * scale

        # 处理范围
        start_frame, end_frame = 0, total_frames
        if preview or preview_at:
            preview_frames = int(fps * 30)
            if preview_at:
                start_frame = self._time_to_frame(preview_at, fps)
            else:
                start_frame = self._find_static_scene_start(strategy.scenes)
            end_frame = min(start_frame + preview_frames, total_frames)
        elif segment:
            start_frame = self._time_to_frame(segment[0], fps)
            end_frame = min(self._time_to_frame(segment[1], fps), total_frames)

        resume_frame = 0
        if checkpoint:
            resume_frame = checkpoint.frame_index

        # FFmpeg 编码器
        encode_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-crf", "16",
            "-pix_fmt", "yuv420p",
            output_video,
        ]
        encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)

        frame_idx = 0
        processed = 0
        bg_cache: Optional[np.ndarray] = None
        bg_orig: Optional[np.ndarray] = None
        current_scene_idx = -1
        input_hash = self._file_hash(input_video)
        last_checkpoint_frame = resume_frame
        t0 = time.time()

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame

        try:
            while frame_idx < end_frame:
                if self._interrupted:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx < resume_frame:
                    frame_idx += 1
                    continue

                scene = self._find_scene(strategy.scenes, frame_idx)
                scene_idx = self._scene_index(strategy.scenes, frame_idx)

                if scene_idx != current_scene_idx:
                    current_scene_idx = scene_idx
                    bg_cache = None
                    bg_orig = None
                    if scene and scene.type == "static":
                        bg_orig, bg_cache = self._extract_and_upscale_background(
                            input_video, scene, resolver, cap, width, height
                        )
                        logger.info("场景 %d: static, 背景已缓存", scene_idx)
                    else:
                        stype = scene.type if scene else "unknown"
                        logger.info("场景 %d: %s, 全帧处理", scene_idx, stype)

                # 超分处理
                if scene and scene.type == "static" and bg_cache is not None and bg_orig is not None:
                    out_frame = self._composite_with_background(
                        frame, bg_orig, bg_cache, resolver, scale
                    )
                else:
                    out_frame = self._safe_enhance_frame(frame, resolver, out_w, out_h)

                encoder.stdin.write(out_frame.tobytes())
                frame_idx += 1
                processed += 1

                # 进度显示
                if processed % 50 == 0:
                    elapsed = time.time() - t0
                    fps_actual = processed / elapsed if elapsed > 0 else 0
                    total = end_frame - start_frame
                    done_ratio = processed / total if total > 0 else 0
                    eta = (total - processed) / fps_actual if fps_actual > 0 else 0
                    sys.stdout.write(
                        f"\r  [{self._bar(done_ratio)}] {done_ratio*100:.1f}% "
                        f"| {processed}/{total} | {fps_actual:.2f}fps "
                        f"| ETA {self._fmt_time(eta)} | Scene{scene_idx}"
                    )
                    sys.stdout.flush()

                # 断点保存
                if processed - (last_checkpoint_frame - resume_frame) >= 500:
                    self._save_checkpoint(input_hash, frame_idx, scene_idx, strategy)
                    last_checkpoint_frame = frame_idx

        finally:
            cap.release()
            if encoder.stdin:
                encoder.stdin.close()
            encoder.wait()
            sys.stdout.write("\n")

        logger.info("Stage 2 完成: %d 帧已处理", processed)
        return processed

    def _stage2_lanczos(self, input_video: str, output_video: str) -> int:
        """CPU lanczos 放大（无 GPU 回退）。"""
        cmd = [
            "ffmpeg", "-y", "-i", input_video,
            "-vf", "scale=3840:2160:flags=lanczos,unsharp=3:3:0.3",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-an", output_video,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        cap = cv2.VideoCapture(output_video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total

    # ------------------------------------------------------------------
    # Stage 2 helpers
    # ------------------------------------------------------------------

    def _extract_and_upscale_background(self, video_path: str, scene: SceneSegment,
                                        resolver: Any, cap: cv2.VideoCapture,
                                        w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        """采样场景帧 → 中位数背景 → AI 超分缓存。"""
        total = scene.end_frame - scene.start_frame
        if total <= 0:
            raise ValueError("空场景段")

        # 首尾各 2 帧 + 中间均匀 6 帧
        indices = []
        indices.extend([scene.start_frame, scene.start_frame + 1])
        indices.extend([scene.end_frame - 2, scene.end_frame - 1])
        mid_count = min(6, total - 4)
        if mid_count > 0:
            step = max(1, (total - 4) // (mid_count + 1))
            for i in range(1, mid_count + 1):
                indices.append(scene.start_frame + 2 + i * step)
        indices = sorted(set(min(i, scene.end_frame - 1) for i in indices))

        frames = []
        old_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame.astype(np.float32))
        cap.set(cv2.CAP_PROP_POS_FRAMES, old_pos)

        if not frames:
            raise RuntimeError("无法采样背景帧")

        bg_orig = np.median(np.stack(frames), axis=0).astype(np.uint8)
        bg_upscaled = self._safe_enhance_frame(bg_orig, resolver,
                                                w * resolver.get_scale_factor(),
                                                h * resolver.get_scale_factor())
        return bg_orig, bg_upscaled

    def _composite_with_background(self, frame: np.ndarray, bg_orig: np.ndarray,
                                   bg_4k: np.ndarray, resolver: Any,
                                   scale: int) -> np.ndarray:
        """前景裁切超分 → 粘贴到缓存背景。"""
        diff = cv2.absdiff(frame, bg_orig)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray_diff, (0, 0), 3)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=2)

        coords = cv2.findNonZero(mask)
        if coords is None:
            return bg_4k.copy()

        x, y, bw, bh = cv2.boundingRect(coords)
        pad = max(1, int(frame.shape[1] * 0.008))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + bw + pad)
        y2 = min(frame.shape[0], y + bh + pad)

        # 4px 对齐
        x1 = x1 - (x1 % 4)
        y1 = y1 - (y1 % 4)
        x2 = x2 + (4 - x2 % 4) if x2 % 4 else x2
        y2 = y2 + (4 - y2 % 4) if y2 % 4 else y2
        x2 = min(x2, frame.shape[1])
        y2 = min(y2, frame.shape[0])

        region = frame[y1:y2, x1:x2]
        region_up = self._safe_enhance_frame(
            region, resolver,
            (x2 - x1) * scale, (y2 - y1) * scale,
        )

        output = bg_4k.copy()
        sx, sy = x1 * scale, y1 * scale
        rh, rw = region_up.shape[:2]
        oh, ow = output.shape[:2]
        rh = min(rh, oh - sy)
        rw = min(rw, ow - sx)
        region_up = region_up[:rh, :rw]

        # Alpha 渐变混合边缘
        blend_px = min(pad * scale, rh // 2, rw // 2, 10)
        if blend_px > 0:
            alpha_mask = np.ones((rh, rw), dtype=np.float32)
            for i in range(blend_px):
                a = i / blend_px
                alpha_mask[i, :] = a
                alpha_mask[rh - 1 - i, :] = a
                alpha_mask[:, i] = np.minimum(alpha_mask[:, i], a)
                alpha_mask[:, rw - 1 - i] = np.minimum(alpha_mask[:, rw - 1 - i], a)
            alpha_3 = alpha_mask[:, :, np.newaxis]
            bg_region = output[sy:sy+rh, sx:sx+rw].astype(np.float32)
            blended = (region_up.astype(np.float32) * alpha_3
                       + bg_region * (1 - alpha_3))
            output[sy:sy+rh, sx:sx+rw] = blended.astype(np.uint8)
        else:
            output[sy:sy+rh, sx:sx+rw] = region_up

        return output

    def _safe_enhance_frame(self, frame: np.ndarray, resolver: Any,
                            target_w: int, target_h: int) -> np.ndarray:
        """带 OOM 回退的帧超分。"""
        try:
            return resolver.enhance_frame(frame)
        except Exception as e:
            logger.warning("GPU 超分失败 (%s)，回退 lanczos", e)
            return cv2.resize(frame, (target_w, target_h),
                              interpolation=cv2.INTER_LANCZOS4)

    # ------------------------------------------------------------------
    # Stage 3: 后处理
    # ------------------------------------------------------------------

    def _stage3_postprocess(self, input_video: str, audio_path: str,
                            output_path: str, strategy: EnhancementStrategy) -> None:
        """分区域时序修复 + 音频增强 + 最终编码。"""

        codec_map = {
            "h264": ["-c:v", "libx264", "-preset", "slow"],
            "h265": ["-c:v", "libx265", "-preset", "medium"],
            "prores": ["-c:v", "prores_ks", "-profile:v", "2"],
        }
        codec_args = codec_map.get(strategy.codec, codec_map["h264"])

        temp_temporal = self.temp_dir / "stage3_temporal.mp4"
        self._temp_files.append(temp_temporal)

        # 分区域时序修复 (Python 帧级处理)
        self._temporal_repair(input_video, str(temp_temporal), strategy)

        # 锐化 + 音视频合并
        has_audio = Path(audio_path).exists() and Path(audio_path).stat().st_size > 0

        cmd = ["ffmpeg", "-y", "-i", str(temp_temporal)]
        if has_audio:
            cmd += ["-i", audio_path]
        cmd += ["-vf", "unsharp=3:3:0.3"]
        cmd += codec_args
        cmd += ["-crf", str(strategy.crf), "-pix_fmt", "yuv420p"]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "1:a:0"]
        else:
            cmd += ["-an"]
        cmd.append(output_path)

        subprocess.run(cmd, capture_output=True, check=True)
        logger.info("Stage 3 完成: %s", output_path)

    def _temporal_repair(self, input_video: str, output_video: str,
                         strategy: EnhancementStrategy) -> None:
        """FFmpeg pipe → Python 帧级时序平滑 → FFmpeg pipe 编码。"""
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开: {input_video}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        decode_cmd = [
            "ffmpeg", "-i", input_video,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-v", "quiet", "-",
        ]
        encode_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "fast", "-crf", "16",
            "-pix_fmt", "yuv420p", "-v", "quiet",
            output_video,
        ]

        decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE)
        encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

        frame_size = w * h * 3
        alpha = 0.15
        prev_frame: Optional[np.ndarray] = None
        ema_buffer: Optional[np.ndarray] = None

        try:
            while True:
                raw = decoder.stdout.read(frame_size)
                if len(raw) != frame_size:
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()

                if prev_frame is not None:
                    diff = cv2.absdiff(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                    )
                    low_motion = diff < 25
                    high_motion = ~low_motion

                    if ema_buffer is None:
                        ema_buffer = frame.astype(np.float32)
                    else:
                        ema_buffer = ema_buffer * (1 - alpha) + frame.astype(np.float32) * alpha

                    smoothed = ema_buffer.astype(np.uint8)
                    mask_3ch = np.stack([low_motion]*3, axis=-1)
                    frame = np.where(mask_3ch, smoothed, frame)

                prev_frame = frame.copy()
                encoder.stdin.write(frame.tobytes())

        finally:
            decoder.stdout.close()
            encoder.stdin.close()
            decoder.wait()
            encoder.wait()

    # ------------------------------------------------------------------
    # GPU resolver 管理
    # ------------------------------------------------------------------

    def _get_gpu_resolver(self, strategy: EnhancementStrategy) -> Optional[Any]:
        """延迟加载 GPU 超分器。"""
        if self._gpu_resolver is not None:
            return self._gpu_resolver

        try:
            from ..gpu_stage.gpu_super_resolver import GPUSuperResolver
            model_map = {
                "x4plus": "RealESRGAN_x4plus",
                "x2plus": "RealESRGAN_x2plus",
            }
            model_name = model_map.get(strategy.model_name, "RealESRGAN_x4plus")
            self._gpu_resolver = GPUSuperResolver(
                model_name=model_name,
                tile_size=strategy.tile_size,
            )
            return self._gpu_resolver
        except Exception as e:
            logger.warning("GPU 超分器初始化失败: %s", e)
            return None

    # ------------------------------------------------------------------
    # 信号处理 / 生命周期
    # ------------------------------------------------------------------

    def _register_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            sig_name = signal.Signals(signum).name
            logger.info("收到 %s, 正在保存断点...", sig_name)
            self._interrupted = True

        try:
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)
        except (OSError, ValueError):
            pass

    def _cleanup_on_exit(self) -> None:
        for f in self._temp_files:
            try:
                if f.exists():
                    f.unlink()
            except Exception:
                pass
        if self._gpu_resolver is not None:
            try:
                self._gpu_resolver.cleanup()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _save_checkpoint(self, input_hash: str, frame_idx: int,
                         scene_idx: int, strategy: EnhancementStrategy) -> None:
        from ..models.data_models import ProcessingStrategy
        cp = self.checkpoint_system.create_checkpoint(
            processor_state={"model": strategy.model_name, "tile": strategy.tile_size},
            processing_progress=0.0,
            current_stage="stage2",
            input_path="",
            output_path="",
            strategy=ProcessingStrategy(),
        )
        cp.input_hash = input_hash
        cp.frame_index = frame_idx
        cp.scene_index = scene_idx
        self.checkpoint_system.save_checkpoint(cp)

    @staticmethod
    def _file_hash(path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            h.update(f.read(1024 * 1024))
        return h.hexdigest()

    @staticmethod
    def _find_scene(scenes: List[SceneSegment], frame_idx: int) -> Optional[SceneSegment]:
        for s in scenes:
            if s.start_frame <= frame_idx < s.end_frame:
                return s
        return None

    @staticmethod
    def _scene_index(scenes: List[SceneSegment], frame_idx: int) -> int:
        for i, s in enumerate(scenes):
            if s.start_frame <= frame_idx < s.end_frame:
                return i
        return -1

    @staticmethod
    def _find_static_scene_start(scenes: List[SceneSegment]) -> int:
        for s in scenes:
            if s.type == "static":
                return s.start_frame
        return 0

    @staticmethod
    def _time_to_frame(time_str: str, fps: float) -> int:
        parts = time_str.split(":")
        seconds = 0.0
        for p in parts:
            seconds = seconds * 60 + float(p)
        return int(seconds * fps)

    @staticmethod
    def _bar(ratio: float, width: int = 30) -> str:
        filled = int(width * ratio)
        return "\u2588" * filled + "\u2591" * (width - filled)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"

    def _build_report(self, strategy: EnhancementStrategy,
                      frames: int, elapsed: float) -> Dict[str, Any]:
        static = sum(1 for s in strategy.scenes if s.type == "static")
        gradual = sum(1 for s in strategy.scenes if s.type == "gradual")
        dynamic = sum(1 for s in strategy.scenes if s.type == "dynamic")
        return {
            "frames_processed": frames,
            "elapsed_seconds": elapsed,
            "avg_fps": frames / elapsed if elapsed > 0 else 0,
            "model": strategy.model_name,
            "tile_size": strategy.tile_size,
            "codec": strategy.codec,
            "crf": strategy.crf,
            "scenes_total": len(strategy.scenes),
            "scenes_static": static,
            "scenes_gradual": gradual,
            "scenes_dynamic": dynamic,
        }
