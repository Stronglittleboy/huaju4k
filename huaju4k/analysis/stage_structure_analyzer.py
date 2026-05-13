"""
Stage Structure Analyzer for Theater Enhancement

This module analyzes video structure to extract objective numerical features
for theater-grade enhancement strategy generation.

Two-level scene segmentation:
  Level 1 — Abrupt change detection via frame difference thresholding
  Level 2 — Gradual change detection via sliding-window brightness analysis
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

from ..models.data_models import StructureFeatures, SceneSegment

logger = logging.getLogger(__name__)

THUMBNAIL_HEIGHT = 320
GRADUAL_WINDOW_SEC = 5
GRADUAL_BRIGHTNESS_THRESHOLD = 0.15
STATIC_DIFF_THRESHOLD = 0.02


class StageStructureAnalyzer:
    """
    Analyzes stage structure features from video files.

    Performs two-level scene segmentation (abrupt + gradual change detection)
    and extracts global lighting/noise/edge statistics.
    """

    def __init__(self, sample_frames: int = 30):
        self.sample_frames = sample_frames

    def analyze_structure(self, video_path: str) -> StructureFeatures:
        """
        Main analysis entry point.

        Runs two-level scene segmentation and global frame sampling,
        returning a populated StructureFeatures with per-scene classification.
        """
        logger.info("Starting stage structure analysis for: %s", video_path)

        if not Path(video_path).exists():
            raise RuntimeError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            logger.info(
                "Video info: %dx%d, %.1ffps, %.1fs, %d frames",
                width, height, fps, duration, total_frames,
            )

            # --- Two-level scene segmentation ---
            boundaries = self._detect_scene_boundaries(cap, total_frames, fps)
            segments = self._detect_gradual_changes(
                cap, boundaries, total_frames, fps,
            )
            scenes = self._classify_segments(cap, segments, fps)
            logger.info("Scene segmentation: %d scenes detected", len(scenes))

            # --- Global statistics from sampled frames ---
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frames = self._sample_frames(cap, total_frames)
            if not frames:
                raise RuntimeError("No frames could be sampled from video")

            logger.info("Sampled %d frames for global analysis", len(frames))

            lighting = self._analyze_lighting_structure(frames)
            edge_density = self._analyze_edge_density(frames)
            motion = self._analyze_frame_changes(frames)
            noise_score = self._analyze_noise_level(frames)

            features = StructureFeatures(
                resolution=(width, height),
                fps=fps,
                duration=duration,
                total_frames=total_frames,
                is_static_camera=motion["frame_diff_mean"] < STATIC_DIFF_THRESHOLD,
                highlight_ratio=lighting["highlight_ratio"],
                dark_ratio=lighting["dark_ratio"],
                midtone_ratio=lighting["midtone_ratio"],
                edge_density=edge_density,
                frame_diff_mean=motion["frame_diff_mean"],
                noise_score=noise_score,
                scenes=scenes,
                sample_frames=len(frames),
                analysis_timestamp=datetime.now(),
            )

            logger.info(
                "Analysis completed: static_camera=%s, highlight_ratio=%.3f, "
                "edge_density=%.3f, scenes=%d",
                features.is_static_camera,
                features.highlight_ratio,
                features.edge_density,
                len(features.scenes),
            )
            return features

        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Two-level scene segmentation
    # ------------------------------------------------------------------

    def _read_thumbnail(
        self, cap: cv2.VideoCapture, frame_idx: int,
    ) -> np.ndarray | None:
        """Read a single frame resized to THUMBNAIL_HEIGHT for fast processing."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        h, w = frame.shape[:2]
        scale = THUMBNAIL_HEIGHT / h
        new_w = max(1, int(w * scale))
        return cv2.resize(frame, (new_w, THUMBNAIL_HEIGHT), interpolation=cv2.INTER_AREA)

    def _detect_scene_boundaries(
        self, cap: cv2.VideoCapture, total_frames: int, fps: float,
    ) -> List[int]:
        """
        Level 1: Abrupt change detection.

        Scans every frame at thumbnail resolution, computing frame-to-frame
        absolute difference. Boundaries are placed where the difference
        exceeds mean + 3*sigma.

        Returns a sorted list of boundary frame indices (always includes 0).
        """
        logger.info("Level 1: scanning for abrupt scene boundaries")
        diffs: List[float] = []
        prev_gray = None

        for idx in range(total_frames):
            thumb = self._read_thumbnail(cap, idx)
            if thumb is None:
                diffs.append(0.0)
                continue

            gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                diffs.append(float(np.mean(diff.astype(np.float32) / 255.0)))
            else:
                diffs.append(0.0)
            prev_gray = gray

            if idx > 0 and idx % 5000 == 0:
                logger.info("  scanned %d / %d frames", idx, total_frames)

        diff_arr = np.array(diffs, dtype=np.float64)
        mean_d = float(np.mean(diff_arr))
        sigma_d = float(np.std(diff_arr))
        threshold = mean_d + 3.0 * sigma_d

        boundaries = [0]
        for i, d in enumerate(diffs):
            if d > threshold:
                boundaries.append(i)

        boundaries = sorted(set(boundaries))
        logger.info(
            "Level 1 complete: %d hard-cut boundaries (threshold=%.4f)",
            len(boundaries) - 1, threshold,
        )
        return boundaries

    def _detect_gradual_changes(
        self,
        cap: cv2.VideoCapture,
        boundaries: List[int],
        total_frames: int,
        fps: float,
    ) -> List[Tuple[int, int]]:
        """
        Level 2: Gradual change detection within Level 1 segments.

        For each segment, uses a sliding window of GRADUAL_WINDOW_SEC seconds.
        If the background mean brightness changes monotonically by more than
        GRADUAL_BRIGHTNESS_THRESHOLD within the window, the segment is split
        at the gradual change boundaries.

        Returns a list of (start_frame, end_frame) tuples.
        """
        logger.info("Level 2: detecting gradual brightness changes")
        window_frames = max(2, int(fps * GRADUAL_WINDOW_SEC))

        seg_endpoints = []
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else total_frames
            seg_endpoints.append((start, end))

        result_segments: List[Tuple[int, int]] = []

        for seg_start, seg_end in seg_endpoints:
            seg_len = seg_end - seg_start
            if seg_len < window_frames:
                result_segments.append((seg_start, seg_end))
                continue

            sample_step = max(1, int(fps))
            brightness: List[Tuple[int, float]] = []
            for idx in range(seg_start, seg_end, sample_step):
                thumb = self._read_thumbnail(cap, idx)
                if thumb is None:
                    continue
                gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
                brightness.append((idx, float(np.mean(gray) / 255.0)))

            if len(brightness) < 2:
                result_segments.append((seg_start, seg_end))
                continue

            window_samples = max(2, GRADUAL_WINDOW_SEC)
            split_points = [seg_start]

            for wi in range(len(brightness) - window_samples + 1):
                win = brightness[wi : wi + window_samples]
                vals = [v for _, v in win]
                change = abs(vals[-1] - vals[0])
                if change < GRADUAL_BRIGHTNESS_THRESHOLD:
                    continue
                diffs = [vals[j + 1] - vals[j] for j in range(len(vals) - 1)]
                if all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs):
                    mid_idx = win[len(win) // 2][0]
                    if mid_idx > split_points[-1] + window_frames:
                        split_points.append(mid_idx)

            split_points.append(seg_end)
            for si in range(len(split_points) - 1):
                result_segments.append((split_points[si], split_points[si + 1]))

        result_segments.sort()
        logger.info("Level 2 complete: %d segments after gradual split", len(result_segments))
        return result_segments

    def _classify_segments(
        self,
        cap: cv2.VideoCapture,
        segments: List[Tuple[int, int]],
        fps: float,
    ) -> List[SceneSegment]:
        """
        Classify each segment as 'static', 'gradual', or 'dynamic'.

        - static:  low frame_diff_mean AND no significant brightness change
        - gradual: low frame_diff_mean BUT significant brightness change
        - dynamic: high frame_diff_mean
        """
        scenes: List[SceneSegment] = []
        sample_count = 10

        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            if seg_len <= 0:
                continue

            step = max(1, seg_len // sample_count)
            indices = list(range(seg_start, seg_end, step))[:sample_count]

            grays: List[np.ndarray] = []
            brightnesses: List[float] = []
            noise_vals: List[float] = []

            for idx in indices:
                thumb = self._read_thumbnail(cap, idx)
                if thumb is None:
                    continue
                gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
                grays.append(gray)
                brightnesses.append(float(np.mean(gray) / 255.0))
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                noise_vals.append(min(1.0, float(lap.var()) / 1000.0))

            if not grays:
                scenes.append(SceneSegment(
                    start_frame=seg_start, end_frame=seg_end,
                    type="dynamic", avg_brightness=0.0, noise_score=0.0,
                ))
                continue

            avg_brightness = float(np.mean(brightnesses))
            noise_score = float(np.mean(noise_vals))

            pair_diffs: List[float] = []
            for i in range(1, len(grays)):
                diff = cv2.absdiff(grays[i - 1], grays[i])
                pair_diffs.append(float(np.mean(diff.astype(np.float32) / 255.0)))

            diff_mean = float(np.mean(pair_diffs)) if pair_diffs else 0.0

            brightness_range = max(brightnesses) - min(brightnesses) if brightnesses else 0.0
            monotonic = False
            if len(brightnesses) >= 3:
                diffs_b = [brightnesses[j + 1] - brightnesses[j] for j in range(len(brightnesses) - 1)]
                monotonic = all(d >= 0 for d in diffs_b) or all(d <= 0 for d in diffs_b)

            if diff_mean >= STATIC_DIFF_THRESHOLD:
                seg_type = "dynamic"
            elif brightness_range >= GRADUAL_BRIGHTNESS_THRESHOLD and monotonic:
                seg_type = "gradual"
            else:
                seg_type = "static"

            scenes.append(SceneSegment(
                start_frame=seg_start,
                end_frame=seg_end,
                type=seg_type,
                avg_brightness=avg_brightness,
                noise_score=noise_score,
            ))

        return scenes

    # ------------------------------------------------------------------
    # Global frame sampling & statistics (existing logic)
    # ------------------------------------------------------------------

    def _sample_frames(
        self, cap: cv2.VideoCapture, total_frames: int,
    ) -> List[np.ndarray]:
        """Sample frames uniformly from the video for global statistics."""
        frames: List[np.ndarray] = []

        if total_frames <= self.sample_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames // self.sample_frames
            frame_indices = [i * step for i in range(self.sample_frames)]

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
            else:
                logger.warning("Failed to read frame at index %d", frame_idx)

        return frames

    def _analyze_lighting_structure(
        self, frames: List[np.ndarray],
    ) -> Dict[str, float]:
        """Analyze lighting structure: highlight / dark / midtone ratios."""
        highlight_ratios: List[float] = []
        dark_ratios: List[float] = []
        midtone_ratios: List[float] = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            normalized = gray.astype(np.float32) / 255.0

            highlight_mask = normalized > 0.85
            dark_mask = normalized < 0.15
            midtone_mask = (normalized >= 0.15) & (normalized <= 0.85)

            total_pixels = normalized.size
            highlight_ratios.append(float(np.sum(highlight_mask) / total_pixels))
            dark_ratios.append(float(np.sum(dark_mask) / total_pixels))
            midtone_ratios.append(float(np.sum(midtone_mask) / total_pixels))

        return {
            "highlight_ratio": float(np.mean(highlight_ratios)),
            "dark_ratio": float(np.mean(dark_ratios)),
            "midtone_ratio": float(np.mean(midtone_ratios)),
        }

    def _analyze_edge_density(self, frames: List[np.ndarray]) -> float:
        """Analyze edge density using Canny edge detector."""
        edge_densities: List[float] = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.sum(edges > 0)
            edge_densities.append(float(edge_pixels / edges.size))

        return float(np.mean(edge_densities))

    def _analyze_frame_changes(
        self, frames: List[np.ndarray],
    ) -> Dict[str, float]:
        """Analyze frame-to-frame changes to estimate global motion."""
        if len(frames) < 2:
            return {"frame_diff_mean": 0.0}

        frame_diffs: List[float] = []

        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, curr_gray)
            diff_normalized = diff.astype(np.float32) / 255.0
            frame_diffs.append(float(np.mean(diff_normalized)))

        return {"frame_diff_mean": float(np.mean(frame_diffs))}

    def _analyze_noise_level(self, frames: List[np.ndarray]) -> float:
        """Analyze noise level using Laplacian variance method."""
        noise_scores: List[float] = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            noise_scores.append(min(1.0, float(variance / 1000.0)))

        return float(np.mean(noise_scores))
