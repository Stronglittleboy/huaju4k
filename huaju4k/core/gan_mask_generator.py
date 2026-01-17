"""
GAN safe mask generator for huaju4k theater enhancement.

This module implements multi-dimensional safe region detection for controlled
GAN enhancement, ensuring AI enhancement is only applied to appropriate areas
while avoiding problematic regions like highlights, shadows, and high-motion areas.
"""

import logging
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from ..models.data_models import GANPolicy

logger = logging.getLogger(__name__)


@dataclass
class MaskGenerationConfig:
    """Configuration for mask generation parameters."""
    
    # Canny edge detection parameters
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    
    # Edge density calculation
    edge_kernel_size: int = 15  # 15x15 neighborhood
    
    # Morphological operations
    morph_kernel_size: int = 5  # 5x5 elliptical kernel
    
    # Minimum region size for processing
    min_region_size: int = 32  # 32x32 pixels
    
    # Default thresholds (can be overridden by GANPolicy)
    default_highlight_threshold: float = 0.85
    default_shadow_threshold: float = 0.15
    default_edge_threshold: float = 0.3
    default_motion_threshold: float = 0.05


class GANSafeMaskGenerator:
    """
    Multi-dimensional GAN safe region mask generator.
    
    This class generates boolean masks indicating safe regions for GAN enhancement
    based on multiple criteria:
    1. Brightness exclusion (avoid highlights and deep shadows)
    2. Edge density analysis (prefer edge-rich regions)
    3. Motion detection (handle moving vs static regions)
    4. GAN strength-based region expansion
    
    The implementation follows the exact algorithm specifications from Task 4.
    """
    
    def __init__(self, config: Optional[MaskGenerationConfig] = None):
        """
        Initialize GAN safe mask generator.
        
        Args:
            config: Optional configuration for mask generation parameters
        """
        self.config = config or MaskGenerationConfig()
        logger.info("GANSafeMaskGenerator initialized")
    
    def generate_multi_dimensional_safe_mask(self, 
                                           frame: np.ndarray,
                                           previous_frame: Optional[np.ndarray],
                                           gan_policy: GANPolicy) -> np.ndarray:
        """
        Generate multi-dimensional GAN safe region mask.
        
        This method implements the complete safe region detection algorithm
        as specified in Task 4, including brightness exclusion, edge density
        calculation, motion detection, and GAN strength-based adjustments.
        
        Args:
            frame: Current frame as numpy array (H, W, C)
            previous_frame: Previous frame for motion detection (optional)
            gan_policy: GAN policy configuration with thresholds
            
        Returns:
            Boolean mask where True indicates safe regions for GAN enhancement
        """
        try:
            if frame is None or frame.size == 0:
                logger.warning("Invalid frame provided, returning empty mask")
                return np.zeros((480, 640), dtype=bool)
            
            height, width = frame.shape[:2]
            
            # Convert to grayscale for analysis
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_normalized = frame_gray.astype(np.float32) / 255.0
            
            logger.debug(f"Generating safe mask for frame: {width}x{height}")
            
            # 1. Brightness exclusion mask
            highlight_mask, shadow_mask = self._generate_brightness_masks(
                frame_normalized, gan_policy
            )
            
            # 2. Edge density mask
            edge_mask = self._generate_edge_density_mask(frame_gray, gan_policy)
            
            # 3. Motion detection mask
            motion_mask = self._generate_motion_mask(
                frame_gray, previous_frame, gan_policy
            )
            
            # 4. Combine masks based on GAN strength
            safe_mask = self._combine_masks_by_strength(
                edge_mask, highlight_mask, shadow_mask, motion_mask, gan_policy
            )
            
            # 5. Apply morphological operations for smoothing
            safe_mask = self._apply_morphological_smoothing(safe_mask)
            
            # 6. Validate and filter small regions
            safe_mask = self._filter_small_regions(safe_mask)
            
            # Log mask statistics
            safe_ratio = np.sum(safe_mask) / safe_mask.size
            logger.debug(f"Safe mask generated: {safe_ratio:.2%} of frame marked as safe")
            
            return safe_mask.astype(bool)
            
        except Exception as e:
            logger.error(f"Safe mask generation failed: {e}")
            # Return conservative empty mask on error
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
    
    def _generate_brightness_masks(self, 
                                 frame_normalized: np.ndarray,
                                 gan_policy: GANPolicy) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate brightness exclusion masks for highlights and shadows.
        
        Args:
            frame_normalized: Normalized grayscale frame (0.0-1.0)
            gan_policy: GAN policy with brightness thresholds
            
        Returns:
            Tuple of (highlight_mask, shadow_mask) where True indicates exclusion areas
        """
        try:
            # Use policy thresholds or defaults
            highlight_threshold = getattr(gan_policy, 'highlight_threshold', 
                                        self.config.default_highlight_threshold)
            shadow_threshold = getattr(gan_policy, 'shadow_threshold',
                                     self.config.default_shadow_threshold)
            
            # Generate exclusion masks
            highlight_mask = frame_normalized > highlight_threshold
            shadow_mask = frame_normalized < shadow_threshold
            
            highlight_ratio = np.sum(highlight_mask) / highlight_mask.size
            shadow_ratio = np.sum(shadow_mask) / shadow_mask.size
            
            logger.debug(f"Brightness masks: highlights={highlight_ratio:.2%}, "
                        f"shadows={shadow_ratio:.2%}")
            
            return highlight_mask, shadow_mask
            
        except Exception as e:
            logger.error(f"Brightness mask generation failed: {e}")
            # Return empty masks on error
            height, width = frame_normalized.shape
            return (np.zeros((height, width), dtype=bool),
                   np.zeros((height, width), dtype=bool))
    
    def _generate_edge_density_mask(self, 
                                  frame_gray: np.ndarray,
                                  gan_policy: GANPolicy) -> np.ndarray:
        """
        Generate edge density mask using Canny edge detection and neighborhood analysis.
        
        Args:
            frame_gray: Grayscale frame (0-255)
            gan_policy: GAN policy with edge threshold
            
        Returns:
            Boolean mask where True indicates high edge density regions
        """
        try:
            # Apply Canny edge detection with specified parameters
            edges = cv2.Canny(
                frame_gray,
                self.config.canny_low_threshold,
                self.config.canny_high_threshold
            )
            
            # Create kernel for neighborhood density calculation
            kernel_size = self.config.edge_kernel_size
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            
            # Calculate edge density in 15x15 neighborhoods
            edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel) / 255.0
            
            # Apply threshold from GAN policy
            edge_threshold = getattr(gan_policy, 'edge_threshold',
                                   self.config.default_edge_threshold)
            
            edge_mask = edge_density > edge_threshold
            
            edge_ratio = np.sum(edge_mask) / edge_mask.size
            logger.debug(f"Edge density mask: {edge_ratio:.2%} high-edge regions, "
                        f"threshold={edge_threshold}")
            
            return edge_mask
            
        except Exception as e:
            logger.error(f"Edge density mask generation failed: {e}")
            # Return empty mask on error
            return np.zeros(frame_gray.shape, dtype=bool)
    
    def _generate_motion_mask(self, 
                            frame_gray: np.ndarray,
                            previous_frame: Optional[np.ndarray],
                            gan_policy: GANPolicy) -> np.ndarray:
        """
        Generate motion detection mask using frame difference.
        
        Args:
            frame_gray: Current grayscale frame
            previous_frame: Previous frame for comparison (optional)
            gan_policy: GAN policy with motion threshold
            
        Returns:
            Boolean mask where True indicates motion regions
        """
        try:
            height, width = frame_gray.shape
            
            # Default to allowing all regions if no previous frame
            if previous_frame is None:
                logger.debug("No previous frame, allowing all regions for motion")
                return np.ones((height, width), dtype=bool)
            
            # Convert previous frame to grayscale if needed
            if len(previous_frame.shape) == 3:
                prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            else:
                prev_gray = previous_frame
            
            # Ensure frames have same dimensions
            if prev_gray.shape != frame_gray.shape:
                prev_gray = cv2.resize(prev_gray, (width, height))
            
            # Calculate frame difference
            frame_diff = cv2.absdiff(frame_gray, prev_gray).astype(np.float32) / 255.0
            
            # Apply motion threshold
            motion_threshold = getattr(gan_policy, 'motion_threshold',
                                     self.config.default_motion_threshold)
            
            motion_mask = frame_diff > motion_threshold
            
            motion_ratio = np.sum(motion_mask) / motion_mask.size
            logger.debug(f"Motion mask: {motion_ratio:.2%} motion regions, "
                        f"threshold={motion_threshold}")
            
            return motion_mask
            
        except Exception as e:
            logger.error(f"Motion mask generation failed: {e}")
            # Return conservative all-motion mask on error
            return np.ones(frame_gray.shape, dtype=bool)
    
    def _combine_masks_by_strength(self, 
                                 edge_mask: np.ndarray,
                                 highlight_mask: np.ndarray,
                                 shadow_mask: np.ndarray,
                                 motion_mask: np.ndarray,
                                 gan_policy: GANPolicy) -> np.ndarray:
        """
        Combine individual masks based on GAN strength policy.
        
        This implements the exact strength-based combination logic from Task 4:
        - weak: Only edge + motion regions
        - medium: Edge + some static regions (avoiding highlights)
        - strong: Larger range but still avoiding highlights and shadows
        
        Args:
            edge_mask: High edge density regions
            highlight_mask: Highlight exclusion regions
            shadow_mask: Shadow exclusion regions  
            motion_mask: Motion detection regions
            gan_policy: GAN policy with strength setting
            
        Returns:
            Combined safe region mask
        """
        try:
            # Start with base safe regions (edges, avoiding highlights and shadows)
            safe_mask = edge_mask & (~highlight_mask) & (~shadow_mask)
            
            # Apply strength-specific combination rules
            strength = gan_policy.strength.lower()
            
            if strength == 'weak':
                # Weak: Only edge + motion regions (most conservative)
                safe_mask = safe_mask & motion_mask
                logger.debug("Applied weak GAN strength: edge + motion only")
                
            elif strength == 'medium':
                # Medium: Edge + some static regions (avoiding highlights)
                static_safe = motion_mask & (~highlight_mask)
                safe_mask = safe_mask | static_safe
                logger.debug("Applied medium GAN strength: edge + static (no highlights)")
                
            elif strength == 'strong':
                # Strong: Larger range but avoid highlights and deep shadows
                expanded_safe = (~highlight_mask) & (~shadow_mask)
                safe_mask = safe_mask | expanded_safe
                logger.debug("Applied strong GAN strength: expanded safe regions")
                
            else:
                logger.warning(f"Unknown GAN strength '{strength}', using medium")
                # Default to medium strength behavior
                static_safe = motion_mask & (~highlight_mask)
                safe_mask = safe_mask | static_safe
            
            safe_ratio = np.sum(safe_mask) / safe_mask.size
            logger.debug(f"Combined mask ({strength} strength): {safe_ratio:.2%} safe regions")
            
            return safe_mask
            
        except Exception as e:
            logger.error(f"Mask combination failed: {e}")
            # Return conservative edge-only mask on error
            return edge_mask & (~highlight_mask) & (~shadow_mask)
    
    def _apply_morphological_smoothing(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to smooth mask boundaries.
        
        Uses 5x5 elliptical kernel with closing followed by opening operations
        as specified in Task 4.
        
        Args:
            mask: Input boolean mask
            
        Returns:
            Smoothed boolean mask
        """
        try:
            # Convert to uint8 for morphological operations
            mask_uint8 = mask.astype(np.uint8) * 255
            
            # Create 5x5 elliptical kernel as specified
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.config.morph_kernel_size, self.config.morph_kernel_size)
            )
            
            # Apply closing operation (fill small holes)
            mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            
            # Apply opening operation (remove small noise)
            mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
            
            # Convert back to boolean
            smoothed_mask = mask_opened > 127
            
            original_ratio = np.sum(mask) / mask.size
            smoothed_ratio = np.sum(smoothed_mask) / smoothed_mask.size
            
            logger.debug(f"Morphological smoothing: {original_ratio:.2%} -> {smoothed_ratio:.2%}")
            
            return smoothed_mask
            
        except Exception as e:
            logger.error(f"Morphological smoothing failed: {e}")
            return mask
    
    def _filter_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """
        Filter out regions smaller than minimum size threshold.
        
        Args:
            mask: Input boolean mask
            
        Returns:
            Filtered mask with small regions removed
        """
        try:
            # Convert to uint8 for contour detection
            mask_uint8 = mask.astype(np.uint8) * 255
            
            # Find connected components
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Create filtered mask
            filtered_mask = np.zeros_like(mask, dtype=bool)
            
            regions_kept = 0
            regions_filtered = 0
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if region meets minimum size requirement
                if w >= self.config.min_region_size and h >= self.config.min_region_size:
                    # Keep this region
                    cv2.fillPoly(filtered_mask.astype(np.uint8), [contour], True)
                    regions_kept += 1
                else:
                    regions_filtered += 1
            
            logger.debug(f"Region filtering: kept {regions_kept}, filtered {regions_filtered} small regions")
            
            return filtered_mask
            
        except Exception as e:
            logger.error(f"Region filtering failed: {e}")
            return mask
    
    def analyze_mask_quality(self, mask: np.ndarray, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the quality and characteristics of a generated mask.
        
        Args:
            mask: Generated boolean mask
            frame: Original frame for reference
            
        Returns:
            Dictionary with mask analysis results
        """
        try:
            total_pixels = mask.size
            safe_pixels = np.sum(mask)
            safe_ratio = safe_pixels / total_pixels
            
            # Find connected components for region analysis
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Analyze regions
            region_sizes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                region_sizes.append(area)
            
            analysis = {
                'total_pixels': total_pixels,
                'safe_pixels': safe_pixels,
                'safe_ratio': safe_ratio,
                'num_regions': len(contours),
                'avg_region_size': np.mean(region_sizes) if region_sizes else 0,
                'max_region_size': np.max(region_sizes) if region_sizes else 0,
                'min_region_size': np.min(region_sizes) if region_sizes else 0,
                'frame_dimensions': frame.shape[:2]
            }
            
            # Quality assessment
            if safe_ratio < 0.05:
                analysis['quality'] = 'very_conservative'
            elif safe_ratio < 0.15:
                analysis['quality'] = 'conservative'
            elif safe_ratio < 0.35:
                analysis['quality'] = 'balanced'
            elif safe_ratio < 0.60:
                analysis['quality'] = 'aggressive'
            else:
                analysis['quality'] = 'very_aggressive'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Mask quality analysis failed: {e}")
            return {
                'error': str(e),
                'safe_ratio': 0.0,
                'quality': 'unknown'
            }
    
    def visualize_mask_components(self, 
                                frame: np.ndarray,
                                previous_frame: Optional[np.ndarray],
                                gan_policy: GANPolicy) -> Dict[str, np.ndarray]:
        """
        Generate visualization of individual mask components for debugging.
        
        Args:
            frame: Current frame
            previous_frame: Previous frame (optional)
            gan_policy: GAN policy configuration
            
        Returns:
            Dictionary with individual mask components as images
        """
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_normalized = frame_gray.astype(np.float32) / 255.0
            
            # Generate individual components
            highlight_mask, shadow_mask = self._generate_brightness_masks(
                frame_normalized, gan_policy
            )
            edge_mask = self._generate_edge_density_mask(frame_gray, gan_policy)
            motion_mask = self._generate_motion_mask(frame_gray, previous_frame, gan_policy)
            
            # Generate final combined mask
            final_mask = self.generate_multi_dimensional_safe_mask(
                frame, previous_frame, gan_policy
            )
            
            # Convert to visualization images (0-255)
            visualizations = {
                'original_frame': frame,
                'highlight_exclusion': (highlight_mask * 255).astype(np.uint8),
                'shadow_exclusion': (shadow_mask * 255).astype(np.uint8),
                'edge_density': (edge_mask * 255).astype(np.uint8),
                'motion_regions': (motion_mask * 255).astype(np.uint8),
                'final_safe_mask': (final_mask * 255).astype(np.uint8)
            }
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Mask visualization failed: {e}")
            return {'error': str(e)}