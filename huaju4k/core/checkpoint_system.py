"""
Checkpoint system for huaju4k video enhancement.

This module provides checkpoint creation, saving, loading, and integrity verification
for video processing operations to enable recovery from interruptions.
"""

import os
import json
import hashlib
import logging
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from ..models.data_models import CheckpointData, ProcessingStrategy
from ..utils.system_utils import ensure_directory

logger = logging.getLogger(__name__)


class CheckpointSystem:
    """
    Checkpoint system for video processing operations.
    
    Provides checkpoint creation, saving, loading, and integrity verification
    to enable recovery from processing interruptions.
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize checkpoint system.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file extension
        self.checkpoint_ext = ".checkpoint"
        
        # Maximum number of checkpoints to keep per operation
        self.max_checkpoints_per_operation = 10
        
        logger.info(f"Checkpoint system initialized: {self.checkpoint_dir}")
    
    def create_checkpoint(self, 
                         processor_state: Dict[str, Any],
                         processing_progress: float,
                         current_stage: str,
                         input_path: str,
                         output_path: str,
                         strategy: ProcessingStrategy,
                         metadata: Optional[Dict[str, Any]] = None) -> CheckpointData:
        """
        Create a new checkpoint.
        
        Args:
            processor_state: Current state of the video processor
            processing_progress: Progress as float 0.0-1.0
            current_stage: Current processing stage name
            input_path: Path to input video file
            output_path: Path to output video file
            strategy: Processing strategy configuration
            metadata: Additional metadata
            
        Returns:
            CheckpointData object
        """
        try:
            # Generate unique checkpoint ID
            timestamp = datetime.now()
            checkpoint_id = self._generate_checkpoint_id(input_path, current_stage, timestamp)
            
            # Create checkpoint data
            checkpoint = CheckpointData(
                checkpoint_id=checkpoint_id,
                timestamp=timestamp,
                processor_state=processor_state.copy(),
                processing_progress=processing_progress,
                current_stage=current_stage,
                input_path=input_path,
                output_path=output_path,
                strategy=strategy,
                metadata=metadata or {}
            )
            
            # Add system metadata
            checkpoint.metadata.update({
                'created_by': 'huaju4k_checkpoint_system',
                'version': '1.0',
                'input_file_size': self._get_file_size(input_path),
                'input_file_hash': self._calculate_file_hash(input_path)
            })
            
            logger.info(f"Created checkpoint: {checkpoint_id} for stage '{current_stage}' "
                       f"at {processing_progress:.1%} progress")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    def save_checkpoint(self, checkpoint: CheckpointData) -> bool:
        """
        Save checkpoint to persistent storage.
        
        Args:
            checkpoint: Checkpoint data to save
            
        Returns:
            True if saved successfully
        """
        try:
            # Validate checkpoint
            if not checkpoint.is_valid():
                logger.error(f"Invalid checkpoint data: {checkpoint.checkpoint_id}")
                return False
            
            # Create checkpoint file path
            checkpoint_file = self.checkpoint_dir / f"{checkpoint.checkpoint_id}{self.checkpoint_ext}"
            
            # Prepare checkpoint data for serialization
            checkpoint_dict = self._serialize_checkpoint(checkpoint)
            
            # Add integrity hash
            checkpoint_dict['integrity_hash'] = self._calculate_checkpoint_hash(checkpoint_dict)
            
            # Write checkpoint file
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_dict, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved checkpoint: {checkpoint.checkpoint_id} to {checkpoint_file}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints(checkpoint.input_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint.checkpoint_id}: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointData]:
        """
        Load checkpoint from persistent storage.
        
        Args:
            checkpoint_id: ID of checkpoint to load
            
        Returns:
            CheckpointData object or None if not found/invalid
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}{self.checkpoint_ext}"
            
            if not checkpoint_file.exists():
                logger.warning(f"Checkpoint file not found: {checkpoint_file}")
                return None
            
            # Load checkpoint data
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_dict = json.load(f)
            
            # Verify integrity
            if not self._verify_checkpoint_integrity(checkpoint_dict):
                logger.error(f"Checkpoint integrity verification failed: {checkpoint_id}")
                return None
            
            # Deserialize checkpoint
            checkpoint = self._deserialize_checkpoint(checkpoint_dict)
            
            # Additional validation
            if not checkpoint.is_valid():
                logger.error(f"Loaded checkpoint is invalid: {checkpoint_id}")
                return None
            
            logger.info(f"Loaded checkpoint: {checkpoint_id} from {checkpoint_file}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    def list_checkpoints(self, input_path: Optional[str] = None) -> List[CheckpointData]:
        """
        List available checkpoints.
        
        Args:
            input_path: Filter by input path (optional)
            
        Returns:
            List of CheckpointData objects
        """
        checkpoints = []
        
        try:
            # Find all checkpoint files
            checkpoint_files = list(self.checkpoint_dir.glob(f"*{self.checkpoint_ext}"))
            
            for checkpoint_file in checkpoint_files:
                try:
                    # Extract checkpoint ID from filename
                    checkpoint_id = checkpoint_file.stem
                    
                    # Load checkpoint
                    checkpoint = self.load_checkpoint(checkpoint_id)
                    if checkpoint is None:
                        continue
                    
                    # Filter by input path if specified
                    if input_path and checkpoint.input_path != input_path:
                        continue
                    
                    checkpoints.append(checkpoint)
                    
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint from {checkpoint_file}: {e}")
                    continue
            
            # Sort by timestamp (newest first)
            checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
            
            logger.info(f"Found {len(checkpoints)} valid checkpoints")
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}{self.checkpoint_ext}"
            
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info(f"Deleted checkpoint: {checkpoint_id}")
                return True
            else:
                logger.warning(f"Checkpoint file not found for deletion: {checkpoint_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    def verify_checkpoint_integrity(self, checkpoint_id: str) -> bool:
        """
        Verify checkpoint integrity.
        
        Args:
            checkpoint_id: ID of checkpoint to verify
            
        Returns:
            True if checkpoint is valid and intact
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}{self.checkpoint_ext}"
            
            if not checkpoint_file.exists():
                logger.warning(f"Checkpoint file not found: {checkpoint_id}")
                return False
            
            # Load checkpoint data
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_dict = json.load(f)
            
            # Verify integrity hash
            if not self._verify_checkpoint_integrity(checkpoint_dict):
                logger.error(f"Checkpoint integrity verification failed: {checkpoint_id}")
                return False
            
            # Verify input file still exists and matches
            input_path = checkpoint_dict.get('input_path')
            if input_path and Path(input_path).exists():
                current_hash = self._calculate_file_hash(input_path)
                stored_hash = checkpoint_dict.get('metadata', {}).get('input_file_hash')
                
                if stored_hash and current_hash != stored_hash:
                    logger.warning(f"Input file hash mismatch for checkpoint {checkpoint_id}")
                    return False
            
            logger.info(f"Checkpoint integrity verified: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify checkpoint integrity {checkpoint_id}: {e}")
            return False
    
    def cleanup_checkpoints(self, max_age_days: int = 30) -> int:
        """
        Clean up old checkpoints.
        
        Args:
            max_age_days: Maximum age of checkpoints to keep
            
        Returns:
            Number of checkpoints deleted
        """
        deleted_count = 0
        
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
            checkpoint_files = list(self.checkpoint_dir.glob(f"*{self.checkpoint_ext}"))
            
            for checkpoint_file in checkpoint_files:
                try:
                    # Check file modification time
                    if checkpoint_file.stat().st_mtime < cutoff_time:
                        checkpoint_id = checkpoint_file.stem
                        if self.delete_checkpoint(checkpoint_id):
                            deleted_count += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to check/delete old checkpoint {checkpoint_file}: {e}")
                    continue
            
            logger.info(f"Cleaned up {deleted_count} old checkpoints")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")
            return 0
    
    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Get checkpoint information without loading full data.
        
        Args:
            checkpoint_id: ID of checkpoint
            
        Returns:
            Dictionary with checkpoint information
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}{self.checkpoint_ext}"
            
            if not checkpoint_file.exists():
                return None
            
            # Load only metadata
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_dict = json.load(f)
            
            info = {
                'checkpoint_id': checkpoint_dict.get('checkpoint_id'),
                'timestamp': checkpoint_dict.get('timestamp'),
                'current_stage': checkpoint_dict.get('current_stage'),
                'processing_progress': checkpoint_dict.get('processing_progress'),
                'input_path': checkpoint_dict.get('input_path'),
                'output_path': checkpoint_dict.get('output_path'),
                'file_size': checkpoint_file.stat().st_size,
                'metadata': checkpoint_dict.get('metadata', {})
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get checkpoint info {checkpoint_id}: {e}")
            return None
    
    def _generate_checkpoint_id(self, input_path: str, stage: str, timestamp: datetime) -> str:
        """Generate unique checkpoint ID."""
        # Use input filename, stage, and timestamp
        input_name = Path(input_path).stem
        time_str = timestamp.strftime('%Y%m%d_%H%M%S')
        
        # Add hash for uniqueness
        hash_input = f"{input_path}_{stage}_{timestamp.isoformat()}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        return f"{input_name}_{stage}_{time_str}_{hash_suffix}"
    
    def _serialize_checkpoint(self, checkpoint: CheckpointData) -> Dict[str, Any]:
        """Serialize checkpoint data to dictionary."""
        checkpoint_dict = asdict(checkpoint)
        
        # Convert datetime to ISO string
        checkpoint_dict['timestamp'] = checkpoint.timestamp.isoformat()
        
        # Serialize strategy
        checkpoint_dict['strategy'] = asdict(checkpoint.strategy)
        
        return checkpoint_dict
    
    def _deserialize_checkpoint(self, checkpoint_dict: Dict[str, Any]) -> CheckpointData:
        """Deserialize checkpoint data from dictionary."""
        # Convert timestamp back to datetime
        timestamp_str = checkpoint_dict['timestamp']
        timestamp = datetime.fromisoformat(timestamp_str)
        
        # Deserialize strategy
        strategy_dict = checkpoint_dict['strategy']
        strategy = ProcessingStrategy(**strategy_dict)
        
        # Create checkpoint object
        checkpoint = CheckpointData(
            checkpoint_id=checkpoint_dict['checkpoint_id'],
            timestamp=timestamp,
            processor_state=checkpoint_dict['processor_state'],
            processing_progress=checkpoint_dict['processing_progress'],
            current_stage=checkpoint_dict['current_stage'],
            input_path=checkpoint_dict['input_path'],
            output_path=checkpoint_dict['output_path'],
            strategy=strategy,
            metadata=checkpoint_dict.get('metadata', {})
        )
        
        return checkpoint
    
    def _calculate_checkpoint_hash(self, checkpoint_dict: Dict[str, Any]) -> str:
        """Calculate integrity hash for checkpoint data."""
        # Create a copy without the hash field
        data_for_hash = checkpoint_dict.copy()
        data_for_hash.pop('integrity_hash', None)
        
        # Convert to JSON string for consistent hashing
        json_str = json.dumps(data_for_hash, sort_keys=True, default=str)
        
        # Calculate SHA256 hash
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _verify_checkpoint_integrity(self, checkpoint_dict: Dict[str, Any]) -> bool:
        """Verify checkpoint integrity using hash."""
        stored_hash = checkpoint_dict.get('integrity_hash')
        if not stored_hash:
            logger.warning("No integrity hash found in checkpoint")
            return False
        
        calculated_hash = self._calculate_checkpoint_hash(checkpoint_dict)
        return stored_hash == calculated_hash
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {e}")
            return "unknown"
    
    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        try:
            return Path(file_path).stat().st_size
        except Exception:
            return 0
    
    def _cleanup_old_checkpoints(self, input_path: str) -> None:
        """Clean up old checkpoints for a specific input file."""
        try:
            # Get all checkpoints for this input
            checkpoints = self.list_checkpoints(input_path)
            
            # Keep only the most recent N checkpoints
            if len(checkpoints) > self.max_checkpoints_per_operation:
                # Sort by timestamp (newest first)
                checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
                
                # Delete excess checkpoints
                for checkpoint in checkpoints[self.max_checkpoints_per_operation:]:
                    self.delete_checkpoint(checkpoint.checkpoint_id)
                    
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints for {input_path}: {e}")