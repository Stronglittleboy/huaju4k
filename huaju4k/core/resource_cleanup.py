"""
Resource cleanup mechanisms for huaju4k video enhancement.

This module provides automatic resource cleanup on completion/failure
and temporary file management system.
"""

import os
import atexit
import signal
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResourceTracker:
    """Track allocated resources for cleanup."""
    
    resource_id: str
    resource_type: str  # 'file', 'memory', 'process', 'gpu_memory'
    resource_path: Optional[Path] = None
    size_mb: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    cleanup_callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResourceCleanupManager:
    """
    Manages automatic resource cleanup on completion/failure.
    
    This class tracks all allocated resources and ensures they are
    properly cleaned up when processing completes or fails.
    """
    
    def __init__(self, auto_cleanup: bool = True):
        """
        Initialize resource cleanup manager.
        
        Args:
            auto_cleanup: Whether to automatically cleanup on exit
        """
        self.auto_cleanup = auto_cleanup
        self._resources: Dict[str, ResourceTracker] = {}
        self._cleanup_callbacks: List[Callable] = []
        self._lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Register cleanup handlers
        if auto_cleanup:
            atexit.register(self.cleanup_all)
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        
        # Start background cleanup thread
        self._start_cleanup_thread()
        
        logger.info("ResourceCleanupManager initialized")
    
    def register_resource(self, resource_id: str, resource_type: str,
                         resource_path: Optional[Path] = None,
                         size_mb: Optional[int] = None,
                         cleanup_callback: Optional[Callable] = None,
                         **metadata) -> None:
        """
        Register a resource for tracking and cleanup.
        
        Args:
            resource_id: Unique identifier for the resource
            resource_type: Type of resource ('file', 'memory', 'process', 'gpu_memory')
            resource_path: Path to resource (for files)
            size_mb: Size of resource in MB
            cleanup_callback: Custom cleanup function
            **metadata: Additional metadata
        """
        with self._lock:
            tracker = ResourceTracker(
                resource_id=resource_id,
                resource_type=resource_type,
                resource_path=resource_path,
                size_mb=size_mb,
                cleanup_callback=cleanup_callback,
                metadata=metadata
            )
            
            self._resources[resource_id] = tracker
            logger.debug(f"Registered resource: {resource_id} ({resource_type})")
    
    def unregister_resource(self, resource_id: str) -> bool:
        """
        Unregister a resource (without cleanup).
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            True if resource was found and removed
        """
        with self._lock:
            if resource_id in self._resources:
                del self._resources[resource_id]
                logger.debug(f"Unregistered resource: {resource_id}")
                return True
            return False
    
    def cleanup_resource(self, resource_id: str) -> bool:
        """
        Clean up a specific resource.
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            True if cleanup was successful
        """
        with self._lock:
            if resource_id not in self._resources:
                logger.warning(f"Resource not found for cleanup: {resource_id}")
                return False
            
            tracker = self._resources[resource_id]
            
        try:
            success = self._cleanup_single_resource(tracker)
            
            if success:
                with self._lock:
                    del self._resources[resource_id]
                logger.debug(f"Cleaned up resource: {resource_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error cleaning up resource {resource_id}: {e}")
            return False
    
    def cleanup_all(self) -> Dict[str, bool]:
        """
        Clean up all registered resources.
        
        Returns:
            Dictionary mapping resource IDs to cleanup success status
        """
        logger.info("Starting cleanup of all resources")
        
        results = {}
        
        # Get copy of resources to avoid modification during iteration
        with self._lock:
            resources_to_cleanup = list(self._resources.items())
        
        # Clean up resources in reverse order of registration
        for resource_id, tracker in reversed(resources_to_cleanup):
            try:
                success = self._cleanup_single_resource(tracker)
                results[resource_id] = success
                
                if success:
                    with self._lock:
                        self._resources.pop(resource_id, None)
                
            except Exception as e:
                logger.error(f"Error cleaning up resource {resource_id}: {e}")
                results[resource_id] = False
        
        # Execute additional cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")
        
        # Stop cleanup thread
        self._shutdown_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        
        successful_cleanups = sum(1 for success in results.values() if success)
        total_resources = len(results)
        
        logger.info(f"Cleanup completed: {successful_cleanups}/{total_resources} resources cleaned")
        
        return results
    
    def add_cleanup_callback(self, callback: Callable) -> None:
        """
        Add a callback to be executed during cleanup.
        
        Args:
            callback: Function to call during cleanup
        """
        self._cleanup_callbacks.append(callback)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """
        Get summary of currently tracked resources.
        
        Returns:
            Dictionary with resource statistics
        """
        with self._lock:
            resources_by_type = {}
            total_size_mb = 0
            
            for tracker in self._resources.values():
                resource_type = tracker.resource_type
                if resource_type not in resources_by_type:
                    resources_by_type[resource_type] = 0
                resources_by_type[resource_type] += 1
                
                if tracker.size_mb:
                    total_size_mb += tracker.size_mb
            
            return {
                'total_resources': len(self._resources),
                'resources_by_type': resources_by_type,
                'total_size_mb': total_size_mb,
                'oldest_resource': min(
                    (t.created_at for t in self._resources.values()),
                    default=None
                )
            }
    
    def cleanup_old_resources(self, max_age_hours: int = 24) -> int:
        """
        Clean up resources older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of resources cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        with self._lock:
            old_resources = [
                (resource_id, tracker) 
                for resource_id, tracker in self._resources.items()
                if tracker.created_at < cutoff_time
            ]
        
        for resource_id, tracker in old_resources:
            if self.cleanup_resource(resource_id):
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old resources")
        
        return cleaned_count
    
    def _cleanup_single_resource(self, tracker: ResourceTracker) -> bool:
        """Clean up a single resource based on its type."""
        try:
            if tracker.cleanup_callback:
                # Use custom cleanup callback
                tracker.cleanup_callback()
                return True
            
            elif tracker.resource_type == 'file':
                # Clean up file resource
                if tracker.resource_path and tracker.resource_path.exists():
                    if tracker.resource_path.is_file():
                        tracker.resource_path.unlink()
                    elif tracker.resource_path.is_dir():
                        import shutil
                        shutil.rmtree(tracker.resource_path)
                    return True
                return True  # File already gone
            
            elif tracker.resource_type == 'memory':
                # Memory cleanup (mostly handled by garbage collector)
                # Could implement specific memory release here
                return True
            
            elif tracker.resource_type == 'process':
                # Process cleanup
                pid = tracker.metadata.get('pid')
                if pid:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        return True
                    except ProcessLookupError:
                        return True  # Process already gone
                    except PermissionError:
                        logger.warning(f"No permission to kill process {pid}")
                        return False
                return True
            
            elif tracker.resource_type == 'gpu_memory':
                # GPU memory cleanup (implementation depends on GPU library)
                # This would need specific implementation based on CUDA/OpenCL
                return True
            
            else:
                logger.warning(f"Unknown resource type: {tracker.resource_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error cleaning up {tracker.resource_type} resource: {e}")
            return False
    
    def _start_cleanup_thread(self) -> None:
        """Start background thread for periodic cleanup."""
        def cleanup_worker():
            while not self._shutdown_event.wait(timeout=300):  # Check every 5 minutes
                try:
                    # Clean up old resources
                    self.cleanup_old_resources(max_age_hours=24)
                except Exception as e:
                    logger.error(f"Error in background cleanup: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for cleanup."""
        logger.info(f"Received signal {signum}, cleaning up resources")
        self.cleanup_all()


class TemporaryFileManager:
    """
    Manages temporary files with automatic cleanup.
    
    This class provides a context manager for temporary files
    and ensures they are cleaned up properly.
    """
    
    def __init__(self, temp_dir: Optional[Path] = None, 
                 cleanup_manager: Optional[ResourceCleanupManager] = None):
        """
        Initialize temporary file manager.
        
        Args:
            temp_dir: Directory for temporary files
            cleanup_manager: Resource cleanup manager to use
        """
        self.temp_dir = temp_dir or Path.cwd() / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.cleanup_manager = cleanup_manager or ResourceCleanupManager()
        self._temp_files: List[Path] = []
        
        logger.debug(f"TemporaryFileManager initialized with temp_dir: {self.temp_dir}")
    
    def create_temp_file(self, prefix: str = "huaju4k_", suffix: str = ".tmp",
                        content: Optional[bytes] = None) -> Path:
        """
        Create a temporary file.
        
        Args:
            prefix: File name prefix
            suffix: File name suffix
            content: Optional initial content
            
        Returns:
            Path to created temporary file
        """
        try:
            import tempfile
            
            # Create temporary file
            fd, temp_path = tempfile.mkstemp(
                prefix=prefix,
                suffix=suffix,
                dir=str(self.temp_dir)
            )
            
            # Write initial content if provided
            if content:
                os.write(fd, content)
            
            os.close(fd)
            
            temp_file = Path(temp_path)
            self._temp_files.append(temp_file)
            
            # Register with cleanup manager
            self.cleanup_manager.register_resource(
                resource_id=f"temp_file_{temp_file.name}",
                resource_type="file",
                resource_path=temp_file,
                size_mb=0  # Will be updated when file is written
            )
            
            logger.debug(f"Created temporary file: {temp_file}")
            return temp_file
            
        except Exception as e:
            logger.error(f"Error creating temporary file: {e}")
            raise
    
    def create_temp_dir(self, prefix: str = "huaju4k_") -> Path:
        """
        Create a temporary directory.
        
        Args:
            prefix: Directory name prefix
            
        Returns:
            Path to created temporary directory
        """
        try:
            import tempfile
            
            temp_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(self.temp_dir)))
            
            # Register with cleanup manager
            self.cleanup_manager.register_resource(
                resource_id=f"temp_dir_{temp_dir.name}",
                resource_type="file",
                resource_path=temp_dir
            )
            
            logger.debug(f"Created temporary directory: {temp_dir}")
            return temp_dir
            
        except Exception as e:
            logger.error(f"Error creating temporary directory: {e}")
            raise
    
    def cleanup_temp_files(self) -> int:
        """
        Clean up all temporary files created by this manager.
        
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        
        for temp_file in self._temp_files[:]:  # Copy to avoid modification during iteration
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    cleaned_count += 1
                self._temp_files.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} temporary files")
        return cleaned_count
    
    def get_temp_file_size(self, temp_file: Path) -> int:
        """
        Get size of temporary file in bytes.
        
        Args:
            temp_file: Path to temporary file
            
        Returns:
            File size in bytes
        """
        try:
            return temp_file.stat().st_size if temp_file.exists() else 0
        except Exception as e:
            logger.warning(f"Error getting file size for {temp_file}: {e}")
            return 0
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_temp_files()


# Global cleanup manager instance
_global_cleanup_manager: Optional[ResourceCleanupManager] = None


def get_global_cleanup_manager() -> ResourceCleanupManager:
    """Get or create global cleanup manager instance."""
    global _global_cleanup_manager
    
    if _global_cleanup_manager is None:
        _global_cleanup_manager = ResourceCleanupManager()
    
    return _global_cleanup_manager


def register_for_cleanup(resource_id: str, resource_type: str, **kwargs) -> None:
    """
    Register a resource for cleanup using global manager.
    
    Args:
        resource_id: Unique identifier for the resource
        resource_type: Type of resource
        **kwargs: Additional arguments for resource registration
    """
    manager = get_global_cleanup_manager()
    manager.register_resource(resource_id, resource_type, **kwargs)


def cleanup_resource(resource_id: str) -> bool:
    """
    Clean up a specific resource using global manager.
    
    Args:
        resource_id: Resource identifier
        
    Returns:
        True if cleanup was successful
    """
    manager = get_global_cleanup_manager()
    return manager.cleanup_resource(resource_id)