"""
File system utilities for huaju4k video enhancement.

This module provides file and directory management utilities,
including path validation, cleanup operations, and disk space monitoring.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
import tempfile
import logging

logger = logging.getLogger(__name__)


def ensure_directory(directory_path: str) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        Path object for the directory
        
    Raises:
        OSError: If directory cannot be created
    """
    path = Path(directory_path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        raise


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
        OSError: If file cannot be accessed
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        return path.stat().st_size
    except OSError as e:
        logger.error(f"Failed to get file size for {file_path}: {e}")
        raise


def get_available_disk_space(directory_path: str) -> int:
    """
    Get available disk space in bytes for given directory.
    
    Args:
        directory_path: Path to check disk space for
        
    Returns:
        Available disk space in bytes
        
    Raises:
        OSError: If disk space cannot be determined
    """
    try:
        if os.name == 'nt':  # Windows
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(directory_path),
                ctypes.pointer(free_bytes),
                None,
                None
            )
            return free_bytes.value
        else:  # Unix-like systems
            statvfs = os.statvfs(directory_path)
            return statvfs.f_frsize * statvfs.f_bavail
    except Exception as e:
        logger.error(f"Failed to get disk space for {directory_path}: {e}")
        raise OSError(f"Cannot determine disk space: {e}")


def cleanup_temp_files(temp_dir: str, pattern: str = "*", 
                      older_than_hours: int = 24) -> int:
    """
    Clean up temporary files older than specified time.
    
    Args:
        temp_dir: Directory containing temporary files
        pattern: File pattern to match (default: all files)
        older_than_hours: Remove files older than this many hours
        
    Returns:
        Number of files removed
        
    Raises:
        OSError: If cleanup operation fails
    """
    temp_path = Path(temp_dir)
    if not temp_path.exists():
        return 0
    
    import time
    current_time = time.time()
    cutoff_time = current_time - (older_than_hours * 3600)
    
    removed_count = 0
    try:
        for file_path in temp_path.glob(pattern):
            if file_path.is_file():
                file_mtime = file_path.stat().st_mtime
                if file_mtime < cutoff_time:
                    file_path.unlink()
                    removed_count += 1
                    logger.debug(f"Removed temp file: {file_path}")
    except Exception as e:
        logger.error(f"Error during temp file cleanup: {e}")
        raise OSError(f"Cleanup failed: {e}")
    
    logger.info(f"Cleaned up {removed_count} temporary files")
    return removed_count


def safe_file_move(source_path: str, destination_path: str, 
                  overwrite: bool = False) -> bool:
    """
    Safely move file from source to destination.
    
    Args:
        source_path: Source file path
        destination_path: Destination file path
        overwrite: Whether to overwrite existing destination file
        
    Returns:
        True if move was successful
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        FileExistsError: If destination exists and overwrite is False
        OSError: If move operation fails
    """
    source = Path(source_path)
    destination = Path(destination_path)
    
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination file exists: {destination_path}")
    
    try:
        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file
        shutil.move(str(source), str(destination))
        logger.info(f"Moved file from {source_path} to {destination_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to move file from {source_path} to {destination_path}: {e}")
        raise OSError(f"File move failed: {e}")


def create_temp_directory(prefix: str = "huaju4k_") -> str:
    """
    Create a temporary directory for processing.
    
    Args:
        prefix: Prefix for temporary directory name
        
    Returns:
        Path to created temporary directory
        
    Raises:
        OSError: If temporary directory cannot be created
    """
    try:
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        logger.debug(f"Created temporary directory: {temp_dir}")
        return temp_dir
    except Exception as e:
        logger.error(f"Failed to create temporary directory: {e}")
        raise OSError(f"Cannot create temp directory: {e}")


def get_unique_filename(base_path: str, extension: str = "") -> str:
    """
    Generate unique filename by appending number if file exists.
    
    Args:
        base_path: Base file path without extension
        extension: File extension (with or without dot)
        
    Returns:
        Unique file path
    """
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    
    counter = 0
    while True:
        if counter == 0:
            candidate = base_path + extension
        else:
            candidate = f"{base_path}_{counter}{extension}"
        
        if not Path(candidate).exists():
            return candidate
        
        counter += 1
        if counter > 9999:  # Prevent infinite loop
            raise OSError("Cannot generate unique filename")


def validate_file_permissions(file_path: str, 
                            read: bool = True, 
                            write: bool = False) -> bool:
    """
    Validate file permissions.
    
    Args:
        file_path: Path to file to check
        read: Whether read permission is required
        write: Whether write permission is required
        
    Returns:
        True if all required permissions are available
    """
    path = Path(file_path)
    
    if not path.exists():
        return False
    
    try:
        if read and not os.access(path, os.R_OK):
            return False
        
        if write and not os.access(path, os.W_OK):
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking permissions for {file_path}: {e}")
        return False