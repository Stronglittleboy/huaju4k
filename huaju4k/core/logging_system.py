"""
Comprehensive logging system for huaju4k video enhancement.

This module provides structured logging with multiple levels, performance logging,
and log rotation management for the video enhancement toolkit.
"""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from enum import Enum
import traceback

# Performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class LogLevel(Enum):
    """Enhanced log levels for video processing."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    PERFORMANCE = "PERFORMANCE"
    PROGRESS = "PROGRESS"
    SYSTEM = "SYSTEM"


@dataclass
class PerformanceMetrics:
    """Performance metrics for logging."""
    
    timestamp: datetime
    operation: str
    duration_ms: float
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    items_processed: Optional[int] = None
    throughput: Optional[float] = None  # items per second
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SystemMetrics:
    """System resource metrics."""
    
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    temperature_celsius: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def __init__(self, include_system_info: bool = True):
        super().__init__()
        self.include_system_info = include_system_info
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured information."""
        # Base log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add thread information
        if hasattr(record, 'thread'):
            log_data['thread_id'] = record.thread
            log_data['thread_name'] = getattr(record, 'threadName', 'Unknown')
        
        # Add exception information
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add custom fields
        for key, value in record.__dict__.items():
            if key.startswith('custom_'):
                log_data[key[7:]] = value  # Remove 'custom_' prefix
        
        # Add system information if requested
        if self.include_system_info and PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                log_data['system'] = {
                    'cpu_percent': process.cpu_percent(),
                    'memory_mb': process.memory_info().rss / 1024 / 1024,
                    'pid': process.pid
                }
            except Exception:
                pass  # Ignore system info errors
        
        return json.dumps(log_data, ensure_ascii=False, separators=(',', ':'))


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._timers: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def timer(self, operation: str, **kwargs):
        """Context manager for timing operations."""
        start_time = time.perf_counter()
        start_metrics = self._get_system_metrics() if PSUTIL_AVAILABLE else None
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Calculate performance metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                operation=operation,
                duration_ms=duration_ms,
                **kwargs
            )
            
            # Add system metrics if available
            if start_metrics and PSUTIL_AVAILABLE:
                end_metrics = self._get_system_metrics()
                metrics.cpu_percent = end_metrics.cpu_percent
                metrics.memory_mb = end_metrics.memory_available_mb
            
            self.log_performance(metrics)
    
    def start_timer(self, operation: str) -> None:
        """Start a named timer."""
        with self._lock:
            self._timers[operation] = time.perf_counter()
    
    def end_timer(self, operation: str, **kwargs) -> float:
        """End a named timer and log performance."""
        with self._lock:
            if operation not in self._timers:
                self.logger.warning(f"Timer '{operation}' not found")
                return 0.0
            
            start_time = self._timers.pop(operation)
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                operation=operation,
                duration_ms=duration_ms,
                **kwargs
            )
            
            self.log_performance(metrics)
            return duration_ms
    
    def log_performance(self, metrics: PerformanceMetrics) -> None:
        """Log performance metrics."""
        self.logger.info(
            f"Performance: {metrics.operation} completed in {metrics.duration_ms:.2f}ms",
            extra={'custom_performance_metrics': metrics.to_dict()}
        )
    
    def _get_system_metrics(self) -> Optional[SystemMetrics]:
        """Get current system metrics."""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_usage_percent=disk.percent
            )
        except Exception as e:
            self.logger.debug(f"Failed to get system metrics: {e}")
            return None


class LoggingSystem:
    """
    Comprehensive logging system with structured logging, performance tracking,
    and log rotation management.
    """
    
    def __init__(self, 
                 log_dir: Union[str, Path] = "logs",
                 app_name: str = "huaju4k",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 console_level: LogLevel = LogLevel.INFO,
                 file_level: LogLevel = LogLevel.DEBUG):
        """
        Initialize logging system.
        
        Args:
            log_dir: Directory for log files
            app_name: Application name for log files
            max_file_size: Maximum size per log file in bytes
            backup_count: Number of backup files to keep
            console_level: Minimum level for console output
            file_level: Minimum level for file output
        """
        self.log_dir = Path(log_dir)
        self.app_name = app_name
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.console_level = console_level
        self.file_level = file_level
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger instances
        self.loggers: Dict[str, logging.Logger] = {}
        self.performance_loggers: Dict[str, PerformanceLogger] = {}
        
        # System monitoring
        self.system_monitor_active = False
        self.system_monitor_thread: Optional[threading.Thread] = None
        self.system_monitor_interval = 30  # seconds
        
        # Initialize root logger
        self._setup_root_logger()
        
        # Create main application logger
        self.main_logger = self.get_logger("huaju4k")
        self.main_logger.info("Logging system initialized")
    
    def _setup_root_logger(self) -> None:
        """Set up root logger configuration."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.console_level.value))
        
        # Simple formatter for console
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger with the specified name.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger instance
        """
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        # File handler with rotation
        log_file = self.log_dir / f"{self.app_name}_{name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, self.file_level.value))
        
        # Structured formatter for files
        file_formatter = StructuredFormatter(include_system_info=True)
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
        
        # Store logger
        self.loggers[name] = logger
        
        return logger
    
    def get_performance_logger(self, name: str) -> PerformanceLogger:
        """
        Get or create a performance logger.
        
        Args:
            name: Performance logger name
            
        Returns:
            PerformanceLogger instance
        """
        if name in self.performance_loggers:
            return self.performance_loggers[name]
        
        logger = self.get_logger(f"perf_{name}")
        perf_logger = PerformanceLogger(logger)
        
        self.performance_loggers[name] = perf_logger
        return perf_logger
    
    def create_error_log(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """
        Create detailed error log entry.
        
        Args:
            error: Exception that occurred
            context: Additional context information
        """
        logger = self.get_logger("errors")
        
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }
        
        logger.error(
            f"Error occurred: {error}",
            exc_info=True,
            extra={'custom_error_data': error_data}
        )
    
    def log_system_info(self) -> None:
        """Log current system information."""
        logger = self.get_logger("system")
        
        system_info = {
            'platform': sys.platform,
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'timestamp': datetime.now().isoformat()
        }
        
        if PSUTIL_AVAILABLE:
            try:
                system_info.update({
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                    'disk_total_gb': psutil.disk_usage('/').total / 1024**3
                })
            except Exception as e:
                logger.debug(f"Failed to get detailed system info: {e}")
        
        logger.info("System information", extra={'custom_system_info': system_info})
    
    def start_system_monitoring(self, interval: int = 30) -> None:
        """
        Start system resource monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.system_monitor_active:
            return
        
        self.system_monitor_interval = interval
        self.system_monitor_active = True
        
        self.system_monitor_thread = threading.Thread(
            target=self._system_monitor_loop,
            daemon=True
        )
        self.system_monitor_thread.start()
        
        self.main_logger.info(f"Started system monitoring (interval: {interval}s)")
    
    def stop_system_monitoring(self) -> None:
        """Stop system resource monitoring."""
        self.system_monitor_active = False
        
        if self.system_monitor_thread:
            self.system_monitor_thread.join(timeout=5)
        
        self.main_logger.info("Stopped system monitoring")
    
    def _system_monitor_loop(self) -> None:
        """System monitoring loop."""
        logger = self.get_logger("system_monitor")
        
        while self.system_monitor_active:
            try:
                if PSUTIL_AVAILABLE:
                    metrics = self._get_system_metrics()
                    if metrics:
                        logger.info(
                            f"System metrics: CPU {metrics.cpu_percent:.1f}%, "
                            f"Memory {metrics.memory_percent:.1f}%, "
                            f"Disk {metrics.disk_usage_percent:.1f}%",
                            extra={'custom_system_metrics': metrics.to_dict()}
                        )
                
                time.sleep(self.system_monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(self.system_monitor_interval)
    
    def _get_system_metrics(self) -> Optional[SystemMetrics]:
        """Get current system metrics."""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_usage_percent=disk.percent
            )
        except Exception:
            return None
    
    def create_session_log(self, session_id: str, operation: str, parameters: Dict[str, Any]) -> None:
        """
        Create session start log.
        
        Args:
            session_id: Unique session identifier
            operation: Operation being performed
            parameters: Operation parameters
        """
        logger = self.get_logger("sessions")
        
        session_data = {
            'session_id': session_id,
            'operation': operation,
            'parameters': parameters,
            'start_time': datetime.now().isoformat()
        }
        
        logger.info(
            f"Session started: {operation} (ID: {session_id})",
            extra={'custom_session_data': session_data}
        )
    
    def close_session_log(self, session_id: str, success: bool, 
                         duration_seconds: float, results: Dict[str, Any] = None) -> None:
        """
        Create session completion log.
        
        Args:
            session_id: Session identifier
            success: Whether session completed successfully
            duration_seconds: Session duration
            results: Session results
        """
        logger = self.get_logger("sessions")
        
        session_data = {
            'session_id': session_id,
            'success': success,
            'duration_seconds': duration_seconds,
            'end_time': datetime.now().isoformat(),
            'results': results or {}
        }
        
        status = "completed successfully" if success else "failed"
        logger.info(
            f"Session {status}: {session_id} (duration: {duration_seconds:.2f}s)",
            extra={'custom_session_data': session_data}
        )
    
    def cleanup_old_logs(self, max_age_days: int = 30) -> None:
        """
        Clean up old log files.
        
        Args:
            max_age_days: Maximum age of log files to keep
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cleaned_count = 0
        
        try:
            for log_file in self.log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    cleaned_count += 1
            
            self.main_logger.info(f"Cleaned up {cleaned_count} old log files")
            
        except Exception as e:
            self.main_logger.error(f"Error cleaning up logs: {e}")
    
    def get_log_summary(self) -> Dict[str, Any]:
        """
        Get summary of logging system status.
        
        Returns:
            Dictionary with logging system information
        """
        log_files = list(self.log_dir.glob("*.log"))
        total_size = sum(f.stat().st_size for f in log_files)
        
        return {
            'log_directory': str(self.log_dir),
            'active_loggers': len(self.loggers),
            'performance_loggers': len(self.performance_loggers),
            'log_files_count': len(log_files),
            'total_log_size_mb': total_size / 1024 / 1024,
            'system_monitoring_active': self.system_monitor_active,
            'psutil_available': PSUTIL_AVAILABLE
        }
    
    def shutdown(self) -> None:
        """Shutdown logging system and cleanup resources."""
        self.stop_system_monitoring()
        
        # Close all handlers
        for logger in self.loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        
        self.main_logger.info("Logging system shutdown complete")


# Global logging system instance
_global_logging_system: Optional[LoggingSystem] = None


def get_global_logging_system() -> LoggingSystem:
    """Get or create global logging system instance."""
    global _global_logging_system
    
    if _global_logging_system is None:
        _global_logging_system = LoggingSystem()
    
    return _global_logging_system


def get_logger(name: str) -> logging.Logger:
    """Get logger from global logging system."""
    return get_global_logging_system().get_logger(name)


def get_performance_logger(name: str) -> PerformanceLogger:
    """Get performance logger from global logging system."""
    return get_global_logging_system().get_performance_logger(name)


def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """Log error using global logging system."""
    get_global_logging_system().create_error_log(error, context)


def setup_logging(log_dir: str = "logs", 
                 console_level: LogLevel = LogLevel.INFO,
                 file_level: LogLevel = LogLevel.DEBUG,
                 enable_system_monitoring: bool = True) -> LoggingSystem:
    """
    Set up global logging system.
    
    Args:
        log_dir: Directory for log files
        console_level: Console logging level
        file_level: File logging level
        enable_system_monitoring: Whether to enable system monitoring
        
    Returns:
        Configured LoggingSystem instance
    """
    global _global_logging_system
    
    _global_logging_system = LoggingSystem(
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level
    )
    
    if enable_system_monitoring:
        _global_logging_system.start_system_monitoring()
    
    return _global_logging_system