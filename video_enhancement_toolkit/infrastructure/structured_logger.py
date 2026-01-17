"""
Structured Logging System

Implementation of structured logging with JSON formatting, multiple handlers,
log rotation, and comprehensive error capture.
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from .interfaces import ILogger
from .models import LoggingConfig, LogLevel


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log string
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'operation'):
            log_data["operation"] = record.operation
        if hasattr(record, 'details'):
            log_data["details"] = record.details
        if hasattr(record, 'performance_metrics'):
            log_data["performance_metrics"] = record.performance_metrics
        if hasattr(record, 'error_context'):
            log_data["error_context"] = record.error_context
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class StructuredLogger(ILogger):
    """Structured logging system with JSON formatting and multiple handlers."""
    
    def __init__(self, config: LoggingConfig, logger_name: str = "video_enhancement_toolkit"):
        """Initialize structured logger.
        
        Args:
            config: Logging configuration
            logger_name: Name for the logger
        """
        self.config = config
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.operation_history: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.error_history: List[Dict[str, Any]] = []
        
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Set up logger with handlers and formatters."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        self.logger.setLevel(self.config.log_level.value)
        
        # Create formatters
        json_formatter = JSONFormatter()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            if self.config.json_format:
                console_handler.setFormatter(json_formatter)
            else:
                console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.config.log_file_path:
            log_path = Path(self.config.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(json_formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def log_operation(self, operation: str, details: Dict[str, Any], level: str = "INFO") -> None:
        """Log structured operation data.
        
        Args:
            operation: Name of the operation
            details: Dictionary with operation details
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        timestamp = datetime.now()
        
        # Store in operation history
        operation_record = {
            "timestamp": timestamp.isoformat(),
            "operation": operation,
            "details": details,
            "level": level
        }
        self.operation_history.append(operation_record)
        
        # Log the operation
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(
            log_level,
            f"Operation: {operation}",
            extra={
                "operation": operation,
                "details": details
            }
        )
    
    def log_performance(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics.
        
        Args:
            metrics: Dictionary with performance metrics
        """
        timestamp = datetime.now()
        
        # Store in performance history
        performance_record = {
            "timestamp": timestamp.isoformat(),
            "metrics": metrics
        }
        self.performance_history.append(performance_record)
        
        # Log the performance metrics
        self.logger.info(
            f"Performance metrics recorded",
            extra={
                "performance_metrics": metrics
            }
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log detailed error information.
        
        Args:
            error: Exception that occurred
            context: Dictionary with error context
        """
        timestamp = datetime.now()
        
        # Store in error history
        error_record = {
            "timestamp": timestamp.isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        self.error_history.append(error_record)
        
        # Log the error
        self.logger.error(
            f"Error occurred: {type(error).__name__}: {str(error)}",
            exc_info=error,
            extra={
                "error_context": context
            }
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive processing report.
        
        Returns:
            Dictionary with processing report data
        """
        report = {
            "report_generated": datetime.now().isoformat(),
            "logger_name": self.logger_name,
            "configuration": asdict(self.config),
            "statistics": {
                "total_operations": len(self.operation_history),
                "total_performance_records": len(self.performance_history),
                "total_errors": len(self.error_history)
            },
            "operations": self.operation_history[-10:],  # Last 10 operations
            "performance_metrics": self.performance_history[-10:],  # Last 10 metrics
            "errors": self.error_history[-5:]  # Last 5 errors
        }
        
        # Log report generation
        self.log_operation(
            "generate_report",
            {
                "operations_count": len(self.operation_history),
                "performance_records_count": len(self.performance_history),
                "errors_count": len(self.error_history)
            }
        )
        
        return report
    
    def clear_history(self) -> None:
        """Clear operation, performance, and error history."""
        self.operation_history.clear()
        self.performance_history.clear()
        self.error_history.clear()
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get operation history.
        
        Returns:
            List of operation records
        """
        return self.operation_history.copy()
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get performance history.
        
        Returns:
            List of performance records
        """
        return self.performance_history.copy()
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """Get error history.
        
        Returns:
            List of error records
        """
        return self.error_history.copy()
    
    def set_log_level(self, level: LogLevel) -> None:
        """Change log level dynamically.
        
        Args:
            level: New log level
        """
        self.config.log_level = level
        self.logger.setLevel(level.value)
        
        self.log_operation(
            "set_log_level",
            {"new_level": level.value}
        )


def create_logger(config: LoggingConfig, logger_name: str = "video_enhancement_toolkit") -> StructuredLogger:
    """Factory function to create a structured logger.
    
    Args:
        config: Logging configuration
        logger_name: Name for the logger
        
    Returns:
        Configured StructuredLogger instance
    """
    return StructuredLogger(config, logger_name)