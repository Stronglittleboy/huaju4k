"""
Recovery Monitoring and Logging System

Comprehensive monitoring and logging of recovery actions and outcomes.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, Counter
import statistics

from .interfaces import ILogger
from .error_models import (
    ErrorReport, RecoveryAction, ErrorCategory, ErrorSeverity,
    CheckpointData
)


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    attempt_id: str
    error_id: str
    recovery_action: RecoveryAction
    timestamp: datetime
    success: bool
    duration_seconds: float
    error_category: ErrorCategory
    error_severity: ErrorSeverity
    operation: str
    stage: str
    attempt_number: int
    strategy_description: str
    failure_reason: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class RecoverySession:
    """Record of a complete recovery session for an error."""
    session_id: str
    error_id: str
    operation: str
    stage: str
    start_time: datetime
    end_time: Optional[datetime]
    total_attempts: int
    successful_attempts: int
    failed_attempts: int
    final_outcome: str  # "resolved", "aborted", "escalated"
    recovery_attempts: List[RecoveryAttempt]
    checkpoints_created: List[str]
    total_duration_seconds: Optional[float] = None


@dataclass
class RecoveryMetrics:
    """Aggregated recovery metrics."""
    total_errors: int
    total_recovery_attempts: int
    success_rate: float
    average_attempts_per_error: float
    most_common_error_category: ErrorCategory
    most_effective_recovery_action: RecoveryAction
    average_recovery_time_seconds: float
    checkpoint_usage_rate: float
    error_category_distribution: Dict[str, int]
    recovery_action_effectiveness: Dict[str, float]
    operation_error_rates: Dict[str, float]
    stage_error_rates: Dict[str, float]


class RecoveryMonitor:
    """Monitor and track recovery actions and outcomes."""
    
    def __init__(self, logger: ILogger, monitoring_dir: str = "recovery_monitoring"):
        self.logger = logger
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(exist_ok=True)
        
        # In-memory tracking
        self.recovery_attempts: List[RecoveryAttempt] = []
        self.recovery_sessions: Dict[str, RecoverySession] = {}
        self.active_sessions: Dict[str, RecoverySession] = {}
        
        # Metrics tracking
        self.error_counts = Counter()
        self.recovery_success_counts = Counter()
        self.recovery_failure_counts = Counter()
        self.operation_error_counts = Counter()
        self.stage_error_counts = Counter()
        
        # Load existing data
        self._load_monitoring_data()
    
    def start_recovery_session(self, error_report: ErrorReport) -> str:
        """Start tracking a new recovery session."""
        session_id = f"session_{error_report.error_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session = RecoverySession(
            session_id=session_id,
            error_id=error_report.error_id,
            operation=error_report.context.operation if error_report.context else "unknown",
            stage=error_report.context.component if error_report.context else "unknown",
            start_time=datetime.now(),
            end_time=None,
            total_attempts=0,
            successful_attempts=0,
            failed_attempts=0,
            final_outcome="in_progress",
            recovery_attempts=[],
            checkpoints_created=[]
        )
        
        self.active_sessions[error_report.error_id] = session
        self.recovery_sessions[session_id] = session
        
        # Update counters
        self.error_counts[error_report.category] += 1
        self.operation_error_counts[session.operation] += 1
        self.stage_error_counts[session.stage] += 1
        
        self.logger.log_operation(
            "recovery_session_started",
            {
                "session_id": session_id,
                "error_id": error_report.error_id,
                "error_category": error_report.category.value,
                "error_severity": error_report.severity.value,
                "operation": session.operation,
                "stage": session.stage
            }
        )
        
        return session_id
    
    def record_recovery_attempt(self, error_report: ErrorReport, recovery_action: RecoveryAction,
                              success: bool, duration_seconds: float, 
                              failure_reason: Optional[str] = None,
                              context: Optional[Dict[str, Any]] = None) -> str:
        """Record a recovery attempt."""
        session = self.active_sessions.get(error_report.error_id)
        if not session:
            # Create a new session if one doesn't exist
            session_id = self.start_recovery_session(error_report)
            session = self.active_sessions[error_report.error_id]
        
        attempt_id = f"attempt_{error_report.error_id}_{len(session.recovery_attempts) + 1}"
        
        # Find strategy description
        strategy_description = "Unknown strategy"
        for strategy in error_report.recovery_strategies:
            if strategy.action == recovery_action:
                strategy_description = strategy.description
                break
        
        attempt = RecoveryAttempt(
            attempt_id=attempt_id,
            error_id=error_report.error_id,
            recovery_action=recovery_action,
            timestamp=datetime.now(),
            success=success,
            duration_seconds=duration_seconds,
            error_category=error_report.category,
            error_severity=error_report.severity,
            operation=session.operation,
            stage=session.stage,
            attempt_number=len(session.recovery_attempts) + 1,
            strategy_description=strategy_description,
            failure_reason=failure_reason,
            context=context
        )
        
        # Update session
        session.recovery_attempts.append(attempt)
        session.total_attempts += 1
        
        if success:
            session.successful_attempts += 1
            self.recovery_success_counts[recovery_action] += 1
        else:
            session.failed_attempts += 1
            self.recovery_failure_counts[recovery_action] += 1
        
        # Add to global list
        self.recovery_attempts.append(attempt)
        
        self.logger.log_operation(
            "recovery_attempt_recorded",
            {
                "attempt_id": attempt_id,
                "session_id": session.session_id,
                "error_id": error_report.error_id,
                "recovery_action": recovery_action.value,
                "success": success,
                "duration_seconds": duration_seconds,
                "attempt_number": attempt.attempt_number,
                "failure_reason": failure_reason
            }
        )
        
        return attempt_id
    
    def record_checkpoint_creation(self, error_id: str, checkpoint: CheckpointData) -> None:
        """Record checkpoint creation during recovery."""
        session = self.active_sessions.get(error_id)
        if session:
            session.checkpoints_created.append(checkpoint.checkpoint_id)
            
            self.logger.log_operation(
                "recovery_checkpoint_created",
                {
                    "session_id": session.session_id,
                    "error_id": error_id,
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "operation": checkpoint.operation,
                    "stage": checkpoint.stage,
                    "progress": checkpoint.progress
                }
            )
    
    def end_recovery_session(self, error_id: str, final_outcome: str) -> Optional[str]:
        """End a recovery session."""
        session = self.active_sessions.get(error_id)
        if not session:
            return None
        
        session.end_time = datetime.now()
        session.final_outcome = final_outcome
        session.total_duration_seconds = (session.end_time - session.start_time).total_seconds()
        
        # Remove from active sessions
        del self.active_sessions[error_id]
        
        self.logger.log_operation(
            "recovery_session_ended",
            {
                "session_id": session.session_id,
                "error_id": error_id,
                "final_outcome": final_outcome,
                "total_attempts": session.total_attempts,
                "successful_attempts": session.successful_attempts,
                "failed_attempts": session.failed_attempts,
                "total_duration_seconds": session.total_duration_seconds,
                "checkpoints_created": len(session.checkpoints_created)
            }
        )
        
        # Save session data
        self._save_session_data(session)
        
        return session.session_id
    
    def get_recovery_metrics(self, time_window_hours: Optional[int] = None) -> RecoveryMetrics:
        """Get aggregated recovery metrics."""
        cutoff_time = None
        if time_window_hours:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Filter data by time window
        relevant_attempts = self.recovery_attempts
        relevant_sessions = list(self.recovery_sessions.values())
        
        if cutoff_time:
            relevant_attempts = [
                attempt for attempt in self.recovery_attempts
                if attempt.timestamp >= cutoff_time
            ]
            relevant_sessions = [
                session for session in self.recovery_sessions.values()
                if session.start_time >= cutoff_time
            ]
        
        if not relevant_attempts or not relevant_sessions:
            return RecoveryMetrics(
                total_errors=0,
                total_recovery_attempts=0,
                success_rate=0.0,
                average_attempts_per_error=0.0,
                most_common_error_category=ErrorCategory.SYSTEM,
                most_effective_recovery_action=RecoveryAction.RETRY,
                average_recovery_time_seconds=0.0,
                checkpoint_usage_rate=0.0,
                error_category_distribution={},
                recovery_action_effectiveness={},
                operation_error_rates={},
                stage_error_rates={}
            )
        
        # Calculate metrics
        total_errors = len(relevant_sessions)
        total_attempts = len(relevant_attempts)
        successful_attempts = sum(1 for attempt in relevant_attempts if attempt.success)
        success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
        
        # Average attempts per error
        average_attempts_per_error = total_attempts / total_errors if total_errors > 0 else 0.0
        
        # Most common error category
        category_counts = Counter(attempt.error_category for attempt in relevant_attempts)
        most_common_error_category = category_counts.most_common(1)[0][0] if category_counts else ErrorCategory.SYSTEM
        
        # Recovery action effectiveness
        action_success_rates = {}
        for action in RecoveryAction:
            action_attempts = [a for a in relevant_attempts if a.recovery_action == action]
            if action_attempts:
                action_successes = sum(1 for a in action_attempts if a.success)
                action_success_rates[action.value] = action_successes / len(action_attempts)
        
        most_effective_action = RecoveryAction.RETRY
        if action_success_rates:
            best_action_name = max(action_success_rates.keys(), key=lambda k: action_success_rates[k])
            most_effective_action = RecoveryAction(best_action_name)
        
        # Average recovery time
        durations = [attempt.duration_seconds for attempt in relevant_attempts]
        average_recovery_time = statistics.mean(durations) if durations else 0.0
        
        # Checkpoint usage rate
        sessions_with_checkpoints = sum(1 for session in relevant_sessions if session.checkpoints_created)
        checkpoint_usage_rate = sessions_with_checkpoints / total_errors if total_errors > 0 else 0.0
        
        # Error category distribution
        error_category_distribution = {
            category.value: count for category, count in category_counts.items()
        }
        
        # Operation and stage error rates
        operation_counts = Counter(session.operation for session in relevant_sessions)
        stage_counts = Counter(session.stage for session in relevant_sessions)
        
        operation_error_rates = {op: count / total_errors for op, count in operation_counts.items()}
        stage_error_rates = {stage: count / total_errors for stage, count in stage_counts.items()}
        
        return RecoveryMetrics(
            total_errors=total_errors,
            total_recovery_attempts=total_attempts,
            success_rate=success_rate,
            average_attempts_per_error=average_attempts_per_error,
            most_common_error_category=most_common_error_category,
            most_effective_recovery_action=most_effective_action,
            average_recovery_time_seconds=average_recovery_time,
            checkpoint_usage_rate=checkpoint_usage_rate,
            error_category_distribution=error_category_distribution,
            recovery_action_effectiveness=action_success_rates,
            operation_error_rates=operation_error_rates,
            stage_error_rates=stage_error_rates
        )
    
    def generate_recovery_report(self, time_window_hours: Optional[int] = None) -> Dict[str, Any]:
        """Generate a comprehensive recovery report."""
        metrics = self.get_recovery_metrics(time_window_hours)
        
        # Recent sessions
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours or 24)
        recent_sessions = [
            session for session in self.recovery_sessions.values()
            if session.start_time >= cutoff_time
        ]
        
        # Top failure patterns
        failure_patterns = defaultdict(int)
        for session in recent_sessions:
            if session.final_outcome != "resolved":
                pattern = f"{session.operation}:{session.stage}"
                failure_patterns[pattern] += 1
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "time_window_hours": time_window_hours or "all_time",
            "summary": {
                "total_errors": metrics.total_errors,
                "total_recovery_attempts": metrics.total_recovery_attempts,
                "overall_success_rate": f"{metrics.success_rate:.2%}",
                "average_attempts_per_error": f"{metrics.average_attempts_per_error:.1f}",
                "average_recovery_time": f"{metrics.average_recovery_time_seconds:.2f}s",
                "checkpoint_usage_rate": f"{metrics.checkpoint_usage_rate:.2%}"
            },
            "error_analysis": {
                "most_common_category": metrics.most_common_error_category.value,
                "category_distribution": metrics.error_category_distribution,
                "operation_error_rates": metrics.operation_error_rates,
                "stage_error_rates": metrics.stage_error_rates
            },
            "recovery_effectiveness": {
                "most_effective_action": metrics.most_effective_recovery_action.value,
                "action_success_rates": metrics.recovery_action_effectiveness
            },
            "failure_patterns": dict(failure_patterns.most_common(10)),
            "recommendations": self._generate_recommendations(metrics, recent_sessions)
        }
        
        return report
    
    def export_monitoring_data(self, output_path: str) -> None:
        """Export all monitoring data to a file."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "recovery_attempts": [asdict(attempt) for attempt in self.recovery_attempts],
            "recovery_sessions": {
                session_id: asdict(session) for session_id, session in self.recovery_sessions.items()
            },
            "metrics": asdict(self.get_recovery_metrics())
        }
        
        # Convert datetime objects to strings for JSON serialization
        def datetime_converter(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=datetime_converter)
        
        self.logger.log_operation(
            "monitoring_data_exported",
            {"output_path": output_path, "sessions_count": len(self.recovery_sessions)}
        )
    
    # Private helper methods
    
    def _load_monitoring_data(self) -> None:
        """Load existing monitoring data from disk."""
        try:
            sessions_file = self.monitoring_dir / "recovery_sessions.json"
            if sessions_file.exists():
                with open(sessions_file, 'r') as f:
                    data = json.load(f)
                
                # Load sessions
                for session_data in data.get("sessions", []):
                    session = RecoverySession(**session_data)
                    self.recovery_sessions[session.session_id] = session
                    
                    # Load attempts from session
                    for attempt_data in session.recovery_attempts:
                        if isinstance(attempt_data, dict):
                            attempt = RecoveryAttempt(**attempt_data)
                            self.recovery_attempts.append(attempt)
                
                self.logger.log_operation(
                    "monitoring_data_loaded",
                    {"sessions_loaded": len(self.recovery_sessions)}
                )
                
        except Exception as error:
            self.logger.log_error(error, {"context": "load_monitoring_data"})
    
    def _save_session_data(self, session: RecoverySession) -> None:
        """Save session data to disk."""
        try:
            sessions_file = self.monitoring_dir / "recovery_sessions.json"
            
            # Load existing data
            existing_data = {"sessions": []}
            if sessions_file.exists():
                with open(sessions_file, 'r') as f:
                    existing_data = json.load(f)
            
            # Add new session
            session_dict = asdict(session)
            existing_data["sessions"].append(session_dict)
            
            # Save updated data
            def datetime_converter(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            with open(sessions_file, 'w') as f:
                json.dump(existing_data, f, indent=2, default=datetime_converter)
                
        except Exception as error:
            self.logger.log_error(error, {"context": "save_session_data", "session_id": session.session_id})
    
    def _generate_recommendations(self, metrics: RecoveryMetrics, recent_sessions: List[RecoverySession]) -> List[str]:
        """Generate recommendations based on recovery metrics."""
        recommendations = []
        
        # Success rate recommendations
        if metrics.success_rate < 0.7:
            recommendations.append(
                f"Low recovery success rate ({metrics.success_rate:.1%}). "
                "Consider reviewing error handling strategies and improving fallback mechanisms."
            )
        
        # High attempt count recommendations
        if metrics.average_attempts_per_error > 3.0:
            recommendations.append(
                f"High average attempts per error ({metrics.average_attempts_per_error:.1f}). "
                "Consider improving initial error detection and recovery strategy selection."
            )
        
        # Checkpoint usage recommendations
        if metrics.checkpoint_usage_rate < 0.3:
            recommendations.append(
                f"Low checkpoint usage rate ({metrics.checkpoint_usage_rate:.1%}). "
                "Consider implementing more frequent checkpointing for long-running operations."
            )
        
        # Error pattern recommendations
        if metrics.most_common_error_category == ErrorCategory.RESOURCE:
            recommendations.append(
                "Resource errors are most common. Consider implementing better resource monitoring "
                "and preemptive resource management."
            )
        elif metrics.most_common_error_category == ErrorCategory.CONFIGURATION:
            recommendations.append(
                "Configuration errors are most common. Consider improving configuration validation "
                "and providing better default values."
            )
        
        # Recovery action effectiveness
        if metrics.most_effective_recovery_action == RecoveryAction.ABORT:
            recommendations.append(
                "Abort is the most effective recovery action, indicating that many errors are unrecoverable. "
                "Consider improving error prevention and early detection."
            )
        
        return recommendations