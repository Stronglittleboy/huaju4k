"""
User-Friendly Error Reporter

Provides clear, actionable error messages and recovery options for users.
"""

from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt, Confirm

from .error_models import (
    ErrorReport, ErrorCategory, ErrorSeverity, RecoveryAction,
    VideoEnhancementError
)


class UserErrorReporter:
    """User-friendly error reporting and recovery option presentation."""
    
    def __init__(self):
        self.console = Console()
        
        # User-friendly error messages
        self.error_messages = {
            ErrorCategory.CONFIGURATION: {
                "title": "Configuration Issue",
                "icon": "⚙️",
                "description": "There's a problem with your settings or configuration file."
            },
            ErrorCategory.RESOURCE: {
                "title": "System Resource Issue", 
                "icon": "💾",
                "description": "Your system doesn't have enough resources (memory, disk space, etc.)."
            },
            ErrorCategory.PROCESSING: {
                "title": "Processing Error",
                "icon": "🎬",
                "description": "Something went wrong while processing your video or audio."
            },
            ErrorCategory.SYSTEM: {
                "title": "System Error",
                "icon": "🖥️",
                "description": "A system-level problem occurred (permissions, network, etc.)."
            },
            ErrorCategory.VALIDATION: {
                "title": "Input Validation Error",
                "icon": "✅",
                "description": "The input data or parameters are not valid."
            },
            ErrorCategory.NETWORK: {
                "title": "Network Error",
                "icon": "🌐",
                "description": "A network connection problem occurred."
            }
        }
        
        # Recovery action descriptions
        self.recovery_descriptions = {
            RecoveryAction.RETRY: {
                "title": "Try Again",
                "description": "Attempt the operation again",
                "icon": "🔄"
            },
            RecoveryAction.SKIP: {
                "title": "Skip This Step",
                "description": "Skip the problematic part and continue",
                "icon": "⏭️"
            },
            RecoveryAction.ABORT: {
                "title": "Stop Processing",
                "description": "Stop the current operation safely",
                "icon": "🛑"
            },
            RecoveryAction.FALLBACK: {
                "title": "Use Alternative Method",
                "description": "Try a different approach or use default settings",
                "icon": "🔀"
            },
            RecoveryAction.CONTINUE: {
                "title": "Continue Anyway",
                "description": "Ignore the error and continue processing",
                "icon": "➡️"
            }
        }
    
    def display_error(self, error_report: ErrorReport) -> None:
        """Display a user-friendly error message."""
        error_info = self.error_messages.get(error_report.category, {
            "title": "Unknown Error",
            "icon": "❌",
            "description": "An unexpected error occurred."
        })
        
        # Create error panel
        error_text = Text()
        error_text.append(f"{error_info['icon']} {error_info['title']}\n", style="bold red")
        error_text.append(f"{error_info['description']}\n\n", style="dim")
        error_text.append("Error Details:\n", style="bold")
        error_text.append(f"{error_report.message}\n", style="red")
        
        # Add context information if available
        if error_report.context:
            error_text.append(f"\nOperation: {error_report.context.operation}\n", style="dim")
            error_text.append(f"Component: {error_report.context.component}\n", style="dim")
        
        # Add severity indicator
        severity_colors = {
            ErrorSeverity.LOW: "green",
            ErrorSeverity.MEDIUM: "yellow", 
            ErrorSeverity.HIGH: "red",
            ErrorSeverity.CRITICAL: "bold red"
        }
        severity_color = severity_colors.get(error_report.severity, "white")
        error_text.append(f"\nSeverity: {error_report.severity.value.upper()}", style=severity_color)
        
        panel = Panel(
            error_text,
            title=f"Error ID: {error_report.error_id[:8]}",
            border_style="red",
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def display_recovery_options(self, error_report: ErrorReport) -> List[RecoveryAction]:
        """Display available recovery options and return them."""
        if not error_report.recovery_strategies:
            self.console.print("❌ No recovery options available.", style="red")
            return []
        
        self.console.print("\n🔧 Recovery Options:", style="bold blue")
        
        # Create table of recovery options
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Option", style="cyan", width=3)
        table.add_column("Action", style="green", width=20)
        table.add_column("Description", width=50)
        table.add_column("Max Attempts", justify="center", width=12)
        
        available_actions = []
        for i, strategy in enumerate(error_report.recovery_strategies, 1):
            action_info = self.recovery_descriptions.get(strategy.action, {
                "title": strategy.action.value.title(),
                "description": strategy.description,
                "icon": "❓"
            })
            
            table.add_row(
                str(i),
                f"{action_info['icon']} {action_info['title']}",
                strategy.description,
                str(strategy.max_attempts)
            )
            available_actions.append(strategy.action)
        
        self.console.print(table)
        return available_actions
    
    def get_user_recovery_choice(self, error_report: ErrorReport) -> Optional[RecoveryAction]:
        """Get user's choice for error recovery."""
        available_actions = self.display_recovery_options(error_report)
        
        if not available_actions:
            return None
        
        # Add option to view technical details
        self.console.print(f"\n[dim]Type 'details' to see technical information[/dim]")
        self.console.print(f"[dim]Type 'quit' to exit the application[/dim]")
        
        while True:
            try:
                choice = Prompt.ask(
                    "\nWhat would you like to do?",
                    choices=[str(i) for i in range(1, len(available_actions) + 1)] + ["details", "quit"],
                    default="1"
                )
                
                if choice == "details":
                    self._display_technical_details(error_report)
                    continue
                elif choice == "quit":
                    return None
                else:
                    choice_index = int(choice) - 1
                    return available_actions[choice_index]
                    
            except (ValueError, IndexError):
                self.console.print("❌ Invalid choice. Please try again.", style="red")
                continue
    
    def display_recovery_result(self, action: RecoveryAction, success: bool, message: str = "") -> None:
        """Display the result of a recovery action."""
        action_info = self.recovery_descriptions.get(action, {
            "title": action.value.title(),
            "icon": "❓"
        })
        
        if success:
            self.console.print(
                f"✅ {action_info['icon']} {action_info['title']} - Success!",
                style="bold green"
            )
            if message:
                self.console.print(f"   {message}", style="dim green")
        else:
            self.console.print(
                f"❌ {action_info['icon']} {action_info['title']} - Failed",
                style="bold red"
            )
            if message:
                self.console.print(f"   {message}", style="dim red")
    
    def confirm_recovery_action(self, action: RecoveryAction, strategy_description: str) -> bool:
        """Ask user to confirm a recovery action."""
        action_info = self.recovery_descriptions.get(action, {
            "title": action.value.title(),
            "icon": "❓"
        })
        
        return Confirm.ask(
            f"Confirm: {action_info['icon']} {action_info['title']} - {strategy_description}?"
        )
    
    def display_checkpoint_info(self, checkpoint_id: str, operation: str, stage: str, progress: float) -> None:
        """Display checkpoint creation information."""
        self.console.print(
            f"💾 Checkpoint created: {checkpoint_id[:8]}...",
            style="bold blue"
        )
        self.console.print(
            f"   Operation: {operation} | Stage: {stage} | Progress: {progress:.1%}",
            style="dim"
        )
    
    def display_resume_options(self, checkpoints: List[Dict[str, Any]]) -> Optional[str]:
        """Display available checkpoints for resuming and get user choice."""
        if not checkpoints:
            self.console.print("📁 No checkpoints available for resuming.", style="yellow")
            return None
        
        self.console.print("\n📁 Available Checkpoints:", style="bold blue")
        
        # Create table of checkpoints
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Option", style="cyan", width=3)
        table.add_column("Operation", style="green", width=15)
        table.add_column("Stage", style="yellow", width=15)
        table.add_column("Progress", justify="center", width=10)
        table.add_column("Created", style="dim", width=20)
        
        for i, checkpoint in enumerate(checkpoints, 1):
            table.add_row(
                str(i),
                checkpoint["operation"],
                checkpoint["stage"],
                f"{checkpoint['progress']:.1%}",
                checkpoint["timestamp"]
            )
        
        self.console.print(table)
        
        # Get user choice
        while True:
            try:
                choice = Prompt.ask(
                    "\nSelect checkpoint to resume from (or 'cancel')",
                    choices=[str(i) for i in range(1, len(checkpoints) + 1)] + ["cancel"],
                    default="cancel"
                )
                
                if choice == "cancel":
                    return None
                else:
                    choice_index = int(choice) - 1
                    return checkpoints[choice_index]["checkpoint_id"]
                    
            except (ValueError, IndexError):
                self.console.print("❌ Invalid choice. Please try again.", style="red")
                continue
    
    def _display_technical_details(self, error_report: ErrorReport) -> None:
        """Display technical error details."""
        details_text = Text()
        details_text.append("Technical Details:\n", style="bold")
        details_text.append(f"Error ID: {error_report.error_id}\n")
        details_text.append(f"Category: {error_report.category.value}\n")
        details_text.append(f"Severity: {error_report.severity.value}\n")
        details_text.append(f"Message: {error_report.message}\n\n")
        
        if error_report.context:
            details_text.append("Context:\n", style="bold")
            details_text.append(f"Operation: {error_report.context.operation}\n")
            details_text.append(f"Component: {error_report.context.component}\n")
            details_text.append(f"Timestamp: {error_report.context.timestamp}\n\n")
        
        details_text.append("Stack Trace:\n", style="bold")
        details_text.append(error_report.technical_details, style="dim")
        
        panel = Panel(
            details_text,
            title="Technical Details",
            border_style="yellow",
            padding=(1, 2)
        )
        
        self.console.print(panel)
        
        # Wait for user to continue
        Prompt.ask("\nPress Enter to continue", default="")