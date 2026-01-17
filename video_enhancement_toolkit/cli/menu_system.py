"""
Menu System Implementation

Interactive menu system with input validation for the CLI interface.
"""

import re
from typing import List, Optional, Callable
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text

from .interfaces import IMenuSystem
from .models import MenuOption, UserInput


class MenuSystem(IMenuSystem):
    """Interactive menu system implementation."""
    
    def __init__(self):
        """Initialize menu system."""
        self.console = Console()
    
    def display_menu(self, options: List[MenuOption], title: str = "") -> str:
        """Display interactive menu with options.
        
        Args:
            options: List of menu options to display
            title: Optional menu title
            
        Returns:
            Selected option key
        """
        # Create menu table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Option", style="cyan", width=4)
        table.add_column("Title", style="white", width=25)
        table.add_column("Description", style="dim", width=50)
        
        # Add options to table
        for option in options:
            if option.enabled:
                table.add_row(option.key, option.title, option.description)
            else:
                table.add_row(
                    f"[dim]{option.key}[/dim]",
                    f"[dim]{option.title} (disabled)[/dim]",
                    f"[dim]{option.description}[/dim]"
                )
        
        # Display menu in panel
        if title:
            panel = Panel(table, title=title, border_style="blue")
        else:
            panel = Panel(table, border_style="blue")
        
        self.console.print(panel)
        
        # Get user selection
        valid_keys = [opt.key for opt in options if opt.enabled]
        while True:
            selection = Prompt.ask(
                "[cyan]Select an option[/cyan]",
                choices=valid_keys,
                show_choices=False
            )
            
            if selection in valid_keys:
                return selection
            else:
                self.display_error(f"Invalid selection '{selection}'. Please choose from: {', '.join(valid_keys)}")
    
    def get_user_input(self, prompt: str, validator: Optional[Callable[[str], bool]] = None) -> UserInput:
        """Get validated user input.
        
        Args:
            prompt: Input prompt to display
            validator: Optional validation function
            
        Returns:
            UserInput object with validated input
        """
        while True:
            try:
                user_input = Prompt.ask(f"[cyan]{prompt}[/cyan]")
                
                # Apply validation if provided
                if validator:
                    if validator(user_input):
                        return UserInput(value=user_input, is_valid=True)
                    else:
                        self.display_error("Invalid input. Please try again.")
                        continue
                else:
                    # Basic validation - non-empty string
                    if user_input.strip():
                        return UserInput(value=user_input.strip(), is_valid=True)
                    else:
                        self.display_error("Input cannot be empty. Please try again.")
                        continue
                        
            except KeyboardInterrupt:
                return UserInput(value="", is_valid=False, error_message="Input cancelled by user")
    
    def confirm_action(self, message: str) -> bool:
        """Get user confirmation for actions.
        
        Args:
            message: Confirmation message to display
            
        Returns:
            True if user confirms, False otherwise
        """
        try:
            return Confirm.ask(f"[yellow]{message}[/yellow]")
        except KeyboardInterrupt:
            return False
    
    def display_error(self, error_message: str, suggestions: Optional[List[str]] = None) -> None:
        """Display error message with optional suggestions.
        
        Args:
            error_message: Error message to display
            suggestions: Optional list of suggestions
        """
        error_text = Text()
        error_text.append("Error: ", style="bold red")
        error_text.append(error_message, style="red")
        
        if suggestions:
            error_text.append("\n\nSuggestions:", style="bold yellow")
            for suggestion in suggestions:
                error_text.append(f"\n• {suggestion}", style="yellow")
        
        panel = Panel(
            error_text,
            title="Error",
            border_style="red",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def display_success(self, success_message: str) -> None:
        """Display success message.
        
        Args:
            success_message: Success message to display
        """
        success_text = Text()
        success_text.append("Success: ", style="bold green")
        success_text.append(success_message, style="green")
        
        panel = Panel(
            success_text,
            title="Success",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def get_file_path(self, prompt: str, must_exist: bool = True) -> UserInput:
        """Get and validate file path input.
        
        Args:
            prompt: Input prompt to display
            must_exist: Whether file must exist
            
        Returns:
            UserInput object with validated file path
        """
        import os
        
        def validate_path(path: str) -> bool:
            if not path.strip():
                return False
            if must_exist and not os.path.exists(path):
                self.display_error(f"File not found: {path}")
                return False
            return True
        
        return self.get_user_input(prompt, validate_path)
    
    def get_numeric_input(self, prompt: str, min_value: Optional[float] = None, max_value: Optional[float] = None) -> UserInput:
        """Get and validate numeric input.
        
        Args:
            prompt: Input prompt to display
            min_value: Optional minimum value
            max_value: Optional maximum value
            
        Returns:
            UserInput object with validated numeric input
        """
        def validate_numeric(value: str) -> bool:
            try:
                num_value = float(value)
                if min_value is not None and num_value < min_value:
                    self.display_error(f"Value must be at least {min_value}")
                    return False
                if max_value is not None and num_value > max_value:
                    self.display_error(f"Value must be at most {max_value}")
                    return False
                return True
            except ValueError:
                self.display_error("Please enter a valid number")
                return False
        
        return self.get_user_input(prompt, validate_numeric)
    
    def get_choice_input(self, prompt: str, choices: List[str], case_sensitive: bool = False) -> UserInput:
        """Get input from a list of valid choices.
        
        Args:
            prompt: Input prompt to display
            choices: List of valid choices
            case_sensitive: Whether choices are case sensitive
            
        Returns:
            UserInput object with validated choice
        """
        if not case_sensitive:
            choices_lower = [choice.lower() for choice in choices]
        
        def validate_choice(value: str) -> bool:
            if case_sensitive:
                if value in choices:
                    return True
            else:
                if value.lower() in choices_lower:
                    return True
            
            self.display_error(f"Please choose from: {', '.join(choices)}")
            return False
        
        full_prompt = f"{prompt} ({'/'.join(choices)})"
        result = self.get_user_input(full_prompt, validate_choice)
        
        # Normalize case if not case sensitive
        if result.is_valid and not case_sensitive:
            # Find the original case version
            for choice in choices:
                if choice.lower() == result.value.lower():
                    result.value = choice
                    break
        
        return result
    
    def display_progress_summary(self, title: str, items: List[tuple]) -> None:
        """Display a summary table of progress or results.
        
        Args:
            title: Table title
            items: List of (name, status) tuples
        """
        table = Table(title=title, show_header=True)
        table.add_column("Item", style="cyan", width=30)
        table.add_column("Status", style="white", width=20)
        
        for name, status in items:
            # Color code status
            if "success" in status.lower() or "complete" in status.lower():
                status_style = "[green]" + status + "[/green]"
            elif "error" in status.lower() or "failed" in status.lower():
                status_style = "[red]" + status + "[/red]"
            elif "warning" in status.lower():
                status_style = "[yellow]" + status + "[/yellow]"
            else:
                status_style = status
            
            table.add_row(name, status_style)
        
        self.console.print(table)
    
    def display_workflow_step(self, step_number: int, step_title: str, step_description: str) -> None:
        """Display a workflow step with clear formatting.
        
        Args:
            step_number: Step number in the workflow
            step_title: Title of the step
            step_description: Description of what the step does
        """
        step_text = Text()
        step_text.append(f"Step {step_number}: ", style="bold cyan")
        step_text.append(step_title, style="bold white")
        step_text.append(f"\n{step_description}", style="dim")
        
        panel = Panel(
            step_text,
            border_style="blue",
            padding=(0, 1)
        )
        self.console.print(panel)
    
    def display_validation_error(self, field_name: str, error_message: str, suggestions: Optional[List[str]] = None) -> None:
        """Display validation error with field context.
        
        Args:
            field_name: Name of the field that failed validation
            error_message: Specific error message
            suggestions: Optional list of suggestions to fix the error
        """
        error_text = Text()
        error_text.append(f"Validation Error - {field_name}: ", style="bold red")
        error_text.append(error_message, style="red")
        
        if suggestions:
            error_text.append("\n\nSuggestions:", style="bold yellow")
            for suggestion in suggestions:
                error_text.append(f"\n• {suggestion}", style="yellow")
        
        panel = Panel(
            error_text,
            title="Input Validation Error",
            border_style="red",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def get_confirmation_with_details(self, action: str, details: dict, warnings: Optional[List[str]] = None) -> bool:
        """Get user confirmation with detailed information and optional warnings.
        
        Args:
            action: Action being confirmed
            details: Dictionary of details to display
            warnings: Optional list of warnings to show
            
        Returns:
            True if user confirms, False otherwise
        """
        # Display action details
        details_table = Table(title=f"Confirm {action}", show_header=True)
        details_table.add_column("Detail", style="cyan", width=20)
        details_table.add_column("Value", style="white", width=50)
        
        for key, value in details.items():
            details_table.add_row(key.replace("_", " ").title(), str(value))
        
        self.console.print(details_table)
        
        # Display warnings if any
        if warnings:
            self.console.print("\n[yellow]⚠️  Warnings:[/yellow]")
            for warning in warnings:
                self.console.print(f"  • [yellow]{warning}[/yellow]")
            self.console.print()
        
        # Get confirmation
        try:
            return Confirm.ask(f"[yellow]Proceed with {action}?[/yellow]")
        except KeyboardInterrupt:
            return False