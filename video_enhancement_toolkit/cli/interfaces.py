"""
CLI Interfaces

Abstract base classes for command-line interface components.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import UserInput


class ICLIController(ABC):
    """Interface for main CLI controller."""
    
    @abstractmethod
    def run(self) -> None:
        """Main entry point for CLI application."""
        pass
    
    @abstractmethod
    def display_main_menu(self) -> str:
        """Display main menu and return user selection.
        
        Returns:
            User's menu selection
        """
        pass
    
    @abstractmethod
    def handle_video_processing(self) -> None:
        """Handle video processing workflow."""
        pass
    
    @abstractmethod
    def handle_audio_optimization(self) -> None:
        """Handle audio optimization workflow."""
        pass
    
    @abstractmethod
    def handle_configuration(self) -> None:
        """Handle configuration management workflow."""
        pass
    
    @abstractmethod
    def display_help(self) -> None:
        """Display comprehensive help documentation."""
        pass


class IMenuSystem(ABC):
    """Interface for interactive menu system."""
    
    @abstractmethod
    def display_menu(self, options: List['MenuOption'], title: str = "") -> str:
        """Display interactive menu with options.
        
        Args:
            options: List of menu options to display
            title: Optional menu title
            
        Returns:
            Selected option key
        """
        pass
    
    @abstractmethod
    def get_user_input(self, prompt: str, validator: Optional[Callable[[str], bool]] = None) -> 'UserInput':
        """Get validated user input.
        
        Args:
            prompt: Input prompt to display
            validator: Optional validation function
            
        Returns:
            UserInput object with validated input
        """
        pass
    
    @abstractmethod
    def confirm_action(self, message: str) -> bool:
        """Get user confirmation for actions.
        
        Args:
            message: Confirmation message to display
            
        Returns:
            True if user confirms, False otherwise
        """
        pass
    
    @abstractmethod
    def display_error(self, error_message: str, suggestions: Optional[List[str]] = None) -> None:
        """Display error message with optional suggestions.
        
        Args:
            error_message: Error message to display
            suggestions: Optional list of suggestions
        """
        pass
    
    @abstractmethod
    def display_success(self, success_message: str) -> None:
        """Display success message.
        
        Args:
            success_message: Success message to display
        """
        pass
    
    @abstractmethod
    def get_file_path(self, prompt: str, must_exist: bool = True) -> 'UserInput':
        """Get and validate file path input.
        
        Args:
            prompt: Input prompt to display
            must_exist: Whether file must exist
            
        Returns:
            UserInput object with validated file path
        """
        pass
    
    @abstractmethod
    def get_numeric_input(self, prompt: str, min_value: Optional[float] = None, max_value: Optional[float] = None) -> 'UserInput':
        """Get and validate numeric input.
        
        Args:
            prompt: Input prompt to display
            min_value: Optional minimum value
            max_value: Optional maximum value
            
        Returns:
            UserInput object with validated numeric input
        """
        pass
    
    @abstractmethod
    def get_choice_input(self, prompt: str, choices: List[str], case_sensitive: bool = False) -> 'UserInput':
        """Get input from a list of valid choices.
        
        Args:
            prompt: Input prompt to display
            choices: List of valid choices
            case_sensitive: Whether choices are case sensitive
            
        Returns:
            UserInput object with validated choice
        """
        pass