"""
Dependency Injection Container

Manages dependencies and provides centralized module instantiation.
"""

from typing import Dict, Any, TypeVar, Type, Optional, Callable
from abc import ABC

T = TypeVar('T')


class DIContainer:
    """Dependency injection container for managing module dependencies."""
    
    def __init__(self):
        """Initialize the container."""
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
    
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a singleton service.
        
        Args:
            interface: Interface type
            implementation: Implementation type
        """
        self._services[interface] = implementation
        self._singletons[interface] = None
    
    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a transient service (new instance each time).
        
        Args:
            interface: Interface type
            implementation: Implementation type
        """
        self._services[interface] = implementation
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function for creating instances.
        
        Args:
            interface: Interface type
            factory: Factory function that creates instances
        """
        self._factories[interface] = factory
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register a specific instance.
        
        Args:
            interface: Interface type
            instance: Instance to register
        """
        self._singletons[interface] = instance
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service instance.
        
        Args:
            interface: Interface type to resolve
            
        Returns:
            Instance of the requested service
            
        Raises:
            ValueError: If service is not registered
        """
        # Check for registered instance first
        if interface in self._singletons and self._singletons[interface] is not None:
            return self._singletons[interface]
        
        # Check for factory
        if interface in self._factories:
            instance = self._factories[interface]()
            # If it was registered as singleton, cache it
            if interface in self._singletons:
                self._singletons[interface] = instance
            return instance
        
        # Check for registered service
        if interface in self._services:
            implementation = self._services[interface]
            instance = self._create_instance(implementation)
            
            # If it was registered as singleton, cache it
            if interface in self._singletons:
                self._singletons[interface] = instance
            
            return instance
        
        raise ValueError(f"Service {interface.__name__} is not registered")
    
    def _create_instance(self, implementation: Type[T]) -> T:
        """Create an instance with dependency injection.
        
        Args:
            implementation: Implementation type to instantiate
            
        Returns:
            Instance with injected dependencies
        """
        # Get constructor parameters
        import inspect
        signature = inspect.signature(implementation.__init__)
        parameters = signature.parameters
        
        # Skip 'self' parameter
        param_names = [name for name in parameters.keys() if name != 'self']
        
        # Resolve dependencies
        dependencies = {}
        for param_name in param_names:
            param = parameters[param_name]
            if param.annotation != inspect.Parameter.empty:
                dependencies[param_name] = self.resolve(param.annotation)
        
        return implementation(**dependencies)
    
    def is_registered(self, interface: Type[T]) -> bool:
        """Check if a service is registered.
        
        Args:
            interface: Interface type to check
            
        Returns:
            True if service is registered, False otherwise
        """
        return (interface in self._services or 
                interface in self._factories or 
                interface in self._singletons)
    
    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()


# Global container instance
container = DIContainer()