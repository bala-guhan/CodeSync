from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class ToolSchema:
    """Data class to hold tool schema information"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]

class ToolRegistry:
    """Registry for managing tool functions and their schemas"""
    
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}  # name -> {function, schema}
        
    def _validate_schema(self, schema: dict) -> bool:
        """
        Validate the tool schema format.
        
        Args:
            schema (dict): The schema to validate
            
        Returns:
            bool: True if schema is valid
            
        Raises:
            ValueError: If schema is invalid
        """
        required_fields = ["type", "function"]
        function_fields = ["name", "description", "parameters"]
        param_fields = ["type", "properties", "required"]
        
        # Check top-level fields
        if not all(field in schema for field in required_fields):
            raise ValueError(f"Schema must contain fields: {required_fields}")
            
        # Check function fields
        if not all(field in schema["function"] for field in function_fields):
            raise ValueError(f"Function must contain fields: {function_fields}")
            
        # Check parameters structure
        params = schema["function"]["parameters"]
        if not all(field in params for field in param_fields):
            raise ValueError(f"Parameters must contain fields: {param_fields}")
            
        # Check that required parameters exist in properties
        required = params["required"]
        properties = params["properties"]
        if not all(param in properties for param in required):
            raise ValueError("All required parameters must have property definitions")
            
        return True
        
    def register_tool(self, function: Callable, schema: dict) -> None:
        """
        Register a new tool with its schema.
        
        Args:
            function (Callable): The tool function to register
            schema (dict): The tool's schema
            
        Raises:
            ValueError: If schema is invalid or tool already exists
        """
        # Validate schema
        self._validate_schema(schema)
        
        name = schema["function"]["name"]
        
        # Check if tool already exists
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
            
        # Register the tool
        self._tools[name] = {
            "function": function,
            "schema": schema
        }
        
    def unregister_tool(self, name: str) -> None:
        """
        Unregister a tool.
        
        Args:
            name (str): Name of the tool to unregister
            
        Raises:
            KeyError: If tool doesn't exist
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
            
        del self._tools[name]
        
    def get_tool_function(self, name: str) -> Callable:
        """
        Get a tool's function.
        
        Args:
            name (str): Name of the tool
            
        Returns:
            Callable: The tool's function
            
        Raises:
            KeyError: If tool doesn't exist
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
            
        return self._tools[name]["function"]
        
    def get_tool_schema(self, name: str) -> dict:
        """
        Get a tool's schema.
        
        Args:
            name (str): Name of the tool
            
        Returns:
            dict: The tool's schema
            
        Raises:
            KeyError: If tool doesn't exist
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
            
        return self._tools[name]["schema"]
        
    def get_all_tools(self) -> List[dict]:
        """
        Get schemas for all registered tools.
        
        Returns:
            List[dict]: List of all tool schemas
        """
        return [tool["schema"] for tool in self._tools.values()]
        
    def get_all_functions(self) -> Dict[str, Callable]:
        """
        Get all registered tool functions.
        
        Returns:
            Dict[str, Callable]: Dictionary of tool names to functions
        """
        return {name: tool["function"] for name, tool in self._tools.items()}
        
    def is_registered(self, name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            name (str): Name of the tool to check
            
        Returns:
            bool: True if tool is registered
        """
        return name in self._tools
        
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear() 