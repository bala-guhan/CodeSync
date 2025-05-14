from .file_tools import (
    read_file,
    write_file,
    find_function,
    replace_code,
    find_relevant_files
)
from .tool_registry import ToolRegistry

# Create the registry instance
registry = ToolRegistry()

# Register only read_file, write_file, and find_function tools
registry.register_tool(read_file, {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read content from a file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to read"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: 'utf-8')"
                }
            },
            "required": ["file_path"]
        }
    }
})

registry.register_tool(write_file, {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path of the file to write to"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write into the file"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: 'utf-8')"
                }
            },
            "required": ["file_path", "content"]
        }
    }
})

registry.register_tool(find_function, {
    "type": "function",
    "function": {
        "name": "find_function",
        "description": "Locate the most relevant function in a Python file based on a problem statement and return its code.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the Python file"
                },
                "problem_statement": {
                    "type": "string",
                    "description": "The problem statement or description"
                }
            },
            "required": ["file_path", "problem_statement"]
        }
    }
})

registry.register_tool(replace_code, {
    "type": "function",
    "function": {
        "name": "replace_code",
        "description": "Replace a specific code segment in a file with new code.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to modify"
                },
                "old_code": {
                    "type": "string",
                    "description": "The code segment to be replaced"
                },
                "new_code": {
                    "type": "string",
                    "description": "The new code to insert"
                }
            },
            "required": ["file_path", "old_code", "new_code"]
        }
    }
})

registry.register_tool(find_relevant_files, {
    "type": "function",
    "function": {
        "name": "find_relevant_files",
        "description": "Given an entry file and a problem statement, find all relevant files (and their code) that may need to be edited to fix the bug.",
        "parameters": {
            "type": "object",
            "properties": {
                "entry_file": {
                    "type": "string",
                    "description": "The main file to start from"
                },
                "problem_statement": {
                    "type": "string",
                    "description": "The bug/problem description"
                },
                "codebase_dir": {
                    "type": "string",
                    "description": "Directory to search for code files (default: current dir)"
                }
            },
            "required": ["entry_file", "problem_statement"]
        }
    }
})

# Export the registry and all functions
__all__ = [
    'read_file',
    'write_file',
    'find_function',
    'replace_code',
    'find_relevant_files',
    'registry'
]
