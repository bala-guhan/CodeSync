from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import re
import ast
from dataclasses import dataclass

@dataclass
class CodeLocation:
    """Data class to hold code location information"""
    line_no: int
    line: str
    file_path: Path

def search_pattern(file_path: Union[str, Path], pattern: str) -> List[Tuple[int, str]]:
    """
    Search for a pattern in a file and return matching lines with their line numbers.
    
    Args:
        file_path (Union[str, Path]): Path to the file to search
        pattern (str): Regular expression pattern to search for
        
    Returns:
        List[Tuple[int, str]]: List of (line_number, line_content) tuples
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        matches = []
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                matches.append((i, line.rstrip()))
                
        return matches
    except Exception as e:
        raise IOError(f"Error searching file {file_path}: {str(e)}")

def parse_ast(file_path: Union[str, Path]) -> ast.Module:
    """
    Parse a Python file into an AST.
    
    Args:
        file_path (Union[str, Path]): Path to the Python file
        
    Returns:
        ast.Module: The parsed AST
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        SyntaxError: If the file contains invalid Python syntax
        IOError: If there's an error reading the file
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        raise SyntaxError(f"Invalid Python syntax in {file_path}: {str(e)}")
    except Exception as e:
        raise IOError(f"Error parsing file {file_path}: {str(e)}")

def find_functions(ast_tree: ast.Module, name_pattern: Optional[str] = None) -> List[ast.FunctionDef]:
    """
    Find function definitions in an AST.
    
    Args:
        ast_tree (ast.Module): The AST to search
        name_pattern (Optional[str]): Optional regex pattern to match function names
        
    Returns:
        List[ast.FunctionDef]: List of matching function definitions
    """
    functions = []
    pattern = re.compile(name_pattern) if name_pattern else None
    
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.FunctionDef):
            if pattern is None or pattern.search(node.name):
                functions.append(node)
                
    return functions

def find_imports(ast_tree: ast.Module) -> List[Tuple[str, List[str]]]:
    """
    Find all imports in an AST.
    
    Args:
        ast_tree (ast.Module): The AST to search
        
    Returns:
        List[Tuple[str, List[str]]]: List of (module_name, imported_names) tuples
    """
    imports = []
    
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.append((name.name, [name.asname or name.name]))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            names = [name.asname or name.name for name in node.names]
            imports.append((module, names))
            
    return imports

def extract_comments(ast_tree: ast.Module) -> Dict[int, str]:
    """
    Extract comments from an AST.
    
    Args:
        ast_tree (ast.Module): The AST to search
        
    Returns:
        Dict[int, str]: Dictionary mapping line numbers to comment text
    """
    comments = {}
    
    for node in ast.walk(ast_tree):
        if hasattr(node, 'lineno'):
            # Get the source line
            if hasattr(node, 'end_lineno'):
                line_no = node.end_lineno
            else:
                line_no = node.lineno
                
            # Get the comment if it exists
            if hasattr(node, 'end_col_offset'):
                comment = getattr(node, 'comment', None)
                if comment:
                    comments[line_no] = comment.strip()
                    
    return comments

def get_function_info(function_node: ast.FunctionDef) -> Dict[str, Any]:
    """
    Extract detailed information about a function from its AST node.
    
    Args:
        function_node (ast.FunctionDef): The function AST node
        
    Returns:
        Dict[str, Any]: Dictionary containing function information
            - name: Function name
            - args: List of argument names
            - defaults: List of default values
            - returns: Return type annotation
            - docstring: Function docstring
            - decorators: List of decorator names
    """
    return {
        'name': function_node.name,
        'args': [arg.arg for arg in function_node.args.args],
        'defaults': [ast.unparse(d) for d in function_node.args.defaults],
        'returns': ast.unparse(function_node.returns) if function_node.returns else None,
        'docstring': ast.get_docstring(function_node),
        'decorators': [ast.unparse(d) for d in function_node.decorator_list]
    } 