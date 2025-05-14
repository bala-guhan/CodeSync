from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import json
import os
import shutil
import glob
from datetime import datetime
import ast
import difflib

def read_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """
    Read the contents of a file.
    
    Args:
        file_path (Union[str, Path]): Path to the file to read
        encoding (str): File encoding (default: 'utf-8')
        
    Returns:
        str: Contents of the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        return file_path.read_text(encoding=encoding)
    except Exception as e:
        raise IOError(f"Error reading file {file_path}: {str(e)}")

def write_file(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
    """
    Write content to a file.
    
    Args:
        file_path (Union[str, Path]): Path to the file to write
        content (str): Content to write to the file
        encoding (str): File encoding (default: 'utf-8')
        
    Raises:
        IOError: If there's an error writing to the file
    """
    file_path = Path(file_path)
    try:
        file_path.write_text(content, encoding=encoding)
        return f"File written successfully with content\n : {content}"
    except Exception as e:
        raise IOError(f"Error writing to file {file_path}: {str(e)}")

def read_json(file_path: Union[str, Path], encoding: str = 'utf-8') -> dict:
    """
    Read and parse a JSON file.
    
    Args:
        file_path (Union[str, Path]): Path to the JSON file
        encoding (str): File encoding (default: 'utf-8')
        
    Returns:
        dict: Parsed JSON content
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        IOError: If there's an error reading the file
    """
    content = read_file(file_path, encoding)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {str(e)}", e.doc, e.pos)

def write_json(file_path: Union[str, Path], data: dict, encoding: str = 'utf-8', indent: int = 4) -> None:
    """
    Write data to a JSON file.
    
    Args:
        file_path (Union[str, Path]): Path to the JSON file
        data (dict): Data to write as JSON
        encoding (str): File encoding (default: 'utf-8')
        indent (int): Number of spaces for indentation (default: 4)
        
    Raises:
        IOError: If there's an error writing to the file
    """
    try:
        content = json.dumps(data, indent=indent)
        write_file(file_path, content, encoding)
    except Exception as e:
        raise IOError(f"Error writing JSON to file {file_path}: {str(e)}")

def list_directory(directory_path: Union[str, Path], pattern: str = "*") -> List[str]:
    """
    List contents of a directory with optional pattern matching.
    
    Args:
        directory_path (Union[str, Path]): Path to the directory
        pattern (str): Glob pattern to match files (default: "*")
        
    Returns:
        List[str]: List of file/directory names
        
    Raises:
        NotADirectoryError: If the path is not a directory
        IOError: If there's an error accessing the directory
    """
    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory_path}")
    
    try:
        return [str(p) for p in directory_path.glob(pattern)]
    except Exception as e:
        raise IOError(f"Error listing directory {directory_path}: {str(e)}")

def search_files(directory_path: Union[str, Path], pattern: str, recursive: bool = True) -> List[str]:
    """
    Search for files matching a pattern in a directory.
    
    Args:
        directory_path (Union[str, Path]): Path to search in
        pattern (str): Pattern to match (e.g., "*.py" for Python files)
        recursive (bool): Whether to search recursively (default: True)
        
    Returns:
        List[str]: List of matching file paths
        
    Raises:
        IOError: If there's an error during search
    """
    directory_path = Path(directory_path)
    try:
        if recursive:
            return [str(p) for p in directory_path.rglob(pattern)]
        return [str(p) for p in directory_path.glob(pattern)]
    except Exception as e:
        raise IOError(f"Error searching files in {directory_path}: {str(e)}")

def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get detailed information about a file.
    
    Args:
        file_path (Union[str, Path]): Path to the file
        
    Returns:
        Dict[str, Any]: Dictionary containing file information
            - size: File size in bytes
            - created: Creation timestamp
            - modified: Last modification timestamp
            - is_file: Whether it's a file
            - is_dir: Whether it's a directory
            - extension: File extension
            
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error getting file info
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        stats = file_path.stat()
        return {
            "size": stats.st_size,
            "created": datetime.fromtimestamp(stats.st_ctime),
            "modified": datetime.fromtimestamp(stats.st_mtime),
            "is_file": file_path.is_file(),
            "is_dir": file_path.is_dir(),
            "extension": file_path.suffix
        }
    except Exception as e:
        raise IOError(f"Error getting file info for {file_path}: {str(e)}")

def create_directory(directory_path: Union[str, Path], parents: bool = True) -> None:
    """
    Create a directory and its parents if they don't exist.
    
    Args:
        directory_path (Union[str, Path]): Path to create
        parents (bool): Whether to create parent directories (default: True)
        
    Raises:
        IOError: If there's an error creating the directory
    """
    directory_path = Path(directory_path)
    try:
        directory_path.mkdir(parents=parents, exist_ok=True)
    except Exception as e:
        raise IOError(f"Error creating directory {directory_path}: {str(e)}")

def copy_file(src: Union[str, Path], dst: Union[str, Path], overwrite: bool = False) -> None:
    """
    Copy a file from source to destination.
    
    Args:
        src (Union[str, Path]): Source file path
        dst (Union[str, Path]): Destination file path
        overwrite (bool): Whether to overwrite if destination exists (default: False)
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        FileExistsError: If destination exists and overwrite is False
        IOError: If there's an error copying the file
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination file exists: {dst}")
    
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        raise IOError(f"Error copying file from {src} to {dst}: {str(e)}")

def move_file(src: Union[str, Path], dst: Union[str, Path], overwrite: bool = False) -> None:
    """
    Move a file from source to destination.
    
    Args:
        src (Union[str, Path]): Source file path
        dst (Union[str, Path]): Destination file path
        overwrite (bool): Whether to overwrite if destination exists (default: False)
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        FileExistsError: If destination exists and overwrite is False
        IOError: If there's an error moving the file
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination file exists: {dst}")
    
    try:
        shutil.move(src, dst)
    except Exception as e:
        raise IOError(f"Error moving file from {src} to {dst}: {str(e)}")

def delete_file(file_path: Union[str, Path]) -> None:
    """
    Delete a file.
    
    Args:
        file_path (Union[str, Path]): Path to the file to delete
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error deleting the file
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        file_path.unlink()
    except Exception as e:
        raise IOError(f"Error deleting file {file_path}: {str(e)}")

def find_function(file_path: Union[str, Path], problem_statement: str) -> Optional[str]:
    """
    Locate the most relevant function in a Python file based on a problem statement.
    Returns the full source code of the function that best matches the problem statement using keyword overlap and fuzzy matching.

    Args:
        file_path (Union[str, Path]): Path to the Python file
        problem_statement (str): The problem statement or description

    Returns:
        Optional[str]: The full source code of the most relevant function, or None if not found
    """
    import ast
    import difflib
    from pathlib import Path

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except Exception as e:
        raise ValueError(f"Could not parse Python file: {e}")

    problem_statement_lc = problem_statement.lower()
    problem_keywords = set(
        word.lower() for word in problem_statement.replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
    )

    best_match_node = None
    best_score = -1

    # For extracting source code lines
    source_lines = source.splitlines(keepends=True)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node) or ""
            func_text = f"{node.name} {' '.join(arg.arg for arg in node.args.args)} {doc}"
            func_keywords = set(word.lower() for word in func_text.replace('(', ' ').replace(')', ' ').replace(',', ' ').split())
            # Keyword overlap score
            overlap_score = len(problem_keywords & func_keywords)
            # Fuzzy match score (difflib)
            fuzzy_score = difflib.SequenceMatcher(None, problem_statement_lc, func_text.lower()).ratio()
            # Combine scores (weight fuzzy more, but keep overlap)
            total_score = overlap_score + fuzzy_score
            if total_score > best_score:
                best_score = total_score
                best_match_node = node

    if best_match_node is None:
        return None

    # Extract the full function source code as a string
    # ast.get_source_segment is available in Python 3.8+, fallback to manual extraction if not available
    try:
        func_code = ast.get_source_segment(source, best_match_node)
        if func_code is not None:
            return func_code
    except Exception:
        pass

    # Fallback: extract lines using lineno and end_lineno (Python 3.8+)
    if hasattr(best_match_node, 'lineno') and hasattr(best_match_node, 'end_lineno'):
        start = best_match_node.lineno - 1
        end = best_match_node.end_lineno
        return ''.join(source_lines[start:end])

    # Fallback: extract from start lineno to next function/class or end of file
    if hasattr(best_match_node, 'lineno'):
        start = best_match_node.lineno - 1
        # Find the next function/class or end of file
        next_start = None
        for node2 in ast.walk(tree):
            if node2 is not best_match_node and hasattr(node2, 'lineno') and node2.lineno > best_match_node.lineno:
                if isinstance(node2, (ast.FunctionDef, ast.ClassDef)):
                    if next_start is None or node2.lineno < next_start:
                        next_start = node2.lineno - 1
        end = next_start if next_start is not None else len(source_lines)
        return ''.join(source_lines[start:end])

    return None

def replace_code(file_path: Union[str, Path], old_code: str, new_code: str) -> bool:
    """
    Replace a specific code segment in a file with new code.
    Args:
        file_path (Union[str, Path]): Path to the file to modify
        old_code (str): The code segment to be replaced
        new_code (str): The new code to insert
    Returns:
        bool: True if replacement was successful, False otherwise
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        content = file_path.read_text(encoding='utf-8')
        if old_code not in content:
            return False
        new_content = content.replace(old_code, new_code, 1)
        file_path.write_text(new_content, encoding='utf-8')
        return True
    except Exception as e:
        raise IOError(f"Error replacing code in {file_path}: {str(e)}")

def find_relevant_files(entry_file: Union[str, Path], problem_statement: str, codebase_dir: Union[str, Path] = '.') -> dict:
    """
    Find all relevant code segments for a bug: extracts all function calls and imports from the entire file,
    and includes any .py file that defines those functions or is imported, returning only the relevant code segments.
    Returns a dict mapping file paths to a list of relevant code segments as strings.
    """
    import ast
    from pathlib import Path

    entry_file = Path(entry_file)
    codebase_dir = Path(codebase_dir)
    if not entry_file.exists():
        raise FileNotFoundError(f"Entry file not found: {entry_file}")

    def read_code(path):
        try:
            return Path(path).read_text(encoding='utf-8')
        except Exception:
            return None

    entry_code = read_code(entry_file)
    tree = ast.parse(entry_code)

    # Extract all function calls and imported names from the whole file
    called_funcs = set()
    imported_names = set()
    imported_modules = set()
    for subnode in ast.walk(tree):
        if isinstance(subnode, ast.Call):
            if hasattr(subnode.func, 'id'):
                called_funcs.add(subnode.func.id)
            elif hasattr(subnode.func, 'attr'):
                called_funcs.add(subnode.func.attr)
        elif isinstance(subnode, ast.ImportFrom):
            if subnode.module:
                imported_modules.add(subnode.module)
            for alias in subnode.names:
                imported_names.add(alias.name)
        elif isinstance(subnode, ast.Import):
            for alias in subnode.names:
                imported_modules.add(alias.name)

    relevant_files = {str(entry_file): [entry_code]}
    for py_file in codebase_dir.rglob('*.py'):
        if py_file == entry_file:
            continue
        code = read_code(py_file)
        if not code:
            continue
        try:
            file_tree = ast.parse(code)
        except Exception:
            continue
        source_lines = code.splitlines(keepends=True)
        relevant_segments = []
        # If file is an imported module, include all top-level functions/classes
        if py_file.stem in imported_modules:
            for node in ast.iter_child_nodes(file_tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    try:
                        segment = ast.get_source_segment(code, node)
                        if segment:
                            relevant_segments.append(segment)
                    except Exception:
                        # Fallback to manual extraction
                        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                            relevant_segments.append(''.join(source_lines[node.lineno-1:node.end_lineno]))
            if relevant_segments:
                relevant_files[str(py_file)] = relevant_segments
            continue
        # Otherwise, include only the functions/classes that are called/imported
        defined_funcs = {node.name: node for node in ast.walk(file_tree) if isinstance(node, ast.FunctionDef)}
        defined_classes = {node.name: node for node in ast.walk(file_tree) if isinstance(node, ast.ClassDef)}
        for name, node in defined_funcs.items():
            if name in called_funcs or name in imported_names:
                try:
                    segment = ast.get_source_segment(code, node)
                    if segment:
                        relevant_segments.append(segment)
                except Exception:
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        relevant_segments.append(''.join(source_lines[node.lineno-1:node.end_lineno]))
        for name, node in defined_classes.items():
            if name in imported_names:
                try:
                    segment = ast.get_source_segment(code, node)
                    if segment:
                        relevant_segments.append(segment)
                except Exception:
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        relevant_segments.append(''.join(source_lines[node.lineno-1:node.end_lineno]))
        if relevant_segments:
            relevant_files[str(py_file)] = relevant_segments

    return relevant_files
