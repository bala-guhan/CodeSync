import subprocess
from typing import Union, List, Dict, Optional
from pathlib import Path
import shlex
import os

def run_command(
    command: Union[str, List[str]],
    cwd: Optional[Union[str, Path]] = None,
    shell: bool = False,
    timeout: Optional[int] = None,
    env: Optional[Dict[str, str]] = None
) -> Dict[str, Union[str, int]]:
    """
    Run a command in the shell and return its output.
    
    Args:
        command (Union[str, List[str]]): Command to run. Can be a string or list of strings.
        cwd (Optional[Union[str, Path]]): Working directory for the command
        shell (bool): Whether to run the command in a shell
        timeout (Optional[int]): Timeout in seconds
        env (Optional[Dict[str, str]]): Environment variables to set
        
    Returns:
        Dict[str, Union[str, int]]: Dictionary containing:
            - stdout: Command output
            - stderr: Error output
            - returncode: Return code
            - success: Whether command succeeded
            
    Raises:
        subprocess.TimeoutExpired: If command times out
        subprocess.CalledProcessError: If command fails
    """
    try:
        # Convert command to list if it's a string
        if isinstance(command, str):
            command = shlex.split(command) if not shell else command
            
        # Convert cwd to Path if it's a string
        if isinstance(cwd, str):
            cwd = Path(cwd)
            
        # Run the command
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=shell,
            timeout=timeout,
            env=env,
            capture_output=True,
            text=True
        )
        
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
            'success': result.returncode == 0
        }
        
    except subprocess.TimeoutExpired as e:
        return {
            'stdout': '',
            'stderr': f'Command timed out after {timeout} seconds',
            'returncode': -1,
            'success': False
        }
    except subprocess.CalledProcessError as e:
        return {
            'stdout': e.stdout if hasattr(e, 'stdout') else '',
            'stderr': e.stderr if hasattr(e, 'stderr') else str(e),
            'returncode': e.returncode,
            'success': False
        }
    except Exception as e:
        return {
            'stdout': '',
            'stderr': str(e),
            'returncode': -1,
            'success': False
        }

def run_python_script(
    script_path: Union[str, Path],
    args: Optional[List[str]] = None,
    cwd: Optional[Union[str, Path]] = None,
    timeout: Optional[int] = None,
    env: Optional[Dict[str, str]] = None
) -> Dict[str, Union[str, int]]:
    """
    Run a Python script and return its output.
    
    Args:
        script_path (Union[str, Path]): Path to the Python script
        args (Optional[List[str]]): Arguments to pass to the script
        cwd (Optional[Union[str, Path]]): Working directory
        timeout (Optional[int]): Timeout in seconds
        env (Optional[Dict[str, str]]): Environment variables to set
        
    Returns:
        Dict[str, Union[str, int]]: Dictionary containing command output and status
    """
    script_path = Path(script_path)
    if not script_path.exists():
        return {
            'stdout': '',
            'stderr': f'Script not found: {script_path}',
            'returncode': -1,
            'success': False
        }
        
    command = ['python', str(script_path)]
    if args:
        command.extend(args)
        
    return run_command(command, cwd=cwd, timeout=timeout, env=env)

def run_pip_command(
    command: str,
    cwd: Optional[Union[str, Path]] = None,
    timeout: Optional[int] = None,
    env: Optional[Dict[str, str]] = None
) -> Dict[str, Union[str, int]]:
    """
    Run a pip command and return its output.
    
    Args:
        command (str): Pip command (e.g., 'install requests')
        cwd (Optional[Union[str, Path]]): Working directory
        timeout (Optional[int]): Timeout in seconds
        env (Optional[Dict[str, str]]): Environment variables to set
        
    Returns:
        Dict[str, Union[str, int]]: Dictionary containing command output and status
    """
    pip_command = f'pip {command}'
    return run_command(pip_command, cwd=cwd, shell=True, timeout=timeout, env=env)

def run_git_command(
    command: str,
    cwd: Optional[Union[str, Path]] = None,
    timeout: Optional[int] = None,
    env: Optional[Dict[str, str]] = None
) -> Dict[str, Union[str, int]]:
    """
    Run a git command and return its output.
    
    Args:
        command (str): Git command (e.g., 'status')
        cwd (Optional[Union[str, Path]]): Working directory
        timeout (Optional[int]): Timeout in seconds
        env (Optional[Dict[str, str]]): Environment variables to set
        
    Returns:
        Dict[str, Union[str, int]]: Dictionary containing command output and status
    """
    git_command = f'git {command}'
    return run_command(git_command, cwd=cwd, shell=True, timeout=timeout, env=env)
