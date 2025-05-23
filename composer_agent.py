import os
import json
import requests
from dotenv import load_dotenv
import uuid
import logging
from jinja2 import Environment, FileSystemLoader
import datetime
from time import time
import pathlib
import ast
import subprocess
import tempfile
from typing import Dict

load_dotenv()

class TrialAgent:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.chat_history = [
            {"role": "system", "content": "You are a helpful Programming assistant named CodeSync!"}
        ]
        self.max_history = 10  # Max number of messages before summarization
        self.max_iterations = 7  # Prevent infinite loops

        self.input_tokens = 0
        self.token_usage = 0
        self.output_tokens = 0
        self.task_completion_time = 0
        self.tools = [
            {"type": "function", "function": {"name": "read_file", "description": "Read content from a file", "parameters": {"type": "object", "properties": {"filepath": {"type": "string", "description": "The path to the file to read"}}, "required": ["filepath"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Write content to a file", "parameters": {"type": "object", "properties": {"filepath": {"type": "string", "description": "The path of the file to write to"}, "data": {"type": "string", "description": "The content to write into the file"}}, "required": ["filepath", "data"]}}},
            {"type": "function", "function": {"name": "list_directory", "description": "List contents of a directory with file types and sizes", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The path to the directory to list"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "replace_function_in_file", "description": "Replaces old function code in a Python file with new code.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "The path to the Python file to modify"}, "old_code": {"type": "string", "description": "The original function code to replace"}, "new_code": {"type": "string", "description": "The new function code to insert"}}, "required": ["file_path", "old_code", "new_code"]}}},
            {"type": "function", "function": {"name": "extract_relevant_function", "description": "Extracts the most relevant function from source code using keyword overlap with a problem description.", "parameters": {"type": "object", "properties": {"file_content": {"type": "string", "description": "The full source code of a Python file"}, "problem_description": {"type": "string", "description": "Description of the bug or issue to match against functions"}}, "required": ["file_content", "problem_description"]}}},
            {"type": "function", "function": {"name": "run_inline_tests_against_module", "description": "Runs test cases (as code strings) against a specified Python file and returns pass/fail results.", "parameters": {"type": "object", "properties": {"filepath": {"type": "string", "description": "Path to the Python file to be tested"}, "fail_to_pass": {"type": "string", "description": "Test code string expected to fail before a fix and pass after"}, "pass_to_pass": {"type": "string", "description": "Test code string expected to always pass"}}, "required": ["filepath", "fail_to_pass", "pass_to_pass"]}}}
        ]
        self.toolkit = [tool['function']['name'] for tool in self.tools]
        self.tool_names = [tool['function']['name'] for tool in self.tools]

            
    def list_directory(self, path: str) -> str:
        """List contents of a directory with file types and sizes"""
        try:
            path_obj = pathlib.Path(path)
            if path_obj.is_dir():
                result = []
                for item in path_obj.iterdir():
                    if item.is_file():
                        size = item.stat().st_size
                        result.append(f"📄 {item.name} ({size} bytes)")
                    else:
                        result.append(f"📁 {item.name}/")
                return "\n".join(result)
            return f"Not a directory: {path}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    
    def read_file(self, filepath: str) -> str:
        """Read content from a file"""
        try:
            path_obj = pathlib.Path(filepath)
            if path_obj.exists():
                data = path_obj.read_text()
                msg = f"File successfully read>\n{filepath}\n {data}"
                return msg
            return f"File not found: {filepath}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
        
    def write_file(self, filepath: str, data: str) -> str:
        """Write content to a file"""
        try:
            path_obj = pathlib.Path(filepath)
            path_obj.write_text(data)
            msg = f"Written to file {filepath} successfully!\n data written=>\n{data}"
            return msg
        except Exception as e:
            return f"Error writing to file: {str(e)}"

    def extract_relevant_function(file_content: str, problem_description: str) -> str:
        """
        Extracts the most relevant function block from the given file content,
        based on word overlap with the problem description.

        Args:
            file_content (str): The full source code of the Python file.
            problem_description (str): The textual description of the bug or issue.

        Returns:
            str: The source code of the most relevant function, or None if none found.
        """
        try:
            tree = ast.parse(file_content)
        except SyntaxError as e:
            print(f"[AST PARSE ERROR] {e}")
            return None

        # Preprocess the description into searchable terms
        keywords = set(problem_description.lower().split())

        candidates = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                try:
                    func_code = ast.get_source_segment(file_content, node)
                    if not func_code:
                        continue

                    # Scoring by keyword match
                    func_code_lower = func_code.lower()
                    match_score = sum(1 for word in keywords if word in func_code_lower)

                    candidates.append((match_score, func_code, node.name))
                except Exception as err:
                    print(f"[WARN] Failed to process function: {err}")
                    continue

        # Return the highest scoring function's code
        if not candidates:
            return None

        candidates.sort(reverse=True, key=lambda x: x[0])
        best_score, best_func, best_name = candidates[0]

        print(f"[INFO] Top match: {best_name} with score {best_score}")

        return best_func

    def replace_function_in_file(file_path: str, old_code: str, new_code: str) -> bool:
        """
        Replaces old function code in the file with new LLM-generated code.
        Returns True if replacement is successful.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if old_code not in content:
            print("[!] Old function code not found in file.")
            return False

        updated_content = content.replace(old_code, new_code)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        return True
    
    def show_available_tools(self):
        """displays available tools"""
        for tool in self.tools:
            print(f"Tool : {tool['function']['name']} : {tool['function']['description']}")

    def call_llm(self, messages, tool_name):
        """Tool-call generating LLM (has tool schema attached)"""
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 0,
            "tools": tool_name
        }
        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
            if response.status_code == 200:
                response_ouput = response.json()
                self.token_usage += response_ouput['usage']['total_tokens']
                self.input_tokens += response_ouput['usage']['prompt_tokens']
                self.output_tokens += response_ouput['usage']['completion_tokens']
                return response_ouput
            else:
                raise Exception(f"API error: {response.status_code} => {response.text}")
        except Exception as e:
            import traceback
            error_msg = f"API error: {response.status_code} => {response.text}\nTraceback:\n{traceback.format_exc()}"
            raise Exception(error_msg)

    def call_llm_without_llm(self, messages):
        """Next step evaluating LLM (no tool schema)"""
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 0.7,
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
        if response.status_code == 200:
            response_ouput = response.json()
            self.token_usage += response_ouput['usage']['total_tokens']
            self.input_tokens += response_ouput['usage']['prompt_tokens']
            self.output_tokens += response_ouput['usage']['completion_tokens']
            return response_ouput
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")

    def summarize_history(self):
        print("\n Summarizing older history to save tokens...")
        historical = self.chat_history[1:-4]  # Skip system + latest 4 messages
        if not historical:
            return

        summary_prompt = [
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": f"Summarize the following conversation in 1 paragraph, focusing on completed task steps, the current state, and the next required action:\n\n" +
                                        "\n".join(f"{msg['role']}: {msg['content']}" for msg in historical)}
        ]

        response = self.call_llm(summary_prompt)
        summary = response['choices'][0]['message']['content']

        self.chat_history = [self.chat_history[0]] + \
                            [{"role": "assistant", "content": f"Summary of previous conversation: {summary}"}] + \
                            self.chat_history[-4:]

    def tool_execution(self, user_input, tool_name):
        """Tool_execution node which finds the tools and parses the arguments"""

        msg = [{"role":"user", "content":user_input}]
        response_data = self.call_llm(msg, tool_name)
        message = response_data['choices'][0]['message']

        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                tool_name = tool_call['function']['name']
                args_json = tool_call['function']['arguments']
                args = json.loads(args_json)
                # print(f"Calling tool: {tool_name} with args: {args}")

                tool_func = getattr(self, tool_name, None)
                if not tool_func:
                    raise Exception(f"Tool function '{tool_name}' not implemented.")

                result = tool_func(**args)
                # returns the message from the tool as it is.
                return result
            
        elif "content" in message:
            result = "a tool call was not generated! retry!"
            print(result)
            return result

        else:
            raise Exception(f"Unexpected message format: {json.dumps(message, indent=2)}")
    
    def evaluate(self, user_input):
        """Evaluates the progress of the task and adds proper instructions with justification."""

        prompt = f"""Based on the conversation history below, determine the next best step using one of the available tools.

        Your response must follow this exact format:
        [1st line] A natural language instruction that clearly describes what to do — this will be interpreted by the model to generate a tool call. provide the paramater which is needs to be parsed and nothing more than that.
        [2nd line] ToolName
        [3rd line] Explanation: A brief justification of why this tool and action are appropriate in this context.

        example : 
        Read the contents of the example.py file. parameters: example.py\n
        read_file\n
        In order to know what is in the example.py file.

        Available Tools:
        - read_file: Read the contents of a file  
        Required arguments → filepath: str

        - write_file: Write content to a file  
        Required arguments → filepath: str, data: str

        - clone_repo: Clone a GitHub repository  
        Required arguments → url: str, filepath: str (target directory)

        - list_dir: List all files in a directory  
        Required arguments → filepath: str

        Guidelines:
        - Use only the tools listed above. Do not provide instruction with tools you don'd have access to.
        - Respond with a single-line natural instruction first, followed by a newline of the toolname and an explanation in the third line.
        - Do NOT respond in JSON or with a function call — just use natural language to describe the action.
        - If no further actions are needed, respond with exactly: TASK_COMPLETE

        Chat History:
        {user_input}
        """

        evaluation_prompt = [{"role": "user", "content": prompt}]

        evaluation_response = self.call_llm_without_llm(evaluation_prompt)
        evaluation_text = evaluation_response['choices'][0]['message']['content'].strip()

        return evaluation_text

    def save_chat_as_html(self, chat_log):
        try:
            os.makedirs("logs", exist_ok=True)

            env = Environment(loader=FileSystemLoader("templates"))
            template = env.get_template("save_log_template.html")

            rendered_html = template.render(
                chat_log=chat_log,
                time_taken= f"{round(self.task_completion_time, 2)} seconds",
                total_tokens=f"{self.token_usage} tokens",
                input_tokens=f"{self.input_tokens} tokens",
                output_tokens=f"{self.output_tokens} tokens"
            )

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"logs/CodeSync-run-{timestamp}.html"

            with open(filename, "w", encoding="utf-8") as f:
                f.write(rendered_html)

            return "Logs added successfully!"
        except Exception as e:
            return f"Error occurred while logging: {e}"

    
    def invoke(self, query):
        """Main loop for running the agent autonomously until task completion"""
        start_time = time()
        self.chat_history.append({"role": "user", "content": query})
        
        for iteration in range(self.max_iterations):
            # Step 1: Evaluate the next step
            eval_response = self.evaluate(self.chat_history)
            self.chat_history.append({"role": "assistant", "content": eval_response})
            
            # Step 2: Check for completion
            if "TASK_COMPLETE" in eval_response:
                print("Done with the task yoohoo!")
                self.chat_history.append({"role": "assistant", "content": "TASK_COMPLETE"})
                break

            # Step 3: Find the tool in the message
            last_message = self.chat_history[-1]['content']
            
            # Find which tool is mentioned in the message using toolkit
            selected_tool = None
            for tool_name in self.toolkit:
                if tool_name in last_message:
                    # Get the corresponding tool object
                    selected_tool = next((tool for tool in self.tools if tool['function']['name'] == tool_name), None)
                    break

            if selected_tool:
                # Execute the tool call with only the specific tool
                result = self.tool_execution(last_message, [selected_tool])
                self.chat_history.append({ "role": "tool", "content": result })
            else:
                print("No tool found in message:", last_message)
                continue

        self.task_completion_time = time()-start_time
        log_response = self.save_chat_as_html(self.chat_history)
        print(log_response)

    def check_and_edit(self, filepath: str, problem_description: str, PASS_TO_PASS: str, FAIL_TO_PASS: str):
        """
        Analyzes a file, finds relevant code, gets fixes from LLM, applies changes and tests them.
        
        Args:
            filepath (str): Path to the Python file to analyze and fix
            problem_description (str): Description of the bug/issue to fix
            PASS_TO_PASS (str): Test case that should always pass
            FAIL_TO_PASS (str): Test case that should fail before fix and pass after
            
        Returns:
            dict: Results of the fix attempt including test results and any errors
        """
        try:
            # Step 1: Read the file content
            file_content = self.read_file(filepath)
            if "Error" in file_content:
                return {"error": f"Failed to read file: {file_content}"}
            
            # Step 2: Extract the relevant function using the problem description
            relevant_function = self.extract_relevant_function(file_content, problem_description)
            if not relevant_function:
                return {"error": "No relevant function found matching the problem description"}
            
            # Step 3: Get LLM's suggested fix
            fix_prompt = f"""Given this code and problem description, provide a fixed version of the function.
            Only return the complete fixed function code, nothing else.
            
            Problem Description:
            {problem_description}
            
            Original Code:
            {relevant_function}

            only provide the code without any other text. No explanations or any extra characters except the corrected code
            <output_format>
            def rectified_function():
                # code here
            </output_format>
            """
            
            fix_response = self.call_llm_without_llm([{"role": "user", "content": fix_prompt}])
            fixed_code = fix_response['choices'][0]['message']['content'].strip()
            
            # Step 4: Replace the old code with the fixed code
            if not self.replace_function_in_file(filepath, relevant_function, fixed_code):
                return {"error": "Failed to replace the function in the file"}
            
            # Step 5: Run tests to verify the fix
            test_results = self.run_inline_tests_against_module(
                filepath=filepath,
                fail_to_pass=FAIL_TO_PASS,
                pass_to_pass=PASS_TO_PASS
            )
            
            # Step 6: Return comprehensive results
            return {
                "status": "success",
                "original_function": relevant_function,
                "fixed_function": fixed_code,
                "test_results": test_results,
                "message": "Fix applied and tested successfully"
            }
            
        except Exception as e:
            import traceback
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }

if __name__ == '__main__':
    agent = TrialAgent()
    
