import os
from pathlib import Path
from tools import registry, find_function, replace_code, find_relevant_files
import json
from dotenv import load_dotenv
import requests
import traceback
from time import time
from jinja2 import Template
from datetime import datetime
import re
from tqdm import tqdm
load_dotenv()

class RepoAgent():
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
        self.value = 0
        self.tools = registry.get_all_tools()
        # Create a dictionary mapping tool names to their descriptions
        self.tool_desc = {
            tool['function']['name']: tool['function']['description']
            for tool in self.tools
        }
        self.token_usage = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.max_iterations = 10  # Maximum number of iterations for task completion
        self.task_completion_time = 0
        
        # Initialize logging
        self.current_log_file = None
        self.log_dir = Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.template_path = Path("templates/save_log_template.html")
        self.load_template()

    def load_template(self):
        """Load the Jinja template for logging"""
        try:
            
            with open(self.template_path, 'r', encoding='utf-8') as f:
                self.template = Template(f.read())
            
        except Exception as e:
            print(f"Error loading template: {e}")
            self.template = None

    def create_new_log_file(self):
        """Create a new log file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = self.log_dir / f"codesync_log_{timestamp}.html"
        self.update_log()  # Initialize the log file

    def update_log(self, error_msg=None):
        """
        Update the log file with current chat history and stats.
        
        Args:
            error_msg (str, optional): Error message to append if any
        """
        if not self.template or not self.current_log_file:
            print(f"[DEBUG] Skipping log update: template loaded? {self.template is not None}, log file set? {self.current_log_file is not None}")
            return

        try:
            # Prepare the context for the template
            context = {
                'time_taken': f"{self.task_completion_time:.2f} seconds",
                'total_tokens': self.token_usage,
                'input_tokens': self.input_tokens,
                'output_tokens': self.output_tokens,
                'chat_log': self.chat_history
            }

            # Add error message if provided
            if error_msg:
                self.chat_history.append({
                    "role": "error",
                    "content": error_msg
                })

            # Render the template
            rendered_html = self.template.render(**context)

            # Write to file
            with open(self.current_log_file, 'w', encoding='utf-8') as f:
                f.write(rendered_html)
            print(f"[DEBUG] Log updated at: {self.current_log_file.resolve()}")

        except Exception as e:
            print(f"Error updating log file: {e}")

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
            error_msg = f"API error: {response.status_code} => {response.text}\nTraceback:\n{traceback.format_exc()}"
            raise Exception(error_msg)
    
    def call_llm_without_tools(self, messages):
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

    def extract_code_from_llm_response(self, llm_response):
        """
        Extracts the code block from an LLM response (supports markdown code blocks).
        If no code block is found, returns the whole content.
        """
        # Get the assistant's message content
        if isinstance(llm_response, dict):
            content = llm_response['choices'][0]['message']['content']
        else:
            content = str(llm_response)
        # Try to extract code between triple backticks (with or without 'python')
        code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", content)
        if code_blocks:
            return code_blocks[0].strip()
        return content.strip()

    def evaluate(self, user_input):
        """
        Evaluates the progress of the task and determines the next best step using available tools.
        
        Args:
            user_input (str): The user's input or task description
            
        Returns:
            str: A formatted response containing:
                - Natural language instruction
                - Tool name
                - Explanation
        """
        prompt = f"""Based on the conversation history below, determine the next best step using one of the available tools.

        Your response must follow this exact format:
        [1st line] A natural language instruction that clearly describes what to do — this will be interpreted by the model to generate a tool call. Provide only the parameters needed, nothing more.
        [2nd line] ToolName
        [3rd line] Explanation: A brief justification of why this tool and action are appropriate in this context.

        Example:
        Read the contents of the example.py file. parameters: example.py
        read_file
        Explanation: To examine the contents of example.py and understand its code.

        Available Tools:
        {chr(10).join(self.tool_desc)}

        Guidelines:
        - Use only the tools listed above. Do not provide instructions with tools you don't have access to.
        - Respond with a single-line natural instruction first, followed by a newline of the toolname and an explanation in the third line.
        - Do NOT respond in JSON or with a function call — just use natural language to describe the action.
        - If no further actions are needed, respond with exactly: TASK_COMPLETE
        - Always provide the exact parameter names as shown in the tool descriptions.
        - For file paths, use relative paths when possible.
        - For commands, provide them exactly as they should be executed.
        - Use the minimal number of tools required. If a single tool can accomplish the task, do not use additional tools. 

        Chat History:
        {user_input}
        """

        evaluation_prompt = [{"role": "user", "content": prompt}]
        evaluation_response = self.call_llm_without_tools(evaluation_prompt)
        evaluation_text = evaluation_response['choices'][0]['message']['content'].strip()

        return evaluation_text

    def tool_execution(self, user_input, tool_name):
        """Tool_execution node which finds the tools and parses the arguments"""
        # Filter tools to only include the specified tool
        filtered_tools = [
            tool for tool in self.tools 
            if tool['function']['name'] == tool_name
        ]
        print(filtered_tools , '\n')
        if not filtered_tools:
            raise Exception(f"Tool '{tool_name}' not found in available tools")

        msg = [{"role":"user", "content":user_input}]
        response_data = self.call_llm(msg, filtered_tools)
        message = response_data['choices'][0]['message']

        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                tool_name = tool_call['function']['name']
                args_json = tool_call['function']['arguments']
                args = json.loads(args_json)
                # print(f"Calling tool: {tool_name} with args: {args}")

                try:
                    # Get the tool function from registry
                    tool_func = registry.get_tool_function(tool_name)
                    if not tool_func:
                        raise Exception(f"Tool function '{tool_name}' not found in registry.")

                    # Execute the tool with the parsed arguments
                    result = tool_func(**args)
                    return result
                except Exception as e:
                    raise Exception(f"Error executing tool '{tool_name}': {str(e)}")
            
        elif "content" in message:
            result = "a tool call was not generated! retry!"
            print(result)
            return result

        else:
            raise Exception(f"Unexpected message format: {json.dumps(message, indent=2)}")
        
    def invoke(self, query):
        """
        Main loop for running the agent autonomously until task completion.
        
        Args:
            query (str): The user's query or task description
            
        Returns:
            dict: Task execution results including:
                - completion_time: Time taken to complete the task
                - chat_history: Complete conversation history
                - final_result: Final result of the task
        """
        start_time = time()
        self.create_new_log_file()  # Create new log file for this task
        self.chat_history.append({"role": "user", "content": query})
        self.update_log()  # Log initial state
        
        for iteration in range(self.max_iterations):
            print(f"\nIteration {iteration + 1}/{self.max_iterations}")
            
            try:
                # Step 1: Evaluate the next step
                eval_response = self.evaluate(self.chat_history)
                self.chat_history.append({"role": "assistant", "content": eval_response})
                self.update_log()  # Log after evaluation
                print(f"Evaluation: {eval_response}")
                
                # Step 2: Check for completion
                if "TASK_COMPLETE" in eval_response:
                    print("Task completed successfully!")
                    self.chat_history.append({"role": "assistant", "content": "TASK_COMPLETE"})
                    self.update_log()  # Log completion
                    break

                # Step 3: Parse the evaluation response
                lines = eval_response.strip().split('\n')
                if len(lines) >= 2:
                    instruction = lines[0]
                    tool_name = lines[1].strip()
                    
                    # Extract parameters from the instruction
                    if "parameters:" in instruction:
                        params_str = instruction.split("parameters:")[1].strip()
                        # The tool_execution will handle parameter parsing
                    
                    try:
                        # Execute the tool
                        result = self.tool_execution(instruction, tool_name)
                        self.chat_history.append({"role": "tool", "content": str(result)})
                        self.update_log()  # Log after tool execution
                        print(f"Tool execution result: {result}")
                    except Exception as e:
                        error_msg = f"Error executing tool {tool_name}: {str(e)}"
                        self.chat_history.append({"role": "error", "content": error_msg})
                        self.update_log(error_msg)  # Log error
                        print(error_msg)
                else:
                    error_msg = "Invalid evaluation response format"
                    self.chat_history.append({"role": "error", "content": error_msg})
                    self.update_log(error_msg)  # Log error
                    print(error_msg)
                    
            except Exception as e:
                error_msg = f"Unexpected error in iteration {iteration + 1}: {str(e)}"
                self.chat_history.append({"role": "error", "content": error_msg})
                self.update_log(error_msg)  # Log error
                print(error_msg)

        self.task_completion_time = time() - start_time
        self.update_log()  # Final log update
        
        # Prepare the final response
        final_response = {
            "completion_time": self.task_completion_time,
            "chat_history": self.chat_history,
            "final_result": self.chat_history[-1]['content'] if self.chat_history else None,
            "iterations": iteration + 1,
            "token_usage": {
                "total": self.token_usage,
                "input": self.input_tokens,
                "output": self.output_tokens
            },
            "log_file": str(self.current_log_file) if self.current_log_file else None
        }
        
        return final_response
    
    def simple_code_edit(self, file_path, query):
        """
        For a given query and entry file, finds all relevant code segments in all involved files,
        gets the corrected code for each segment, and replaces the content in each file.
        """
        start_time = time()
        relevant = find_relevant_files(file_path, query)
        edited_files = []
        # Progress bar for files
        for path in tqdm(relevant, desc="Processing files"):
            code_segments = relevant[path]
            file_edited = False
            # Progress bar for segments in each file
            for segment in tqdm(code_segments, desc=f"Editing segments in {path}", leave=False):
                llm_response = self.call_llm_without_tools([
                    {"role": "user", "content": f"fix the following code: {segment}\n{query}\n- only provide the code, no other text or explanations"}
                ])
                new_code = self.extract_code_from_llm_response(llm_response)
                if new_code and new_code.strip() != segment.strip():
                    success = replace_code(path, old_code=segment, new_code=new_code)
                    if success:
                        file_edited = True
            if file_edited:
                edited_files.append(path)
        if edited_files:
            return f"Code edited successfully in: {', '.join(edited_files)}"
        else:
            return "No code was changed."

# Example usage
if __name__ == "__main__":
    agent = RepoAgent()
    # result = agent.invoke("hey there is a problem in the bubble sort code in main.py file, can you fix it")
    # print("\nTask completed!")
    # print(f"Time taken: {result['completion_time']:.2f} seconds")
    # print(f"Final result: {result['final_result']}")
    # print(f"Log file: {result['log_file']}")
    # # print(agent.tool_desc.values())

    result = agent.simple_code_edit("main.py", "Can you fix the process_user_score which is causing some errors")
    print(result)