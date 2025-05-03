import os
import json
import requests
from dotenv import load_dotenv
import uuid
import logging
from jinja2 import Environment, FileSystemLoader
import datetime
from time import time

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
            {"type": "function", "function": {"name": "clone_repo", "description": "This tool takes in a url of a github repo and clones it in the mentioned filepath", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The URL of the GitHub repository to clone"}, "filepath": {"type": "string", "description": "The path where the repository should be cloned (optional, defaults to repository name in current directory)"}}, "required": ["url"]}}},
            {"type": "function", "function": {"name": "list_directory", "description": "List contents of a directory with file types and sizes", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The path to the directory to list"}}, "required": ["path"]}}}
            ]

    def list_directory(self, path: str) -> str:
        """List contents of a directory with file types and sizes"""
        try:
            if os.path.isdir(path):
                items = os.listdir(path)
                result = []
                for item in items:
                    full_path = os.path.join(path, item)
                    if os.path.isfile(full_path):
                        size = os.path.getsize(full_path)
                        result.append(f"📄 {item} ({size} bytes)")
                    else:
                        result.append(f"📁 {item}/")
                return "\n".join(result)
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    def clone_repo(self, url: str, filepath: str = "") -> str:
        """This tool takes in a url of a github repo and clones it in the mentioned filepath"""
        import git
        import os
        import requests
        
        try: # Check if the repository exists by sending a HEAD request to the URL
            response = requests.head(url)
            if response.status_code != 200:
                return f"Repository does not exist or is inaccessible: {url}"
            
            # If filepath is empty, use the repo name from the URL
            if not filepath:
                repo_name = url.split('/')[-1].replace('.git', '')
                filepath = os.path.join(os.getcwd(), repo_name)
            
            # Check if the filepath already exists
            if os.path.exists(filepath):
                return f"Directory already exists at {filepath}. Please choose a different filepath."
            
            # Clone the repository
            git.Repo.clone_from(url, filepath)
            print(f"Cloned repository from {url} to {filepath}")
            return f"Successfully cloned repository to {filepath}"
        except git.exc.GitCommandError as e:
            return f"Failed to clone repository: {str(e)}"
        except requests.RequestException as e:
            return f"Failed to verify repository: {str(e)}"
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def read_file(self, filepath: str) -> str:
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                data = file.read()
                msg = f"File successfully read>\n{filepath}\n {data}"
                return msg
        else:
            return f"File not found: {filepath}"
        
    def write_file(self, filepath: str, data: str) -> str:
        with open(filepath, 'w') as file:
            file.write(data)
            msg = f"Written to file {filepath} successfully!\n data written=>\n{data}"
            return msg
    
    def show_available_tools(self):
        """displays available tools"""
        for tool in self.tools:
            print(f"Tool : {tool['function']['name']}")
            print(f"Description ; {tool['function']['description']}")

    def call_llm(self, messages, tool_name):
        """Tool-call generating LLM (has tool schema attached)"""
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 0,
            "tools": tool_name
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
        if response.status_code == 200:
            response_ouput = response.json()
            self.token_usage += response_ouput['usage']['total_tokens']
            self.input_tokens += response_ouput['usage']['prompt_tokens']
            self.output_tokens += response_ouput['usage']['completion_tokens']
            return response_ouput
        else:
            raise Exception(f"API error: {response.status_code} => {response.text}")

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

    def tool_execution(self, user_input, tool_name : str):
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
        [1st line] A natural language instruction that clearly describes what to do — this will be interpreted by the model to generate a tool call. provide the code, if the parameter needed includes code
        [2nd line] ToolName
        [3rd line] Explanation: A brief justification of why this tool and action are appropriate in this context.

        example : 
        Read the contents of the example.py file. parameters: specify if important. Must specify if code is one of the parameters\n
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
        - Respond with a single-line natural instruction first, followed by a newline of the toolname and an explanation in the thirs line.
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

            # Step 3: Executing the tool call
            last_message = self.chat_history[-1]['content']
            split = last_message.strip().split('\n')
            instruction = split[0]
            tool_name = split[1]
            explanation = split[2]

            for tool in self.tools:
                if tool['function']['name'] == tool_name:
                    tool_name = tool

            # print(f"last message : {type(last_message)}")
            result = self.tool_execution(instruction, [tool_name])

            self.chat_history.append({ "role": "tool", "content": result })

            # if len(self.chat_history) > self.max_history:
            #     logging.info("Summarizing chat history to maintain context limits.")
            #     self.summarize_history()
        self.task_completion_time = time()-start_time
        log_response = self.save_chat_as_html(self.chat_history)
        print(log_response)
        

if __name__ == '__main__':
    agent = TrialAgent()
    task = """check the errors in the example.py file and correct it"""
    print(agent.invoke(task))
    # print("CodeSync on mission!")
    # print('='*100)

    # res = 'Read the contents of the example.py file, so we can identify and correct the errors in the code. parameters: filepath = example.py, importance: high, code: yes\nread_file\nIn order to know what is in the example.py file and check for errors, we need to read its contents first.'
    
    # def find_tool(toolname :str):
    #     for tool in agent.tools:
    #         if tool['function']['name'] == toolname:
    #             print(tool)

    # res2 = res.strip().split('\n', 1)

    # print(res2)

    
    # try:
    #     agent.invoke(task)
    #     print("="*80)
    #     print("\nChat History:")
    #     for msg in agent.chat_history:
    #         role = msg.get("role", "unknown")
    #         content = msg.get("content", "")
    #         print(f"{role.capitalize()}: {content}\n")

    # except Exception as e:
    #     print(f"Error: {e}")
