import os
import json
import requests
from dotenv import load_dotenv

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
            {"role": "system", "content": "You are a helpful Python assistant."}
        ]
        self.max_history = 2

        # Tool schema for function calling
        self.tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read content from a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "The path to the file to read"
                        }
                    },
                    "required": ["filepath"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "The path of the file to write to"
                        },
                        "data": {
                            "type": "string",
                            "description": "The content to write into the file"
                        }
                    },
                    "required": ["filepath", "data"]
                }
            }
        }]

    def read_file(self, filepath: str) -> str:
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                data = file.read()
                print(f"Read file: {filepath}")
                return data
        else:
            return f"File not found: {filepath}"
        
    def write_file(self, filepath: str, data: str) -> str:
        if os.path.exists(filepath):
            with open(filepath, 'w') as file:
                file.write(data)
                msg = "Written to file {filepath} successfully!"
                return msg
        else:
            return f"File not found: {filepath}"
        
    def call_llm(self, messages):
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 0.4,
            "tools": self.tools
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")

    def invoke(self, user_input):
        self.chat_history.append({"role": "user", "content": user_input})

        response_data = self.call_llm(self.chat_history)
        message = response_data['choices'][0]['message']

        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                tool_name = tool_call['function']['name']
                args_json = tool_call['function']['arguments']
                args = json.loads(args_json)
                print(f"Calling tool: {tool_name} with args: {args}")

                # 🔁 Dynamically call the function by name
                tool_func = getattr(self, tool_name, None)
                if not tool_func:
                    raise Exception(f"Tool function '{tool_name}' not implemented.")

                result = tool_func(**args)

                self.chat_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": result
                })
                return result

        elif "content" in message:
            reply = message["content"].strip()
            self.chat_history.append({"role": "assistant", "content": reply})
            return reply

        else:
            raise Exception(f"Unexpected message format: {json.dumps(message, indent=2)}")

if __name__ == '__main__':
    agent = TrialAgent()
    response = agent.invoke("Read the file test_output.txt and let me know what it is")
    print(response)
