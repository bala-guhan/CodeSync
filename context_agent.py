import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

class CodeSync:
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
        self.summary_enabled = True
        self.max_history = 8  # Total messages (user+assistant)

    def call_llm(self, messages):
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 0.4
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")

    def summarize_history(self):
        print("\n📌 Summarizing older history to save tokens...")

        # Take all but the latest 2 rounds (user+assistant), and summarize
        historical = self.chat_history[1:-4]  # Skip system + latest 4
        if not historical:
            return

        summary_prompt = [
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": "Summarize the following conversation in 1 paragraph:\n\n" +
                                        "\n".join(f"{msg['role']}: {msg['content']}" for msg in historical)}
        ]

        summary = self.call_llm(summary_prompt)

        # Replace historical messages with a single summary message
        self.chat_history = [self.chat_history[0]] + \
                            [{"role": "assistant", "content": f"Summary of previous conversation: {summary}"}] + \
                            self.chat_history[-4:]

    def invoke(self, user_input):
        self.chat_history.append({"role": "user", "content": user_input})

        # Summarize if history is too long
        if self.summary_enabled and len(self.chat_history) > self.max_history:
            self.summarize_history()

        # Call LLM
        reply = self.call_llm(self.chat_history)
        self.chat_history.append({"role": "assistant", "content": reply})
        return reply

    def view_chat_history(self):
        print("\n🗂️ Chat History:")
        for i, msg in enumerate(self.chat_history):
            print(f"{i+1}. {msg['role'].upper()}: {msg['content']}")


# 🔁 Run interactive session
if __name__ == "__main__":
    agent = CodeSync()

    print("💬 Start chatting with the agent (type 'exit' to quit, 'view' to see chat history):")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "view":
            agent.view_chat_history()
            continue

        try:
            reply = agent.invoke(user_input)
            print(f"\n🤖 Agent: {reply}")
        except Exception as e:
            print(f"❌ Error: {e}")
