import json
import re
from langchain.tools import tool
from pydantic import BaseModel, Field

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import messages_from_dict


from tools import add_numbers, sub_numbers, browser


# -------- LOCAL LLaMA --------
llm = ChatLlamaCpp(
    model_path="./gemma-3-4b-it-q4_k_m.gguf",
    temperature=0.5,
    n_ctx=4096,
    verbose=False
)
llm_with_tools = llm.bind_tools([add_numbers, sub_numbers, browser])


# -------- SYSTEM PROMPT --------
prompt = """
You are a tool-using assistant.

You solve tasks using either:
1. Direct reasoning (no tools)
2. Tools (when required)

-----------------------
CRITICAL RULES:

- You MUST follow tool rules exactly.
- Tool output is the ONLY trusted source for factual or computed results.
- NEVER invent tool outputs.
- NEVER guess real-time or factual information.
- If a tool is required, you MUST use it before answering.

-----------------------
1. DIRECT ANSWER (NO TOOLS)

Use this ONLY for:
- greetings (hello, hi, how are you)
- casual conversation

→ Respond naturally
→ DO NOT use any tool

-----------------------
2. TOOL USAGE (MANDATORY)

Use tools for:
- math problems (+, -)
- factual questions (who, what, when, where)
- current / real-time information (latest, today, now, weather, news)

→ You MUST call a tool
→ You MUST NOT answer from memory or knowledge

If unsure → use browser tool

-----------------------
TOOLS:

Math:
- add_numbers(a, b)
- sub_numbers(a, b)

Search:
- browser(query)

-----------------------
REASONING FORMAT (when using tools):

Thought: decide if a tool is needed

Action: tool_name
Action Input: {"a": number, "b": number}
(or {"query": "text"} for browser)

Observation: result from tool

Repeat Thought/Action if needed.

-----------------------
FINAL ANSWER RULE:

After receiving tool result:

Final Answer: <only based on Observation>

- Do NOT include reasoning
- Do NOT mention tools
- Do NOT guess beyond Observation

-----------------------
IMPORTANT:

- If a tool is needed, you MUST NOT answer directly.
- If no tool is needed, NEVER call a tool.
- Keep responses short and accurate.

-----------------------

User:
"""


system = [SystemMessage(content=prompt)]

# -------- MEMORY --------
#MAX_HISTORY = 10
#history = []

"""
def save_history(history):
    with open("history.json", "w") as f:
        json.dump([msg.model_dump() for msg in history], f)
    print("Successfully save chat history!")

def load_history():
    try:
        with open("history.json") as f:
            data = json.load(f)
        print("Successfully load chat history!")
        return message_from_dict(data)
    except:
        return []
"""


def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return None


def run_agent(system, user_input, max_steps=5):
    scratchpad = f"System Prompt: {system}\nUser's question: {user_input}\n"

    for step in range(max_steps):
        print(f"\n--- Step {step+1} ---")

        response = llm_with_tools.invoke(scratchpad)
        text = response.content.strip()
        

        print("LLM:\n", text)

        # ✅ Check for final answer
        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()

        # ✅ Parse action
        if "Action:" in text and "Action Input:" in text:
            try:
                tool_name = re.search(r'Action:\s*(\w+)', text).group(1)
                json_str = re.search(r'\{.*\}', text, re.DOTALL).group()
                args = json.loads(json_str)

                tool_map = {
                    "add_numbers": add_numbers,
                    "sub_numbers": sub_numbers,
                    "browser": browser
                }

                result = tool_map[tool_name].invoke(args)

                print("Tool result:", result)

                # 🔥 Append observation
                scratchpad += text + f"\nObservation: {result}\n"

                #history.append(AIMessage(content=scratchpad))

            except Exception as e:
                print("⚠️ Failed tool execution:", e)
                return text

        else:
            # No tool used → just return
            #history.append(AIMessage(content=text))
            return text

    #history.append(AIMessage(content=text))
    return "Max steps reached."

# -------- RUN --------
from langchain_core.messages import ToolMessage

if __name__ == "__main__":
    #history = load_history()
    while True:
        user_input = input("YOU: ")

        if user_input == "/exit":
            #save_history(history)
            break
        #history = history[-MAX_HISTORY:]
        #history.append(HumanMessage(content=user_input))
        answer = run_agent(system, HumanMessage(content=user_input))
        final = llm.invoke(f"""
        This is the result of your observation:
        {answer}
        Try to answer to user question like you are actually human.
        Do not just place the observation straight, try to emphasize and construct a natural respond.
        Remember this is your observation, you need to answer to user like you really did it.
        User Ask you: {user_input}
        """
        )
        final = final.content
        #history.append(AIMessage(content=final))
        print(f"\nBOT: {final}")

    print("Thank for using me!")
