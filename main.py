import json
import re
from langchain.tools import tool
from pydantic import BaseModel, Field

from langchain_community.chat_models import ChatLlamaCpp
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import messages_from_dict


from tools import add_numbers, sub_numbers, browser


# -------- LOCAL LLaMA --------
llm = ChatLlamaCpp(
    model_path="./gemma-3-4b-it-q4_k_m.gguf",
    temperature=0.2,
    n_ctx=4096,
    checkpointer=InMemorySaver(),
    verbose=False
)
llm_with_tools = llm.bind_tools([add_numbers, sub_numbers, browser])


# -------- SYSTEM PROMPT --------
prompt = """
You are a tool-using assistant.

You have ONLY two behaviors:

-----------------------
CRITICAL RULES:

- Tool output is the ONLY source of truth
- You MUST ignore your own knowledge completely
- You MUST base your final answer ONLY on Observation
- If Observation exists, you MUST use it

If you ignore the tool result, the answer is WRONG.

-----------------------
1. DIRECT ANSWER (no tools)

If the user is chatting (hello, hi, how are you):
→ respond normally
→ DO NOT use tools

-----------------------
2. TOOL USAGE (MANDATORY)

If the question involves:
- math (+, -)
- factual information (who, what, when, where, current, latest)

→ You MUST use a tool
→ You are NOT allowed to answer from your own knowledge

-----------------------
TOOLS:

Math:
- add_numbers(a, b)
- sub_numbers(a, b)

Search:
- browser(query)

-----------------------
FORMATS:

### Math:

Thought: briefly explain what are you doing

Action: add_numbers OR sub_numbers
Action Input: {"a": number, "b": number}

### Search:

Thought: briefly explain what are you doing

Action: browser
Action Input: {"query": "search query"}

-----------------------
After tool result:

You MUST respond:

Final Answer: <answer based ONLY on tool result>

-----------------------

User: """


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
