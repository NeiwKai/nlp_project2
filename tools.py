from ddgs import DDGS
from pydantic import BaseModel, Field
from langchain.tools import tool

# -------- TOOL MATH --------
class CalculationInput(BaseModel):
    a: int = Field(description="Argument a")
    b: int = Field(description="Argument b")

@tool("add_numbers", args_schema=CalculationInput)
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    print("add_numbers used!")
    return a + b

@tool("sub_numbers", args_schema=CalculationInput)
def sub_numbers(a: int, b: int) -> int:
    """Subtraction two numbers together."""
    print("sub_nunbers() used!")
    return a - b

# -------- TOOL UTILITIES --------
class BrowserInput(BaseModel):
    query: str = Field(description="The query that use to browse the internet")

@tool("browser", args_schema=BrowserInput)
def browser(query: str) -> str:
    """Browse the internet with given query."""
    print("browser() used!")

    results = DDGS().text(query, max_results=5)

    # format results into clean text for LLM
    formatted = []
    for r in results:
        formatted.append(
            f"Title: {r.get('title')}\nSnippet: {r.get('body', '')}\n"
        )

    return "\n".join(formatted)
