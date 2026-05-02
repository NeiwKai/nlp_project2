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
    results = DDGS().text(query, max_results=3)
    final = f"""
    Result 1:
    Title: {results[0]["title"]}
    Body: {results[0]["body"]}

    Result 2:
    Title: {results[1]["title"]}
    Body: {results[1]["body"]}

    Result 3:
    Title: {results[2]["title"]}
    Body: {results[2]["body"]}
    """
    return final
