from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    route: str # "math" | "chat"

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers"""
    return a + b

tools = [add]

llm_math = ChatOpenAI(model = "gpt-4o-mini").bind_tools(tools)
llm_chat = ChatOpenAI(model = "gpt-4o-mini")

def classify(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    text = last.content.lower() if isinstance(last, HumanMessage) else ""
    triggers = ["add", "sum", "+", "plus", "total"]
    route = "math" if any(t in text for t in triggers) else "chat"
    return {"route": route}

def math_agent(state: AgentState) -> AgentState:
    system = SystemMessage(content="Be concise. If the user wants addition , CALL the 'add' tool with integers.")
    ai = llm_math.invoke([system]+list(state["messages"]))
    return {"messages": [ai]}

def chat_agent(state: AgentState) -> AgentState:
    system = SystemMessage(content="Be concise and helpful. No tools.")
    ai = llm_chat.invoke([system] + list(state["messages"]))
    return {"messages": [ai]}

def needs_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "end"


graph = StateGraph(AgentState)
graph.add_node("classify", classify)
graph.add_node("math_agent", math_agent)
graph.add_node("chat_agent", chat_agent)
graph.add_node("tools", ToolNode(tools=tools))  # prebuilt executor
graph.set_entry_point("classify")
graph.add_conditional_edges("classify", lambda s: s["route"], {
    "math": "math_agent",
    "chat": "chat_agent",})
graph.add_conditional_edges("math_agent", needs_tools, {"tools": "tools", "end": END})
graph.add_edge("tools", "math_agent")
graph.add_edge("chat_agent", END)
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


print("\n--- Math route (uses tool) ---")
inputs = {"messages": [("user", "Please add 34 and 41.")]}
print_stream(app.stream(inputs, stream_mode="values"))

print("\n--- Chat route (no tool) ---")
inputs = {"messages": [("user", "What is a multi-agent system in one line?")]}
print_stream(app.stream(inputs, stream_mode="values"))