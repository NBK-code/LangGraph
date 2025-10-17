from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a+b

@tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a*b

tools = [add, multiply]
tools_by_name = {t.name: t for t in tools}

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=("You are my assistant. Use the tools to answer to the bext."))
    ai = llm.invoke([system_prompt]+list(state["messages"]))
    return {"messages": [ai]}

def custom_tool_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    tool_calls = last.tool_calls if isinstance(last, AIMessage) else []
    outs = []
    for call in tool_calls:
        name, args = call["name"], (call.get("args") or {})
        tool = tools_by_name.get(name)
        if tool is None:
            outs.append(ToolMessage(content=f"ERROR"))
            continue
        try:
            result = tool.invoke(args)
        except Exception as e:
            result = f"ERROR: {e}"
        outs.append(ToolMessage(content=str(result), tool_call_id = call["id"]))
    return {"messages": outs}

def need_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "end"

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", custom_tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", need_tools, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

print("\n--- Tool path ---")
inputs = {"messages": [("user", "Add 32 and 43 and multiply the result by 56.")]}
print_stream(app.stream(inputs, stream_mode="values"))
