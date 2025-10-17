from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

# -----------------------
# Helpers for pretty prints
# -----------------------
def brief(msg: BaseMessage) -> str:
    """Return a short one-line description of a message."""
    t = msg.__class__.__name__
    content = getattr(msg, "content", "")
    content = (content[:80] + "…") if len(content) > 80 else content
    return f"{t}: {content!r}"

def print_state(where: str, state):
    print(f"\n[DEBUG] {where}:")
    for i, m in enumerate(state["messages"]):
        print(f"  [{i}] {brief(m)}")

# -----------------------
# State
# -----------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# -----------------------
# Tools
# -----------------------
@tool
def add(a: int, b: int):
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int):
    """Multiply two numbers."""
    return a * b

tools = {t.name: t for t in [add, multiply]}

# -----------------------
# LLM (bound to tools)
# -----------------------
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(list(tools.values()))

# -----------------------
# Agent node (LLM call) with prints
# -----------------------
def agent_node(state: AgentState) -> AgentState:
    print_state("agent_node (input)", state)
    system = SystemMessage(content="You are a precise, concise assistant. Use tools for math.")
    ai = llm.invoke([system] + list(state["messages"]))
    out = {"messages": [ai]}
    print_state("agent_node (output)", out)
    return out

# -----------------------
# Custom tool node with prints
# -----------------------
def debug_tool_node(state: AgentState) -> AgentState:
    print_state("tool_node (input)", state)
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", []) or []

    results = []
    for call in tool_calls:
        name = call["name"]
        args = call["args"]
        print(f"[DEBUG] Invoking tool: {name} with args: {args}")

        try:
            tool_fn = tools[name]
        except KeyError:
            result = f"ERROR: Unknown tool '{name}'"
        else:
            try:
                result = tool_fn.invoke(args)
            except Exception as e:
                result = f"ERROR: {e}"

        # ToolMessage needs the tool_call_id back to the LLM
        tm = ToolMessage(
            content=str(result),
            tool_call_id=call["id"],
        )
        print(f"[DEBUG] Tool result for {name}: {result}")
        results.append(tm)

    out = {"messages": results}
    print_state("tool_node (output)", out)
    return out

# -----------------------
# Should continue?
# -----------------------
def needs_tools(state: AgentState):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"

# -----------------------
# Build graph
# -----------------------
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", debug_tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", needs_tools, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

app = graph.compile()

# -----------------------
# Run / Demo
# -----------------------
def print_stream(stream):
    print("\n=== STREAM START ===")
    for s in stream:
        msg = s["messages"][-1]
        try:
            msg.pretty_print()
        except Exception:
            print(type(msg), getattr(msg, "content", msg))
    print("=== STREAM END ===\n")

inputs = {
    "messages": [("user", "Add 34 and 41, then multiply the result by 3. Also say 'hi'.")]
}
print_stream(app.stream(inputs, stream_mode="values"))
