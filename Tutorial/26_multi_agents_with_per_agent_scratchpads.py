from typing import Annotated, Sequence, TypedDict, List
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver  # so scratchpads persist across calls (same thread_id)

load_dotenv()

def sum_counts(existing: int | None, new: int | None) -> int:
    ex = existing if existing else 0
    nw = new if new else 0
    return ex + nw

def append_notes(existing: List[str] | None, new: List[str] | None) ->  List[str]:
    ex = list(existing)  if existing else []
    nw = list(new) if new else []
    for n in nw:
        if n:
            ex.append(n)
    return ex

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    task: str
    math_turns: Annotated[int, sum_counts]
    math_notes: Annotated[List[str], append_notes]
    chat_turns: Annotated[int, sum_counts]
    chat_notes: Annotated[List[str], append_notes]

@tool
def add(a: int | None, b: int | None) -> int:
    """Adds two numbers"""
    return a + b

tools = [add]

manager_llm = ChatOpenAI(model="gpt-4o-mini")
math_llm = ChatOpenAI(model = "gpt-4o-mini").bind_tools(tools)
chat_llm = ChatOpenAI(model = "gpt-4o-mini")

def manager(state: AgentState) -> AgentState:
    
    system = SystemMessage(content=("Choose the best specialist.\n"
        "Reply with EXACTLY one line:\n"
        "TASK: math    (if user asks to add numbers)\n"
        "TASK: chat    (otherwise)"))
    ai = manager_llm.invoke([system] + list(state["messages"]))
    content = (ai.content or "").strip().lower()
    task = "math" if "task:" in content and "math" in content else "chat"
    return {"messages": [ai], "task": task}

def math_agent(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    is_human_message = isinstance(last, HumanMessage)
    turns  = state.get("math_turns", 0)
    notes = state.get("math_notes", [])
    mem = f"(math_turns = {turns})"

    if notes:
        mem += " | notes: " + " | ".join(notes[-3:])

    system = SystemMessage(content="Be concise and helpful. No tools.\n" + mem)
    ai = chat_llm.invoke([system] + list(state["messages"]))

    out: AgentState = {"messages": [ai]}
    if is_human_message:
        user_txt = last.content if isinstance(last, HumanMessage) else ""
        out["chat_turns"] = 1
        out["chat_notes"] = [f"asked: {user_txt[:80]}"]
    return out

def chat_agent(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    is_human_message = isinstance(last, HumanMessage)
    turns = state.get("chat_turns", 0)
    notes = state.get("chat_notes", [])
    mem = f"(chat_turns={turns})"
    if notes:
        mem += " | notes: " + " | ".join(notes[-3:])

    system = SystemMessage(content="Be concise and helpful. No tools.\n" + mem)
    ai = chat_llm.invoke([system] + list(state["messages"]))

    out: AgentState = {"messages": [ai]}
    if is_human_message:
        user_txt = last.content if isinstance(last, HumanMessage) else ""
        out["chat_turns"] = 1
        out["chat_notes"] = [f"asked: {user_txt[:80]}"]
    return out

def needs_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "end"


graph = StateGraph(AgentState)
graph.add_node("manager", manager)
graph.add_node("math_agent", math_agent)
graph.add_node("chat_agent", chat_agent)
graph.add_node("tools", ToolNode(tools=tools))
graph.set_entry_point("manager")
graph.add_conditional_edges("manager", lambda s: s.get("task", "chat"), {
    "math": "math_agent",
    "chat": "chat_agent",})
graph.add_conditional_edges("math_agent", needs_tools, {"tools": "tools", "end": END})
graph.add_edge("tools", "math_agent")
graph.add_edge("chat_agent", END)

app = graph.compile(checkpointer=MemorySaver())

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

THREAD_ID = "multi-agents-demo"

print("\n--- Turn 1: math route ---")
inputs = {"messages": [("user", "Please add 12 and 30.")]}
print_stream(app.stream(inputs, stream_mode="values",
                        config={"configurable": {"thread_id": THREAD_ID}}))

print("\n--- Turn 2: chat route ---")
inputs = {"messages": [("user", "Explain in one line what the math agent does.")] }
print_stream(app.stream(inputs, stream_mode="values",
                        config={"configurable": {"thread_id": THREAD_ID}}))

print("\n--- Turn 3: math route again (scratchpads should grow) ---")
inputs = {"messages": [("user", "Add 7 and 5.")]}
print_stream(app.stream(inputs, stream_mode="values",
                        config={"configurable": {"thread_id": THREAD_ID}}))


