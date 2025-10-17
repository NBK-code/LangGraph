from typing import Annotated, Sequence, TypedDict, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

def append_notes(existing: List[str] | None, new: List[str] | None) -> List[str]:
    ex = list(existing) if existing else []
    nw = list(new) if new else []
    for n in nw:
        if n and n not in ex:
            ex.append(n)
    return ex

def sum_counts(existing: int | None, new: int | None) -> int:
    ex = existing if existing else 0
    nw = new if new else 0
    return ex + nw

class AgentState(TypedDict, total = False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    notes: Annotated[List[str], append_notes]
    question_counts: Annotated[int, sum_counts]

@tool
def show_notes():
    """Show all stored notes"""
    return {"event": "show_notes"}

@tool
def remember(text: str):
    """Store a short note for later retrieval"""
    return {"event": "remember", "text": text}

@tool
def how_many_questions():
    """Return how many questions the user has asked so far."""
    return {"event": "how_many_questions"}

TOOLS = [remember, show_notes, how_many_questions]

llm = ChatOpenAI(model = "gpt-4o-mini").bind_tools(tools= TOOLS)

def observe(state: AgentState) -> AgentState:
    """Increment the question counter once per user turn."""
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        return {"question_counts": 1}
    return {}

def agent(state: AgentState) -> AgentState:
    """Prompt the model to use tools for memory actions; otherwise answer briefly."""
    qcount = state["question_counts"] if "question_counts" in state else 0
    system_prompt = SystemMessage(content=
            "You are concise.\n"
            "- To save a note, call the tool `remember(text)` (e.g., when user says 'remember: ...').\n"
            "- To list notes, call `show_notes()`.\n"
            "- To report how many questions the user asked, call `how_many_questions()`.\n"
            f"(question_count so far: {qcount})")
    ai = llm.invoke([system_prompt]+list(state["messages"]))
    return {"messages": [ai]}

def memory_tools_node(state: AgentState) -> AgentState:
    """Custom ToolNode: execute tool calls AND mutate/read state."""
    last = state["messages"][-1]
    tool_calls = last.tool_calls if isinstance(last, AIMessage) else []
    results: List[ToolMessage] = []

    current_notes = list(state["notes"]) if "notes" in state and state["notes"] else []
    qcount = int(state["question_counts"]) if "question_counts" in state else 0

    new_notes: List[str] = []

    for call in tool_calls:
        name = call["name"]
        args = call["args"] or {}

        if name == "remember":
            text = str(args.get("text", "")).strip()
            if text:
                new_notes.append(text)
                results.append(ToolMessage(content=f"Added note: {text}", tool_call_id = call["id"]))
            else:
                results.append(ToolMessage(content="ERROR: empty note.", tool_call_id = call["id"]))
    
        elif name == "show_notes":
            content = "Your notes:\n- " + "\n- ".join(current_notes) if current_notes else "(no notes yet)"
            results.append(ToolMessage(content=content, tool_call_id = call["id"]))

        elif name == "how_many_questions":
            results.append(ToolMessage(content=str(qcount), tool_call_id=call["id"]))

        else:
            results.append(ToolMessage(content=f"ERROR: unknown tool '{name}'", tool_call_id=call["id"]))
        
    out: AgentState = {"messages": results}
    if new_notes:
        out["notes"] = new_notes
    return out

def needs_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"

graph = StateGraph(AgentState)
graph.add_node("observe", observe)
graph.add_node("agent", agent)
graph.add_node("memory_tools", memory_tools_node)

graph.set_entry_point("observe")
graph.add_edge("observe", "agent")
graph.add_conditional_edges("agent", needs_tools, {"tools": "memory_tools", "end": END})
graph.add_edge("memory_tools", "agent")

app = graph.compile(checkpointer=MemorySaver())

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

THREAD_ID = "demo-thread"

print("\n---- 1) Save a note via tool ----")
inputs = {"messages": [("user", "remember: buy milk at 6pm")]}
print_stream(app.stream(inputs, stream_mode="values",
                        config={"configurable": {"thread_id": THREAD_ID}}))

print("\n---- 2) Ask for notes (LLM should call show_notes) ----")
inputs = {"messages": [("user", "Please show notes.")]}
print_stream(app.stream(inputs, stream_mode="values",
                        config={"configurable": {"thread_id": THREAD_ID}}))

print("\n---- 3) Ask how many questions so far (tool reads state) ----")
inputs = {"messages": [("user", "How many questions have I asked?")]}
print_stream(app.stream(inputs, stream_mode="values",
                        config={"configurable": {"thread_id": THREAD_ID}}))

