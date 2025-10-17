from typing import Annotated, Sequence, TypedDict
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    route: str  # "util" | "time" | "chat"

@tool
def reverse(text: str):
    """Reverse a string."""
    return text[::-1]

@tool
def word_count(text: str):
    """Count words in a string."""
    return len(text.split())

util_tools = [reverse, word_count]

@tool
def get_time():
    """Return current local time as YYYY-MM-DD HH:MM:SS."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

time_tools = [get_time]

llm_util = ChatOpenAI(model="gpt-4o-mini").bind_tools(util_tools)
llm_time = ChatOpenAI(model="gpt-4o-mini").bind_tools(time_tools)
llm_chat = ChatOpenAI(model="gpt-4o-mini")  # fallback, no tools

def classify(state: AgentState) -> AgentState:
    # Rule-based router to keep things simple.
    text = state["messages"][-1].content.lower()

    route = "chat" #default

    # Check utility keywords
    util_keywords = ["reverse", "word count", "count words"]
    found = False
    for k in util_keywords:
        if k in text:
            route = "util"
            found = True
            break

    # If not utility, check time keywords
    if not found:
        time_keywords = ["time", "date", "clock"]
        for k in time_keywords:
            if k in text:
                route = "time"
                break

    return {"route": route}

def util_agent(state: AgentState) -> AgentState:
    system = SystemMessage(content="You are a string utility assistant. Use tools for string ops; be concise.")
    ai = llm_util.invoke([system] + list(state["messages"]))
    return {"messages": [ai]}

def time_agent(state: AgentState) -> AgentState:
    system = SystemMessage(content="You provide current time using tools; be concise.")
    ai = llm_time.invoke([system] + list(state["messages"]))
    return {"messages": [ai]}

def chat_agent(state: AgentState) -> AgentState:
    system = SystemMessage(content="You are a brief, helpful general assistant.")
    ai = llm_chat.invoke([system] + list(state["messages"]))
    return {"messages": [ai]}

def needs_tools(state: AgentState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"

graph = StateGraph(AgentState)

graph.add_node("classify", classify)
graph.add_node("util_agent", util_agent)
graph.add_node("time_agent", time_agent)
graph.add_node("chat_agent", chat_agent)

graph.add_node("util_tools", ToolNode(tools=util_tools))
graph.add_node("time_tools", ToolNode(tools=time_tools))

graph.set_entry_point("classify")

def route_from_state(state: AgentState):
    return state["route"]

graph.add_conditional_edges(
    "classify",
    route_from_state,
    {"util": "util_agent", "time": "time_agent", "chat": "chat_agent"},
)

graph.add_conditional_edges("util_agent", needs_tools, {"tools": "util_tools", "end": END})
graph.add_conditional_edges("time_agent", needs_tools, {"tools": "time_tools", "end": END})
graph.add_conditional_edges("chat_agent", needs_tools, {"tools": END, "end": END})

graph.add_edge("util_tools", "util_agent")
graph.add_edge("time_tools", "time_agent")

app = graph.compile()

def print_stream(stream):
    print("\n=== STREAM START ===")
    for s in stream:
        msg = s["messages"][-1]
        try:
            msg.pretty_print()
        except Exception:
            print(type(msg), getattr(msg, "content", msg))
    print("=== STREAM END ===\n")

examples = [
    #("user", "Reverse this: LangGraph is awesome"),
    #("user", "What is the word count of: 'This is a tiny test'?"),
    ("user", "What's the current time?"),
    ("user", "Who are you?"),
]

for ex in examples:
    print("\n=== New run ===")
    print_stream(app.stream({"messages": [ex], "route": ""}, stream_mode="values"))
