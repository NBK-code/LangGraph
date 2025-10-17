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
    task: str   # "math" | "chat"

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers"""
    return a + b

tools = [add]

manager_llm = ChatOpenAI(model="gpt-4o-mini")                 # decides the task
math_llm    = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)  # can call `add`
chat_llm    = ChatOpenAI(model="gpt-4o-mini")                 # plain chat

def manager(state: AgentState) -> AgentState:

    system = SystemMessage(content=(
        "Decide which specialist should answer.\n"
        "Reply with EXACTLY one line:\n"
        "TASK: math    (if the user wants arithmetic addition)\n"
        "TASK: chat    (otherwise)\n"
        "No extra text."
    ))
    ai = manager_llm.invoke([system] + list(state["messages"]))

    # Parse directive
    content = ai.content.strip().lower()
    task = "math" if "task:" in content and "math" in content else "chat"
    return {"messages": [ai], "task": task}

def math_agent(state: AgentState) -> AgentState:
    system = SystemMessage(content="Be concise. If needed, CALL the `add` tool with integers.")
    ai = math_llm.invoke([system] + list(state["messages"]))
    return {"messages": [ai]}

def chat_agent(state: AgentState) -> AgentState:
    system = SystemMessage(content="Be concise and helpful. No tools.")
    ai = chat_llm.invoke([system] + list(state["messages"]))
    return {"messages": [ai]}

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
    "chat": "chat_agent",
})
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

print("\n--- Manager routes to MATH specialist ---")
inputs = {"messages": [("user", "Please add 45 and 27.")]}
print_stream(app.stream(inputs, stream_mode="values"))

print("\n--- Manager routes to CHAT specialist ---")
inputs = {"messages": [("user", "What is a multi-agent system in one sentence?")]}
print_stream(app.stream(inputs, stream_mode="values"))
