from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

search_tool: BaseTool = DuckDuckGoSearchRun()
search_tool.name = "web_search"

tools = [search_tool]

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def agent(state: AgentState) -> AgentState:
    system = SystemMessage(content=(
        "You are concise. When the user asks to look something up, "
        "call the `web_search` tool with a short, specific query. "
        "Otherwise, answer directly."
    ))
    ai = llm.invoke([system] + list(state["messages"]))
    return {"messages": [ai]}

def needs_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools=tools))

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", needs_tools, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

print("\n---- Example 1: direct answer (likely no tool) ----")
inputs = {"messages": [("user", "What is LangGraph in one sentence?")]}
print_stream(app.stream(inputs, stream_mode="values"))

print("\n---- Example 2: force a search ----")
inputs = {"messages": [("user", "Search the web for LangGraph tutorials and show top results.")]}
print_stream(app.stream(inputs, stream_mode="values"))
