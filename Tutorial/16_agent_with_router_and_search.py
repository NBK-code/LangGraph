from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    route: str

search_tool: BaseTool = DuckDuckGoSearchRun()
search_tool.name = "web_search"

tools = [search_tool]

llm_search = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
llm_chat = ChatOpenAI(model="gpt-4o-mini")

def classify(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    text = last.content.lower() if isinstance(last, HumanMessage) else ""
    triggers = ["search", "look up", "find online", "web", "latest", "news", "according to", "sources"]
    route = "search" if any(t in text for t in triggers) else "chat"
    return {"route": route}

def search_agent(state: AgentState) -> AgentState:
    system = SystemMessage(content=(
        "You are concise. If information may be outdated or the user asks to search, "
        "CALL `web_search` with a short, specific query. Then summarize briefly."
    ))
    ai = llm_search.invoke([system] + list(state["messages"]))
    return {"messages": [ai]}

def chat_agent(state: AgentState) -> AgentState:
    system = SystemMessage(content="You are concise. Answer directly without web search.")
    ai = llm_chat.invoke([system] + list(state["messages"]))
    return {"messages": [ai]}

def needs_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "end"

graph = StateGraph(AgentState)
graph.add_node("classify", classify)
graph.add_node("search_agent", search_agent)
graph.add_node("chat_agent", chat_agent)
graph.add_node("tools", ToolNode(tools=tools))

graph.set_entry_point("classify")

def route_from_state(state: AgentState):
    return state["route"]

graph.add_conditional_edges(
    "classify",
    route_from_state,
    {"search": "search_agent", "chat": "chat_agent"},
)

graph.add_conditional_edges("search_agent", needs_tools, {"tools": "tools", "end": END})
graph.add_edge("tools", "search_agent")   # after tool runs, go back to the search agent to read results
graph.add_conditional_edges("chat_agent", needs_tools, {"end": END})  # chat has no tools

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

print("\n---- Chat path (no search) ----")
inputs = {"messages": [("user", "Explain LangGraph in one sentence.")], "route": ""}
print_stream(app.stream(inputs, stream_mode="values"))

print("\n---- Search path (explicit search) ----")
inputs = {"messages": [("user", "Search the web for LangGraph tutorials and list top results.")], "route": ""}
print_stream(app.stream(inputs, stream_mode="values"))

print("\n---- Search path (implicit: latest/news) ----")
inputs = {"messages": [("user", "What are the latest news about LangGraph?")], "route": ""}
print_stream(app.stream(inputs, stream_mode="values"))