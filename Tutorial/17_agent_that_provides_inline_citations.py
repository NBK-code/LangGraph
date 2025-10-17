from typing import Annotated, TypedDict, Sequence, List, Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

def replace_sources(existing: List[Dict[str, str]] | None,
                    new: List[Dict[str, str]] | None) -> List[Dict[str, str]]:
    return list(new or []) #last write wins

class AgentState(TypedDict, total = False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    route: str
    sources: Annotated[List[Dict[str, str]], replace_sources]

search_tool: BaseTool = DuckDuckGoSearchResults()
search_tool.name = "web_search"
tools = [search_tool]
tools_by_name = {t.name: t for t in tools}

llm_search = ChatOpenAI(model = "gpt-4o-mini").bind_tools(tools)
llm_chat = ChatOpenAI(model="gpt-4o-mini")

def classify(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    text = last.content.lower() if isinstance(last, HumanMessage) else ""
    triggers = ["search", "look up", "find online", "web", "latest", "news", "sources"]
    route = "search" if any(t in text for t in triggers) else "chat"
    return {"route": route}

#We have 2 routes. Each route has a llm invocation with its own system prompt.

def search_agent(state: AgentState) -> AgentState:
    """
    The model should call `web_search` when needed. After the tool runs,
    it will see a ToolMessage that enumerates [1], [2], ...; it should cite them.
    """
    system = SystemMessage(content=(
        "You are concise. When information may be outdated or the user asks to search, "
        "call `web_search` with a short query. After results arrive, write 2–4 bullets and "
        "add inline citations like [1], [2] keyed to the enumerated Sources list. "
        "Keep the answer brief."
    ))
    ai = llm_search.invoke([system] + list(state["messages"]))
    return {"messages": [ai]}

def chat_agent(state: AgentState) -> AgentState:
    system = SystemMessage(content="You are concise. Answer directly without searching.")
    ai = llm_chat.invoke([system] + list(state["messages"]))
    return {"messages": [ai]}

#Custom Tool Node
def _normalize(results: Any, k: int = 5) -> List[Dict[str, str]]:
    items = results if isinstance(results, list) else [results]
    norm = []
    for r in items[:k]:
        if isinstance(r, dict):
            title = r.get("title") or r.get("name") or "Untitled"
            url = r.get("link") or r.get("url") or r.get("href") or ""
            snippet = r.get("snippet") or r.get("body") or r.get("text") or ""
        else:
            title, url, snippet = str(r)[:80], "", ""
        norm.append({"title": str(title), "url": str(url), "snippet": str(snippet)})
    return norm

def custom_tools(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    tool_calls = last.tool_calls if isinstance(last, AIMessage) else []
    out_msgs: List[ToolMessage] = []
    saved_sources: List[Dict[str, str]] = []

    for call in tool_calls:
        name  = call["name"]
        args = call.get("args", {}) or {}
        tool = tools_by_name.get(name)
        if tool is None:
            out_msgs.append(ToolMessage(content=f"ERROR: unknown tool {name}", tool_call_id = call["id"]))
            continue

        try:
            results = tool.invoke(args)
        except Exception as e:
            out_msgs.append(ToolMessage(content=f"ERROR: {e}", tool_call_id = call["id"]))
            continue

        saved_sources = _normalize(results, k=5)
        lines = [f"Found {len(saved_sources)} results(s). Sources:"]
        for i,s in enumerate(saved_sources, start=1):
            lines.append(f"[{i}] {s['title']} - {s['url']}")
        out_msgs.append(ToolMessage(content="\n".join(lines), tool_call_id = call["id"]))

    out: AgentState = {"messages": out_msgs}
    if saved_sources:
        out["sources"] = saved_sources
    return out

def needs_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "end"

graph = StateGraph(AgentState)
graph.add_node("classify", classify)
graph.add_node("search_agent", search_agent)
graph.add_node("chat_agent", chat_agent)
graph.add_node("tools", custom_tools)

graph.set_entry_point("classify")

def route_from_state(state: AgentState):
    return state["route"]

graph.add_conditional_edges("classify", route_from_state, {"search": "search_agent", "chat": "chat_agent"})
graph.add_conditional_edges("search_agent", needs_tools, {"tools": "tools", "end": END})
graph.add_edge("tools", "search_agent")      # read results, write the summary
graph.add_conditional_edges("chat_agent", needs_tools, {"end": END})

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


print("\n---- Search with citations ----")
inputs = {"messages": [("user", "Search the web for LangGraph basics and summarize with citations.")], "route": ""}
print_stream(app.stream(inputs, stream_mode="values"))

print("\n---- Chat path (no search) ----")
inputs = {"messages": [("user", "What is LangGraph in one sentence?")], "route": ""}
print_stream(app.stream(inputs, stream_mode="values"))



