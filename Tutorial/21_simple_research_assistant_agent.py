from typing import Annotated, Sequence, TypedDict, List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

def replace_list(existing: List[str] | None, new: List[str] | None) -> List[str]:
    nw = new if new else []
    return list(nw)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    queries: Annotated[List[str], replace_list]

search_tool: BaseTool = DuckDuckGoSearchResults()
search_tool.name = "web_search"
tools = [search_tool]

llm_planner = ChatOpenAI(model = "gpt-4o-mini")
llm_researcher = ChatOpenAI(model = "gpt-4o-mini").bind_tools(tools)
llm_writer = ChatOpenAI(model = "gpt-4o-mini")

def planner(state: AgentState) -> AgentState:
    """Turn user requests into 2-3 short queries"""
    system_prompt = SystemMessage(content="Produce 2-3 short, specific web search queries. One per line.")
    ai = llm_planner.invoke([system_prompt]+list(state["messages"]))
    queries = [ln.strip("-• ").strip() for ln in ai.content.splitlines() if ln.strip()][:3]
    if not queries:
        last = state["messages"][-1]
        queries = [last.content.strip()] if isinstance(last, HumanMessage) else ["LangGraph basics"]
    plan = AIMessage(content="Plan:\n"+"\n".join(f"-{q}" for q in queries))
    return {"messages": [plan], "queries": queries}
    
def researcher(state: AgentState) -> AgentState:
    """Ask the model to call web_search for eachplanned query."""
    qs = state.get("queries", [])
    system_prompt = SystemMessage(content="For each query below, CALL `web_search` once with {query: <text>}.\n"
                + "\n".join(f"- {q}" for q in qs)
    )
    ai = llm_researcher.invoke([system_prompt]+list(state["messages"]))
    return {"messages": [ai]}

def writer(state: AgentState) -> AgentState:
    """Write a brief answer using whatever research/tool output is now in the messages."""
    system = SystemMessage(
        content=(
            "Write 3 to 5 concise bullets answering the user based on the latest research context above. "
            "Do not include URLs or citations. Keep it tight."
        )
    )
    ai = llm_writer.invoke([system] + list(state["messages"]))
    return {"messages": [ai]}

def needs_tools_or_next(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "next"

graph = StateGraph(AgentState)
graph.add_node("planner", planner)
graph.add_node("researcher", researcher)
graph.add_node("tools", ToolNode(tools=tools))
graph.add_node("writer", writer)
graph.set_entry_point("planner")
graph.add_edge("planner", "researcher")
graph.add_conditional_edges("researcher", needs_tools_or_next, {"tools": "tools", "next": "writer"})
graph.add_edge("tools", "researcher")
graph.add_edge("writer", END)

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

print("\n--- Planner → Researcher → Writer (no citations) ---")
inputs = {"messages": [("user", "Find good introductions to LangGraph and summarize the basics.")]}
print_stream(app.stream(inputs, stream_mode="values"))