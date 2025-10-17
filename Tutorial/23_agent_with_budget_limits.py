from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

def sum_ints(existing: int | None, new: int | None) -> int:
    ex = existing if existing else 0
    nw = new if new else 0
    return ex + nw

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    searches_left: Annotated[int, sum_ints]
    max_chars: int

search_tool: BaseTool = DuckDuckGoSearchResults()
search_tool.name = "web_search"  # OpenAI-safe name
tools = [search_tool]
tools_by_name = {t.name: t for t in tools}

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def agent(state: AgentState) -> AgentState:
    budget = state.get("searches_left", 0)
    char_cap = state.get("max_chars", 320)
    system = SystemMessage(content="Be concise. If the user asks for up-to-date or web info, CALL `web_search` "
        f"(you have {budget} search call(s) available). Keep your final answer under ~{char_cap} characters.")
    ai = llm.invoke([system]+list(state["messages"]))
    return {"messages": [ai]}

def need_tools_or_trim(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "trim"

def tools_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    calls = last.tool_calls if isinstance(last, AIMessage) else []
    outs = []
    executed = 0
    remaining = int(state.get("searches_left",0))

    for call in calls:
        if remaining <= 0:
            outs.append(ToolMessage(content="BUDGET: search budget exhausted.", tool_call_id=call["id"]))
            break
        tool = tools_by_name.get(call["name"])
        args = call.get("args") or {}
        if not tool:
            outs.append(ToolMessage(content=f"ERROR: unknown tool '{call['name']}'", tool_call_id=call["id"]))
            continue
        try:
            result = tool.invoke(args)
        except Exception as e:
            outs.append(ToolMessage(content=f"ERROR: {e}", tool_call_id=call["id"]))
            continue
        executed += 1
        remaining -= 1
        outs.append(ToolMessage(content=f"OK: {str(result)[:300]}", tool_call_id=call["id"]))

    out: AgentState = {"messages": outs}
    if executed:
        out["searches_left"] = -executed
    return out

def trim(state: AgentState) -> AgentState:
    cap = int(state.get("max_chars", 320))
    last = state["messages"][-1]
    if isinstance(last, AIMessage):
        content = last.content
        if len(content) > cap:
            content = content[:cap-1].rstrip() + "..."
        return {"messages": [AIMessage(content=content)]}
    return {}

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", tools_node)
graph.add_node("trim", trim)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", need_tools_or_trim, {"tools": "tools", "trim": "trim"})
graph.add_edge("tools", "agent")  # after tools, return to agent to compose answer
graph.add_edge("trim", END)

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

print("\n--- With small budgets (1 search, 280 chars) ---")
inputs = {
    "messages": [("user", "Search the web for LangGraph intro and summarize briefly.")],
    "searches_left": 1,   # start budget
    "max_chars": 280,     # output size cap
}
print_stream(app.stream(inputs, stream_mode="values"))

print("\n--- Budget exhausted (0 searches allowed) ---")
inputs = {
    "messages": [("user", "Search the web for LangGraph tutorials.")],
    "searches_left": 0,   # no searches left
    "max_chars": 200,
}
print_stream(app.stream(inputs, stream_mode="values"))

