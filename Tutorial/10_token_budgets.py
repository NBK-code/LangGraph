from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

MAX_STEPS = 5 #max number of LLM replies
MAX_TOKENS_IN = 100 #cumulative prompt tokens allowed
MAX_TOKENS_OUT = 100 #cumulative completion tokens allowed


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    steps: int
    tokens_in: int
    tokens_out: int

llm = ChatOpenAI(model = 'gpt-4o-mini')

def agent(state: AgentState) -> AgentState:
    """Call the LLM, then increment step + token counters from OpenAI usage metadata"""
    system_prompt = SystemMessage(content = "Be brief. Continue the discussion.")
    ai = llm.invoke([system_prompt]+list(state['messages']))

    usage = {}

    try:
        usage = ai.response_metadata.get("token_usage", {}) or {}
    except Exception:
        usage = {}

    prompt_toks = int(usage.get("prompt_tokens", 0))
    completion_toks = int(usage.get("completion_tokens", 0))

    steps = state['steps'] + 1
    tokens_in = state["tokens_in"] + prompt_toks
    tokens_out = state["tokens_out"] + completion_toks
    
    return {"messages": [ai], "steps": steps, "tokens_in": tokens_in, "tokens_out": tokens_out}

def safety_stop(state: AgentState) -> AgentState:
    msg = AIMessage(
        content=(
            f"⛔ Stopping due to budget.\n"
            f"- steps = {state['steps']} / {MAX_STEPS}\n"
            f"- tokens_in (prompt) = {state['tokens_in']} / {MAX_TOKENS_IN}\n"
            f"- tokens_out (completion) = {state['tokens_out']} / {MAX_TOKENS_OUT}"
        )
    )
    return {"messages": [msg]}

def should_continue(state: AgentState):
    if state["steps"] >= MAX_STEPS:
        return "stop"
    if state["tokens_in"] >= MAX_TOKENS_IN:
        return "stop"
    if state["tokens_out"] >= MAX_TOKENS_OUT:
        return "stop"
    return "again"

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("safety_stop", safety_stop)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"again": "agent", "stop": "safety_stop"})
graph.add_edge("safety_stop", END)
app = graph.compile()

def print_stream(stream):
    print("\n=== STREAM START ===")
    for s in stream:
        m = s["messages"][-1]
        try:
            m.pretty_print()
        except Exception:
            print(getattr(m, "content", m))
    print("=== STREAM END ===\n")

inputs = {
    "messages": [("user", "Start a short conversation about Test Time Training (TTT). Keep replying each turn.")],
    "steps": 0,
    "tokens_in": 0,
    "tokens_out": 0,
}
print_stream(app.stream(inputs, stream_mode="values"))