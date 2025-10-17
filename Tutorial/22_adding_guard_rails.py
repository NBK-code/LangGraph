import re
from typing import Annotated, Sequence, TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    route: str # ok | Clarify

llm = ChatOpenAI(model="gpt-4o-mini")

FORBIDDEN_TERMS = {"password", "api key", "apikey", "credit card"}

EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.I)
URL_RE   = re.compile(r"https?://", re.I)
CC_RE    = re.compile(r"\b\d{13,16}\b")  # naive long number check

def check_guard(text: str) -> List[str]:
    reasons = []
    low = text.lower()

    if len(text) > 160:
        reasons.append("too long")
    if any(term in low for term in FORBIDDEN_TERMS):
        reasons.append("mentions sensitive terms")
    if EMAIL_RE.search(text):
        reasons.append("contains email address")
    if URL_RE.search(text):
        reasons.append("contains a URL")
    if CC_RE.search(text):
        reasons.append("contains a long numeric sequence (possible card number)")
    return reasons

def guard(state: AgentState) -> AgentState:
    last = state['messages'][-1]
    user_text = last.content if isinstance(last, HumanMessage) else ""
    reasons = check_guard(user_text)

    if reasons:
        msg = AIMessage(content="Guard: needs clarification — " + "; ".join(reasons))
        return {"messages": [msg], "route": "clarify"}
    else:
        msg = AIMessage(content="Guard: ok")
        return {"messages": [msg], "route": "ok"}
    
def route_from_guard(state: AgentState) -> str:
    return state.get("route", "clarify")

def agent(state: AgentState) -> AgentState:
    system = SystemMessage(content="Be concise and helpful")
    ai = llm.invoke([system] + list(state["messages"]))
    return {"messages": [ai]}

def clarify(state: AgentState) -> AgentState:
    tip = ("Please rephrase without sensitive info (no passwords/API keys/emails/URLs), "
        "and keep it under 160 characters.")
    return {"messages": [AIMessage(content=tip)]}

graph = StateGraph(AgentState)
graph.add_node("guard", guard)
graph.add_node("agent", agent)
graph.add_node("clarify", clarify)
graph.set_entry_point("guard")
graph.add_conditional_edges("guard", route_from_guard, {"ok":"agent", "clarify": "clarify"})
graph.add_edge("agent", END)
graph.add_edge("clarify", END)

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

print("\n--- PASS (goes to agent) ---")
inputs = {"messages": [("user", "What is LangGraph in one sentence?")]}
print_stream(app.stream(inputs, stream_mode="values"))

print("\n--- FAIL (mentions password → clarify) ---")
inputs = {"messages": [("user", "My password is hunter2; what should I do next?")]}
print_stream(app.stream(inputs, stream_mode="values"))

print("\n--- FAIL (too long → clarify) ---")
inputs = {"messages": [("user", "x"*200)]}
print_stream(app.stream(inputs, stream_mode="values"))