from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv()

def sum_counts(existing: int | None, new: int | None) -> int:
    ex = existing if existing else 0
    nw = new if new else 0
    return ex + nw

class AgentState(TypedDict, total = False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question_count: Annotated[int, sum_counts]

llm = ChatOpenAI(model = "gpt-4o-mini")

def observe(state: AgentState) -> AgentState:
    last = state['messages'][-1]
    if isinstance(last, HumanMessage):
        return {"question_count": 1}
    return {}

def route(state: AgentState) -> str:
    last = state["messages"][-1]
    text = last.content.lower() if isinstance(last, HumanMessage) else ""
    triggers = ["how many questions", "question count", "how many have i asked"]
    for t in triggers:
        if t in text:
            return "report"
    return "chat"

def memory_report(state: AgentState) -> AgentState:
    qcount = state["question_count"] if "question_count" in state else 0
    return {"messages": [AIMessage(content=f"You have asked {qcount} questions so far.")]}

def chat_agent(state: AgentState) -> AgentState:
    qcount = state["question_count"] if "question_count" in state else 0
    system_prompt = SystemMessage(content = f"Give only one sentence answers. The user has asked {qcount} questions so far.")
    ai = llm.invoke([system_prompt]+list(state["messages"]))
    return {"messages": [ai]}

graph = StateGraph(AgentState)
graph.add_node("observe", observe)
graph.add_node("memory_report", memory_report)
graph.add_node("chat_agent", chat_agent)
graph.set_entry_point("observe")
graph.add_conditional_edges("observe",
                            route,
                            {
                                "report": "memory_report",
                                "chat": "chat_agent"
                            })
graph.add_edge("memory_report", END)
graph.add_edge("chat_agent", END)

app = graph.compile(checkpointer=MemorySaver())

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

THREAD_ID = "demo-thread"

print("---- Turn 1 ----")
inputs = {"messages": [("user", "Hi there!")]}
print_stream(app.stream(inputs, stream_mode="values",
                        config={"configurable": {"thread_id": THREAD_ID}}))

print("\n---- Turn 2 ----")
inputs = {"messages": [("user", "Tell me one fact about LangGraph.")]}
print_stream(app.stream(inputs, stream_mode="values",
                        config={"configurable": {"thread_id": THREAD_ID}}))

print("\n---- Turn 3 (ask for the count) ----")
inputs = {"messages": [("user", "How many questions have I asked so far?")]}
print_stream(app.stream(inputs, stream_mode="values",
                        config={"configurable": {"thread_id": THREAD_ID}}))

print("\n---- Turn 4 ----")
inputs = {"messages": [("user", "Give me one more fact about LangGraph.")]}
print_stream(app.stream(inputs, stream_mode="values",
                        config={"configurable": {"thread_id": THREAD_ID}}))

print("\n---- Turn 5 (ask for the count) ----")
inputs = {"messages": [("user", "How many questions have I asked so far?")]}
print_stream(app.stream(inputs, stream_mode="values",
                        config={"configurable": {"thread_id": THREAD_ID}}))
