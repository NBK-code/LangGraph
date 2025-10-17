from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

def sum_counts(existing: int | None, new: int | None) -> int:
    ex = existing if existing else 0
    nw = new if new else 0
    return ex + nw

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question_count: Annotated[int, sum_counts]

llm = ChatOpenAI(model = "gpt-4o-mini")

def observe(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        return {"question_count" : 1}
    return {}

def agent(state: AgentState) -> AgentState:
    qcount = state["question_count"] if "question_count" in state else 0
    system_prompt = SystemMessage(content = 
                                  f"Give only one sentence answer. The user has asked {qcount} questions so far.")
    ai = llm.invoke([system_prompt] + list(state["messages"]))
    return {"messages" : [ai]}

graph = StateGraph(AgentState)
graph.add_node("observe", observe)
graph.add_node("agent", agent)
graph.set_entry_point("observe")
graph.add_edge("observe", "agent")
graph.add_edge("agent", END)

app = graph.compile(checkpointer = MemorySaver())

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

THREAD_ID = "demo"

inputs = {"messages": [("user", "Hi, what is LangGraph?")]}
print_stream(app.stream(inputs, stream_mode="values", 
                        config = {"configurable": {"thread_id": THREAD_ID}}))

inputs = {"messages": [("user", "Give me one more sentence about it.")]}
print_stream(app.stream(inputs, stream_mode="values", 
                        config = {"configurable": {"thread_id": THREAD_ID}}))

inputs = {"messages": [("user", "Give me one more sentence about it.")]}
print_stream(app.stream(inputs, stream_mode="values", 
                        config = {"configurable": {"thread_id": "new_thread"}}))


