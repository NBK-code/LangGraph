from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

def count(existing: int | None, new: int | None) -> int:
    ex = existing if existing else 0
    nw = new if new else 0
    return ex + nw

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    steps: Annotated[int, count]

llm = ChatOpenAI(model = "gpt-4o-mini")

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="Give only one line answer and continue the discussion")
    ai = llm.invoke([system_prompt] + list(state["messages"]))
    #print("\nAI: ", ai.content)
    return {"messages": [ai], "steps": 1}

def should_continue(state: AgentState) -> str:
    count_now = state["steps"]
    print(f"\nCount: {count_now}")
    if count_now < 3:
        return "continue"
    else:
        return "end"
    
graph = StateGraph(AgentState)
graph.add_node("agent", model_call)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent",
                            should_continue,
                            {
                                "continue": "agent",
                                "end": END
                            })
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Talk about LangGraph.")]}
print_stream(app.stream(inputs, stream_mode = "values"))