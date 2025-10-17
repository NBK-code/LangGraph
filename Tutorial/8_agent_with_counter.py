from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    turns: int

@tool
def add(a: int, b: int):
    """This is an addition function that adds two numbers"""
    return a+b

@tool
def multiply(a: int, b: int):
    """This is a multiplication fuction"""
    return a*b

tools = [add, multiply]

llm  = ChatOpenAI(model = "gpt-4o-mini").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    print("\nCounter: ", state["turns"])
    system_prompt = SystemMessage(content="You are my AI assistant. Please answer my query to the best.")
    response = llm.invoke([system_prompt] + list(state["messages"]))
    turns = state["turns"] if "turns" in state else 0
    return {"messages": [response], "turns": turns + 1}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    turns = state["turns"] if "turns" in state else 0
    return "again" if turns < 2 else "end"

    
graph = StateGraph(AgentState)
graph.add_node("agent", model_call)
tool_node = ToolNode(tools = tools)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent",
                            should_continue,
                            {
                                "tools": "tools",
                                "again": "agent",
                                "end": END
                            })
graph.add_edge("tools", "agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "tell a joke")], "turns": 0}
print_stream(app.stream(inputs, stream_mode = "values"))