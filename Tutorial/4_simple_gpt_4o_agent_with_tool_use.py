from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

load_dotenv()

# Annotated - provides additional context without affecting the type, adds to metadata
# Sequence - to automatically handle state updates such as adding a new messages to a chat history
# add_messages - reducer function. tells us how to merge new data into the current data

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

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
    system_prompt = SystemMessage(content = 
                                  "You are my AI assistant. Please answer my query to the best")
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("agent", model_call)
tool_node = ToolNode(tools = tools)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent",
                            should_continue,
                            {
                                "continue": "tools",
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

inputs = {"messages": [("user", "Add 34 + 41 and multiply the result by 3 and tell a joke")]}
print_stream(app.stream(inputs, stream_mode = "values"))




