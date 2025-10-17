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


def keep_last_two_messages(existing: Sequence[BaseMessage],
                           new: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    merged = list(existing) + list(new)
    return merged[-2:]


class AgentState(TypedDict):
    # This tells LangGraph how to merge message updates into state["messages"]
    #messages: Annotated[Sequence[BaseMessage], keep_last_two_messages]
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int):
    """Multiply two numbers."""
    return a * b

tools = [add, multiply]


llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


def agent_node(state: AgentState) -> AgentState:
    print("\nagent_node sees", len(state["messages"]), "messages")
    print("\nState Messages: ", state["messages"])
    system_prompt = SystemMessage(content="You are my AI assistant. Be concise.")
    response = llm.invoke([system_prompt] + list(state["messages"]))
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
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

inputs = {"messages": [("user", "Add 34 + 41 and multiply the result by 3. Then say hi.")]}
print_stream(app.stream(inputs, stream_mode="values"))
