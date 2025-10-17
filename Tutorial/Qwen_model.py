from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
import os, torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

print("Hello")

torch.set_num_threads(2)

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float32,   # CPU uses float32; keeps things simple and reliable
    low_cpu_mem_usage=True)

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
    do_sample=False,
    max_new_tokens=128,
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
    return_full_text=False        # don't echo the prompt back
)

llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen_pipe))

class AgentState(TypedDict):
    messages: List[HumanMessage]

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print("\nAI:", response.content)
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("User: ")

while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("\nUser: ")
