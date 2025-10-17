from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired
import re

load_dotenv()

def sum_counts(existing: int | None, new: int | None) -> int:
    ex = existing if existing else 0
    nw = new if new else 0
    return ex + nw

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question_count: Annotated[int, sum_counts]
    score: Annotated[int, sum_counts]
    subject: NotRequired[str]

llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.6)

def generate_question(state: AgentState) -> AgentState:
    """Generate a question"""
    subject = state.get("subject")
    if not subject:
        subject = input("What do you want to learn about today? ")
    system_prompt = SystemMessage(
        content=f"You are a teaching assistant for {subject}. Generate ONE short question the user can solve. Do not include the solution. Return one short question only."
    )
    ai = llm.invoke([system_prompt])
    # carry the subject forward by returning it; keep your counter +1
    return {"messages": [ai], "question_count": 1, "subject": subject}

def get_answer(state: AgentState) -> AgentState:
    """Get the answer from the user"""
    last = state["messages"][-1]
    print(last.content)
    answer = input("Enter your answer: ")
    return {"messages": [HumanMessage(content = answer)]}

def get_scores(ai_msg: AIMessage) -> int:
    # extract first integer 1..10; clamp just in case
    m = re.search(r"\b(\d{1,2})\b", ai_msg.content)
    val = int(m.group(1)) if m else 0
    return max(0, min(10, val))

def evaluate_answer(state: AgentState) -> AgentState:
    """Evaluate answer here"""
    msgs = list(state["messages"])
    last_q = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
    last_a = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
    system_prompt = SystemMessage(
        content="You are an evaluator. Score the student's LAST answer to the LAST question on a 1–10 scale."
                " Return ONLY the integer (no text)."
    )
    ai = llm.invoke([system_prompt, last_q, last_a])
    print("Evaluator message: ", ai.content)
    score = get_scores(ai)
    return {"messages": [ai], "score": score}

def should_continue(state: AgentState) -> str:
    """Count question counts. If it is less that 5 loop back to generate_question, get_answer, evaluate_answer.
    Else go to summarizer"""
    return "continue" if (state.get("question_count") or 0) < 5 else "end"


def summarizer(state: AgentState) -> AgentState:
    """Summarize the performance and display the score"""
    q = state.get("question_count") or 0
    sc = state.get("score") or 0
    avg = sc / q if q else 0
    summary = AIMessage(content=f"Session over.\nQuestions: {q}\nTotal score: {sc}\nAverage: {avg:.2f}/10")
    print(summary.content)
    return {"messages": [summary]}


graph = StateGraph(AgentState)
graph.add_node("generate_question", generate_question)
graph.add_node("get_answer", get_answer)
graph.add_node("evaluate_answer", evaluate_answer)
graph.add_node("summarizer", summarizer)
graph.set_entry_point("generate_question")
graph.add_edge("generate_question", "get_answer")
graph.add_edge("get_answer", "evaluate_answer")
graph.add_conditional_edges("evaluate_answer", should_continue, {"continue": "generate_question", "end": "summarizer"})
graph.add_edge("summarizer", END)

app = graph.compile()

init = {"messages": [], "question_count": 0, "score": 0, "subject": ""}
app.invoke(init)









