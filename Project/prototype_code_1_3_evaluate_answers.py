from typing import Annotated, Sequence, TypedDict, Literal
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired
import re
import json
import uuid

load_dotenv()

def sum_counts(existing: int | None, new: int | None) -> int:
    ex = existing if existing else 0
    nw = new if new else 0
    return ex + nw

def extend_dict_list(existing: list[dict] | None, new: list[dict] | None) -> list[dict]:
    ex = existing if existing else []
    nw = new if new else []
    return ex + nw

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question_count: Annotated[int, sum_counts]
    score: Annotated[int, sum_counts]
    subject: NotRequired[str]
    batch: NotRequired[list[dict]]
    cursor: NotRequired[int]
    responses: Annotated[list[dict], extend_dict_list]

class QAItem(TypedDict):
    q_id: str
    question: str
    explanation: NotRequired[str]
    answer: str
    answer_type: Literal["text", "numeric"]

generator_llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.9, model_kwargs = {"response_format": {"type": "json_object"}})
judge_llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.0, model_kwargs={"response_format": {"type": "json_object"}})

def intake(state: AgentState) -> AgentState:
    subject = state.get("subject")
    if not subject:
        subject = input("\nWhat do you want to learn about today? ")
    return {"subject": subject}

def generate_batch_questions(state: AgentState) -> AgentState:
    """Generate a batch of questions"""
    subject = state.get("subject")
    
    system_prompt = SystemMessage(
        content=(
            f"You are a teaching assistant for {subject}.\n"
            "Generate 5 distinct graduate level subject-matter questions AND their correct final answers.\n"
            "Provide a detailed explanation that supports the final answer.\n"
            "The answers to the questions can involve mathematical steps. Both calculus and non-calculus mathematics can be part of the explantion.\n"
            "No meta-questions or follow-ups. Do NOT include the solution text inside `question`.\n"
            "For numeric answers, `answer` must be a bare number string (no units/words). "
            "Return ONLY valid JSON in exactly this structure:\n"
            '{"items":[\n'
            '  {"question":"...", "explanation":"...", "answer":"...", "answer_type":"text|numeric"},\n'
            '  {"question":"...", "explanation":"...", "answer":"...", "answer_type":"text|numeric"},\n'
            '  {"question":"...", "explanation":"...", "answer":"...", "answer_type":"text|numeric"},\n'
            '  {"question":"...", "explanation":"...", "answer":"...", "answer_type":"text|numeric"},\n'
            '  {"question":"...", "explanation":"...", "answer":"...", "answer_type":"text|numeric"}\n'
            ']}'
        ))
    ai = generator_llm.invoke([system_prompt])
    raw = ai.content

    try:
        data = json.loads(raw)
        items = data.get("items", [])
    except Exception:
        items = []

    batch: list[QAItem] = []

    for item in items:
        question = str(item.get("question","")).strip()
        explanation = str(item.get("explanation", "")).strip()
        answer = str(item.get("answer", "")).strip()
        answer_type = (item.get("answer_type") or "text").strip().lower()

        if not question or not answer:
            continue
        if answer_type not in ("text", "numeric"):
            try:
                float(answer)
                answer_type = "numeric"
            except Exception:
                answer_type = "text"

        unique_id = str(uuid.uuid4())

        batch.append({
            "q_id": unique_id,
            "question":question,
            "explanation": explanation if explanation else "",
            "answer": answer,
            "answer_type": answer_type
        })

        if len(batch) == 5:
            break

    if not batch:
        unique_id = str(uuid.uuid4())
        batch = [{
            "q_id": unique_id,
            "question": f"Name one key concept in {subject}.",
            "explanation": "",
            "answer": "Answers vary",
            "answer_type": "text"
        }]

    return {"batch": batch, "cursor": 0, "responses": []}

def get_answer(state: AgentState) -> AgentState:
    """Show the questions and collect answers from the user and advance cursor."""
    batch = state.get("batch") or []
    cursor = state.get("cursor", 0)

    if cursor >= len(batch):
        print("\nAll questions in this batch have been answered.")
        return {}
    
    item  = batch[cursor]
    print(f"\nQ {cursor+1}/{len(batch)}: {item['question']}")
    user_answer = input("Your answer: ").strip()

    hm = HumanMessage(content=user_answer)

    return {"messages": [hm],
            "responses": [{"q_id": item["q_id"], "answer": user_answer}],
            "cursor": cursor + 1,
            "question_count": 1
            }

def evaluate_answer(state: AgentState) -> AgentState:
    """Grade the hust submitted answer (cursor-1) using LLM as a semantic judge.
       Returns score (0-10) and prints a short rationale."""
    
    batch = state.get("batch") or []
    cursor = state.get("cursor", 0)
    index = cursor - 1

    if index < 0 or index >= len(batch):
        return {}
    
    item = batch[index]
    qid = item["q_id"]

    user_answer = ""
    for user_response in reversed(state.get("responses", [])):
        if user_response.get("q_id") == qid:
            user_answer = str(user_response.get("answer", "")).strip()
            break

    system_prompt = SystemMessage(content=(
        "You are a fair grader focussed more on the conceptual understanding of the student.\n"
        "Judge the student's answer against the ground truth answer.\n"
        "Consider synonyms, paraphrases, and numeric/text equivalence (e.g., '+1' vs 'positive').\n"
        "Give a score between 0 to 10, with 0 for a blank response and 10 for a perfect response.\n"
        "Also provide a rationale for the score you assign to a particular answer.\n"
        'Return ONLY JSON: {"score": <int 0..10>, "explanation": "<short rationale>"}'
    ))

    payload = {
        "question": item["question"],
        "ground_truth_answer": item["answer"],
        "answer_type": item["answer_type"],
        "model_explanation": item.get("explanation", ""),
        "student_answer": user_answer
    }

    judge_raw = judge_llm.invoke([system_prompt, HumanMessage(content=json.dumps(payload, ensure_ascii=False))]).content

    try:
        obj = json.loads(judge_raw)
        score = int(obj.get("score", 0))
        reason = str(obj.get("explanation", "")).strip()
    except Exception:
        m = re.search(r"\b(\d{1,2})\b", judge_raw)       # look for a 1–2 digit integer anywhere
        score = int(m.group(1)) if m else 0              # if found, use it as the score; else 0
        reason = judge_raw.strip()                       # keep the whole raw text as the rationale

    score = max(0, min(10, score))
    print(f"Score: {score}/10")
    if reason:
        print("Reason:", reason)

    return {"score": score}

def more_questions(state: AgentState) -> str:
    batch = state.get("batch") or []
    cursor = state.get("cursor", 0)
    return "more" if cursor < len(batch) else "done"


def debug_show_batch(state: AgentState) -> AgentState:
    batch = state.get("batch", [])
    total_pts = len(batch) * 10
    total_sc = state.get("score", 0)
    pct = (100.0 * total_sc / total_pts) if total_pts else 0.0
    print("\nBatch complete.")
    print(f"Score: {total_sc}/{total_pts}  (Average: {pct:.1f}%)")
    for i, it in enumerate(batch, 1):
        print(f"{i}. {it['question']}")
        if it.get("explanation"):
            print("   exp:", it["explanation"])
        print("   ans:", it["answer"])
    print()
    return {}


graph = StateGraph(AgentState)
graph.add_node("intake", intake)
graph.add_node("generate_batch_questions", generate_batch_questions)
graph.add_node("get_answers", get_answer)
graph.add_node("evaluate_answer", evaluate_answer)
graph.add_node("debug_show_batch", debug_show_batch)  # summary at end

graph.add_edge(START, "intake")
graph.add_edge("intake", "generate_batch_questions")
graph.add_edge("generate_batch_questions", "get_answers")
graph.add_edge("get_answers", "evaluate_answer")

# Loop until we've evaluated all Qs, then summarize
graph.add_conditional_edges(
    "evaluate_answer",
    more_questions,
    {"more": "get_answers", "done": "debug_show_batch"}
)

graph.add_edge("debug_show_batch", END)

app = graph.compile()

# ---------- Run ----------
init: AgentState = {
    "messages": [],
    "question_count": 0,
    "score": 0,
    "subject": "",   
    "batch": [],
    "cursor": 0,
    "responses": []
}
final_state = app.invoke(init)