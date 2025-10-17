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

def extend_int_list(existing: list[int] | None, new: list[int] | None) -> list[int]:
    ex = existing if existing else []
    nw = new if new else []
    return ex + nw

def extend_str_list(existing: list[str] | None, new: list[str] | None) -> list[str]:
    ex = existing if existing else []
    nw = new if new else []
    return ex + nw

Level = Literal[
    "Elementary School Level",
    "Middle School Level",
    "High School Level",
    "Undergraduate Level",
    "Advanced Undergraduate Level",
    "Graduate Level",
    "Advanced Graduate Level",
]

LEVELS: list[Level] = [
    "Elementary School Level",
    "Middle School Level",
    "High School Level",
    "Undergraduate Level",
    "Advanced Undergraduate Level",
    "Graduate Level",
    "Advanced Graduate Level",
]

LEVEL_DESC: dict[Level, str] = {
    "Elementary School Level": "Single-fact recall; everyday language; no calculations.",
    "Middle School Level": "1–2 step reasoning; simple numerics; basic units/sign awareness.",
    "High School Level": "Multi-step reasoning; algebraic manipulation; standard science vocabulary.",
    "Undergraduate Level": "Conceptual + quantitative; occasional calculus; brief justification.",
    "Advanced Undergraduate Level": "Multi-concept synthesis; approximations; careful units & error.",
    "Graduate Level": "Rigorous definitions; nontrivial derivations; edge cases & assumptions.",
    "Advanced Graduate Level": "Research-style twists; novel combinations; concise formal arguments.",
}

PROMOTE_THRESHOLD = 8.5
DEMOTE_THRESHOLD  = 6.5

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question_count: Annotated[int, sum_counts]
    score: Annotated[int, sum_counts]
    subject: NotRequired[str]
    #batch content/flow
    batch: NotRequired[list[dict]]
    cursor: NotRequired[int]
    responses: Annotated[list[dict], extend_dict_list]
    batch_avg: NotRequired[float]
    #adaptivity
    level: NotRequired[Level]
    seen_questions: Annotated[list[str], extend_str_list]
    batch_scores: Annotated[list[int], extend_int_list]
    continue_flag: NotRequired[bool]

class QAItem(TypedDict):
    q_id: str
    question: str
    explanation: NotRequired[str]
    answer: str
    answer_type: Literal["text", "numeric"]

generator_llm = ChatOpenAI(
    model = "gpt-4o-mini", 
    temperature = 0.9, 
    model_kwargs = {"response_format": {"type": "json_object"}})

judge_llm = ChatOpenAI(
    model = "gpt-4o-mini", 
    temperature = 0.0, 
    model_kwargs={"response_format": {"type": "json_object"}})

def intake(state: AgentState) -> AgentState:
    subject = state.get("subject")
    if not subject:
        subject = input("\nWhat do you want to learn about today? ")
    
    level = state.get("level")
    if not level:
        print("\nChoose a starting level:")
        for i, name in enumerate(LEVELS, 1):
            print(f"{i}. {name}")
        choice = input("Enter 1-7 (default 3 for High School Level): ").strip()

        try:
            idx = int(choice)-1
            if 0 <= idx < len(LEVELS):
                level = LEVELS[idx]
            else:
                level = "High School Level"
        except Exception:
            level = "High School Level"
        print(f"Starting at: {level}")
    return {"subject": subject, "level": level}


def generate_batch_questions(state: AgentState) -> AgentState:
    """Generate a batch of questions"""
    subject = state.get("subject")
    level: Level = state.get("level", "High School Level")
    avoid_list = state.get("seen_questions", [])[-12:]
    avoid_text = "\n-" + "\n-".join(avoid_list) if avoid_list else " (none)"
    
    system_prompt = SystemMessage(
        content=(
            f"You are a teaching assistant for {subject}.\n"
            f"Target **{level}**.\n"
            f"Level profile: {LEVEL_DESC[level]}\n\n"
            "Generate 5 distinct short subject-matter questions AND their correct final answers.\n"
            "Provide a brief explanation (1–3 sentences) that supports the final answer.\n"
            "No meta-questions or follow-ups. Do NOT include the solution value inside `question`.\n"
            "Avoid repeating or paraphrasing ANY of these recent questions:\n"
            f"{avoid_text}\n\n"
            "For numeric answers, `answer` must be a bare number string (no units/words).\n"
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
    seen = set(state.get("seen_questions", []))

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

        if question in seen:
            continue

        unique_id = str(uuid.uuid4())

        batch.append({
            "q_id": unique_id,
            "question":question,
            "explanation": explanation if explanation else "",
            "answer": answer,
            "answer_type": answer_type
        })

        seen.add(question)

        if len(batch) == 5:
            break

    if not batch:
        q = f"Name one key concept in {subject}."
        unique_id = str(uuid.uuid4())
        batch = [{
            "q_id": unique_id,
            "question": q,
            "explanation": "",
            "answer": "Answers vary",
            "answer_type": "text"
        }]
        seen.add(q)

    return {"batch": batch, 
            "cursor": 0, 
            "responses": [],
            "batch_scores": [],
            "seen_questions": list(seen)}

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

    return {"score": score, "batch_scores": [score]}

def more_questions(state: AgentState) -> str:
    batch = state.get("batch") or []
    cursor = state.get("cursor", 0)
    return "more" if cursor < len(batch) else "done"


def debug_show_batch(state: AgentState) -> AgentState:
    batch = state.get("batch", [])
    n = len(batch)
    last_scores = state.get("batch_scores", [])
    batch_total = sum(last_scores)
    
    if n and len(last_scores) != n:
        #Not all the questions in the batch were graded
        n_eff = max(1, len(last_scores))
        batch_avg = batch_total/n_eff
    else:
        batch_avg = (batch_total / n) if n else 0.0

    print("\nBatch complete.")
    print(f"Batch avg: {batch_avg:.2f}/10")
    print(f"Running total: {state.get('score',0)}/{state.get('question_count',0)*10}")

    for i, it in enumerate(batch, 1):
        print(f"{i}. {it['question']}")
        if it.get("explanation"):
            print("   exp:", it["explanation"])
        print("   ans:", it["answer"])
    print()

    #print("\nSeen questions:")
    #print(state["seen_questions"])
    #print("\nResponses:")
    #print(state["responses"])
    return {"batch_avg": batch_avg}

def decide_next_level(state: AgentState) -> AgentState:
    """Promote / stay / demote one level based on batch_avg"""
    level: Level = state.get("level", "High School Level")
    batch_avg = float(state.get("batch_avg", 0.0))
    idx = LEVELS.index(level)

    if batch_avg >= PROMOTE_THRESHOLD and idx < len(LEVELS) - 1:
        new_level = LEVELS[idx + 1]
        movement = "↑ promote"
    elif batch_avg < DEMOTE_THRESHOLD and idx > 0:
        new_level = LEVELS[idx - 1]
        movement = "↓ demote"
    else:
        new_level = level
        movement = "→ stay"

    print(f"Level decision: {movement} — {level} → {new_level}")
    return {"level": new_level}

def ask_continue(state: AgentState) -> AgentState:
    """Ask the learner if they want another batch at the (possibly new) level."""
    choice = input("\nDo you want another batch? (y/n): ").strip().lower()
    cont = choice in {"y", "yes", "1"}
    return {"continue_flag": cont}

def continue_or_end(state: AgentState) -> str:
    return "continue" if state.get("continue_flag") else "end"


graph = StateGraph(AgentState)
graph.add_node("intake", intake)
graph.add_node("generate_batch_questions", generate_batch_questions)
graph.add_node("get_answers", get_answer)
graph.add_node("evaluate_answer", evaluate_answer)
graph.add_node("debug_show_batch", debug_show_batch)     # summarize & compute batch_avg
graph.add_node("decide_next_level", decide_next_level)   # adjust state.level
graph.add_node("ask_continue", ask_continue)             # user choice

graph.add_edge(START, "intake")
graph.add_edge("intake", "generate_batch_questions")
graph.add_edge("generate_batch_questions", "get_answers")
graph.add_edge("get_answers", "evaluate_answer")

# Loop per batch
graph.add_conditional_edges("evaluate_answer", more_questions, {"more": "get_answers", "done": "debug_show_batch"})
graph.add_edge("debug_show_batch", "decide_next_level")
graph.add_edge("decide_next_level", "ask_continue")
graph.add_conditional_edges("ask_continue", continue_or_end, {"continue": "generate_batch_questions", "end": END})

app = graph.compile()

# ---------- Run ----------
init: AgentState = {
    "messages": [],
    "question_count": 0,
    "score": 0,
    "subject": "",
    "batch": [],
    "cursor": 0,
    "responses": [],
    "seen_questions": [],
    "batch_scores": [],
    "level": "",  # you can preset; intake lets the user change
}
final_state = app.invoke(init, config={"recursion_limit": 200})