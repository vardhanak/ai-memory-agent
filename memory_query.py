from typing import Dict, Any, List
from memory_store import query_memories, generate_from_llm, store_memory_local

PROMPT_TEMPLATE = """You are an assistant that answers user questions concisely and correctly.
Use the following recovered personal memories to inform your answer.
Ignore any memories that are not clearly related to the user’s current question.
Never force irrelevant details into the answer.
Do NOT invent facts about the user beyond what's given in memories.

Memories:
{memories_block}

User question:
{question}

Assistant:"""

def build_prompt_from_memories(question: str, retrieved: List[Dict[str, Any]]) -> str:
    if not retrieved:
        memories_block = "(no stored memories found)"
    else:
        parts = []
        for r in retrieved:
            t = r.get("type", "personalization")
            text = r.get("text", "")
            parts.append(f"- {text} (type={t}, idx={r.get('index')}, dist={r.get('distance'):.4f})")
        memories_block = "\n".join(parts)
    prompt = PROMPT_TEMPLATE.format(memories_block=memories_block, question=question)
    return prompt

def query_with_memory(question: str, k: int = 5) -> Dict[str, Any]:
    retrieved = query_memories(question, k=k)
    prompt = build_prompt_from_memories(question, retrieved)
    print("=== PROMPT SENT TO LLM ===")
    print(prompt[:2000])  # print first 2000 chars (avoid huge output)
    print("=== END PROMPT ===")
    gen_resp = generate_from_llm(prompt, max_tokens=256, temperature=0.0)
    answer = gen_resp.get("text", "")
    return {"answer": answer, "retrieved": retrieved, "prompt": prompt, "raw": gen_resp.get("raw")}


def query_without_memory(question: str) -> Dict[str, Any]:

    prompt = """You are an assistant that answers user questions concisely and correctly.
Do NOT use any personal memories or stored information about the user — answer based only on general knowledge.

User question:
{question}

Assistant:""".format(question=question)

    print("=== PROMPT SENT TO LLM (NO MEMORY) ===")
    print(prompt[:2000])
    print("=== END PROMPT ===")
    gen_resp = generate_from_llm(prompt, max_tokens=256, temperature=0.0)
    answer = gen_resp.get("text", "")
    return {"answer": answer, "raw": gen_resp.get("raw"), "prompt": prompt}
