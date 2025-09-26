import os
from GraphMemory import store_memory_local, build_index_from_texts, query_memories, show_index_summary, migrate_profile_to_personalization
from langgraph_flow import query_with_memory, query_without_memory
import gradio as gr

DEMO_MEMORIES = [
    ("I live in Hyderabad.", {"type": "personalization"}),
    ("My favorite cuisine is Italian.", {"type": "personalization"}),
    ("I'm vegetarian.", {"type": "personalization"}),
    ("I like spicy food.", {"type": "personalization"}),
    ("I love riding cars and bikes.", {"type": "personalization"}),
    ("I am preparing for AI engineer role.", {"type": "personalization"})
]

def ensure_demo_index():
    try:
        metas = None
        if hasattr(__import__("GraphMemory"), "_load_meta"):
            metas = __import__("GraphMemory")._load_meta()
    except Exception:
        metas = None
    if not metas and build_index_from_texts:
        texts = [t for t, m in DEMO_MEMORIES]
        meta_objs = [m for t, m in DEMO_MEMORIES]
        print("[app] Building demo index from hardcoded DEMO_MEMORIES...")
        build_index_from_texts(texts, meta_objs)
    else:
        print("[app] Index/meta already exist or cannot check; skipping demo build.")
        try:
            show_index_summary()
        except Exception:
            pass

def ask_callback(query: str):
    if not query or query.strip() == "":
        return "Please type a question.", "[]"
    res = query_with_memory(query, k=5)
    return res["answer"], str(res["retrieved"])


def save_memory_callback(text: str, meta_type: str):
    if not text or text.strip() == "":
        return "No text provided."
    meta = {"type": meta_type or "personalization"}
    store_memory_local(text, meta)
    return f"Saved memory: '{text}' (type={meta['type']})"

def migrate_callback():
    try:
        migrate_profile_to_personalization()
        return "Migration complete (if any 'profile' types existed)."
    except Exception:
        return "Migration function not available."

def ask_callback(query: str):
    if not query or query.strip() == "":
        return "Please type a question."
    res = query_with_memory(query, k=5)
    return res["answer"]

def ask_both(query: str):
    if not query or not query.strip():
        return "Please type a question.", "Please type a question."
    with_mem = query_with_memory(query, k=5)
    without_mem = query_without_memory(query)
    return with_mem.get("answer", ""), without_mem.get("answer", "")

def start_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Personal Memory Assistant — Compare (with vs without personalization)")
        inp = gr.Textbox(label="Ask", placeholder="Type your question here...", lines=2)
        ask_btn = gr.Button("Ask")

        with gr.Row():
            out_with = gr.Textbox(label="Answer (with memory)", lines=8)
            out_without = gr.Textbox(label="Answer (without memory)", lines=8)

        ask_btn.click(fn=ask_both, inputs=inp, outputs=[out_with, out_without])

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    start_ui()