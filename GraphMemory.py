import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from pathlib import Path
import ollama

INDEX_PATH = "memories.index"
META_PATH = "memories_meta.pkl"
EMBED_DIM: Optional[int] = None
FAISS_INDEX_FACTORY = None 

EMBED_MODEL = "nomic-embed-text"  
LLM_MODEL = "llama3.2"           

def get_embedding(text: str) -> np.ndarray:
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    emb = np.asarray(resp["embedding"], dtype=np.float32)
    return emb

def generate_from_llm(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> Dict[str, Any]:
    options = {
        "temperature": float(temperature),
        "num_predict": int(max_tokens),
    }
    resp = ollama.generate(model=LLM_MODEL, prompt=prompt, options=options)

    def _extract_text(r):
        if isinstance(r, dict):
            for k in ("text", "response", "generated", "output", "content"):
                v = r.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
                if isinstance(v, (list, tuple)) and v:
                    first = v[0]
                    if isinstance(first, dict):
                        for k2 in ("content", "text"):
                            if k2 in first and isinstance(first[k2], str) and first[k2].strip():
                                return first[k2].strip()
                    if isinstance(first, str) and first.strip():
                        return first.strip()
            if "choices" in r and isinstance(r["choices"], (list, tuple)) and r["choices"]:
                c = r["choices"][0]
                if isinstance(c, dict):
                    for k in ("text", "message", "content"):
                        if k in c and isinstance(c[k], str) and c[k].strip():
                            return c[k].strip()
                if isinstance(c, str) and c.strip():
                    return c.strip()
            return None

        else:
            for attr in ("text", "response", "generated", "output", "content", "result"):
                if hasattr(r, attr):
                    val = getattr(r, attr)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
                    if isinstance(val, (list, tuple)) and val:
                        first = val[0]
                        if isinstance(first, str) and first.strip():
                            return first.strip()
                        if isinstance(first, dict):
                            for k2 in ("content", "text"):
                                if k2 in first and isinstance(first[k2], str) and first[k2].strip():
                                    return first[k2].strip()
            if hasattr(r, "response"):
                rr = getattr(r, "response")
                if isinstance(rr, str) and rr.strip():
                    return rr.strip()
                nested = _extract_text(rr)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
            return None

    text = _extract_text(resp)
    if not text:
        text = str(resp)

    return {"text": text, "raw": resp}


def _ensure_dirs():
    Path(".").mkdir(parents=True, exist_ok=True)

def _save_meta(meta_list: List[Dict[str, Any]]):
    with open(META_PATH, "wb") as f:
        pickle.dump(meta_list, f)

def _load_meta() -> List[Dict[str, Any]]:
    if not os.path.exists(META_PATH):
        return []
    with open(META_PATH, "rb") as f:
        return pickle.load(f)

def build_index_from_texts(texts: List[str], metas: Optional[List[Dict[str, Any]]] = None):
    _ensure_dirs()
    if metas is None:
        metas = [{} for _ in texts]
    assert len(texts) == len(metas)
    vecs = []
    for t in texts:
        emb = get_embedding(t)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        vecs.append(emb)
    vecs = np.vstack(vecs).astype(np.float32)
    n, dim = vecs.shape
    global EMBED_DIM
    EMBED_DIM = dim
    if FAISS_INDEX_FACTORY:
        index = faiss.index_factory(dim, FAISS_INDEX_FACTORY)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(vecs)
    faiss.write_index(index, INDEX_PATH)
    meta_list = []
    for text, meta in zip(texts, metas):
        item = {"text": text}
        item.update(meta or {})
        meta_list.append(item)
    _save_meta(meta_list)
    print(f"[build_index] Saved index with {index.ntotal} vectors (dim={dim}).")

def load_index_and_meta():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Index or metadata missing. Run build_index_from_texts first.")
    index = faiss.read_index(INDEX_PATH)
    meta_list = _load_meta()
    return index, meta_list

def add_memory_to_index(text: str, meta: Optional[Dict[str, Any]] = None):
    meta = dict(meta or {})
    meta.setdefault("type", "personalization")
    emb = np.asarray(get_embedding(text), dtype=np.float32).reshape(1, -1)
    global EMBED_DIM
    if EMBED_DIM is None:
        EMBED_DIM = emb.shape[1]

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        if index.d != emb.shape[1]:
            raise ValueError(f"Embedding dim {emb.shape[1]} doesn't match index dim {index.d}")
        index.add(emb)
    else:
        index = faiss.IndexFlatL2(emb.shape[1])
        index.add(emb)
    faiss.write_index(index, INDEX_PATH)

    metas = _load_meta()
    metas.append({"text": text, **meta})
    _save_meta(metas)
    print(f"[add_memory] Stored memory. index.ntotal={index.ntotal}, total_meta={len(metas)}")

def store_memory_local(text: str, meta: Optional[Dict[str, Any]] = None):
    meta = dict(meta or {})
    meta.setdefault("type", "personalization")
    add_memory_to_index(text, meta)

def query_memories(query_text: str, k: int = 5, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return []

    index, metas = load_index_and_meta()
    q_emb = np.asarray(get_embedding(query_text), dtype=np.float32).reshape(1, -1)

    if q_emb.shape[1] != index.d:
        raise ValueError(f"Query embedding dim {q_emb.shape[1]} != index dim {index.d}. "
                         "Rebuild index or use matching embedding model.")

    D, I = index.search(q_emb, k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metas):
            continue
        if threshold is not None:
            if isinstance(index, faiss.IndexFlatL2):
                if dist > threshold:
                    continue
            else:
                if dist < threshold:
                    continue
        m = metas[idx].copy()
        m.update({"distance": float(dist), "index": int(idx)})
        results.append(m)
    return results

def migrate_profile_to_personalization():
    metas = _load_meta()
    changed = False
    for item in metas:
        if isinstance(item, dict) and item.get("type") == "profile":
            item["type"] = "personalization"
            changed = True
    if changed:
        _save_meta(metas)
        print("[migrate] Updated profile -> personalization in metadata.")
    else:
        print("[migrate] No profile entries found.")

def show_index_summary():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        print("[show_index_summary] No index or meta found.")
        return
    index, metas = load_index_and_meta()
    print(f"[show_index_summary] index.ntotal={index.ntotal}, total_meta={len(metas)}")
    for i, m in enumerate(metas[:10]):
        print(i, m)
