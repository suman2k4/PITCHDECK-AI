import faiss
import numpy as np
from typing import Any
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # optional
except Exception:
    GoogleGenerativeAIEmbeddings = None  # type: ignore
try:
    import google.generativeai as genai  # optional
except Exception:
    genai = None  # type: ignore
from dotenv import load_dotenv
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()
# Prefer GEMINI_API_KEY, fall back to GOOGLE_API_KEY for compatibility
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
FORCE_OFFLINE = os.getenv("RAG_FORCE_OFFLINE") == "1" or os.getenv("RAG_MODE", "online").lower() == "offline"
if not API_KEY:
    logging.info("No GEMINI_API_KEY/GOOGLE_API_KEY set. Using offline fallbacks for LLM/embeddings.")
else:
    if genai is not None:
        try:
            genai.configure(api_key=API_KEY)
        except Exception:
            logging.warning("Could not configure google.generativeai client. LLM calls may fail.")
    else:
        logging.info("google.generativeai not installed; LLM calls will use fallback.")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
EMBEDDING_MODEL = os.getenv('GEMINI_EMBEDDING_MODEL', 'models/embedding-001')

# Simple in-process cache for FAISS index/texts to avoid disk reads per request
_CACHED_INDEX_PATH: str | None = None
_CACHED_INDEX: Any | None = None
_CACHED_TEXTS: list[str] | None = None


def load_index_and_texts(index_dir: str):
    global _CACHED_INDEX_PATH, _CACHED_INDEX, _CACHED_TEXTS
    if _CACHED_INDEX_PATH == index_dir and _CACHED_INDEX is not None and _CACHED_TEXTS is not None:
        return _CACHED_INDEX, _CACHED_TEXTS
    idx = Path(index_dir)
    if not idx.exists():
        logging.error(f"Index directory not found: {index_dir}")
        return None, None
    try:
        index_path = idx.joinpath('index.faiss')
        if not index_path.exists():
            logging.error(f"FAISS index file missing: {index_path}")
            return None, None
        index = faiss.read_index(str(index_path))
    except Exception as e:
        logging.error(f"Failed to read FAISS index: {e}")
        return None, None
    texts = []
    txt_file = idx.joinpath('index.txt')
    if txt_file.exists():
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                parts = f.read().split('\n---\n')
                texts = [p.strip() for p in parts if p.strip()]
        except Exception as e:
            logging.warning(f"Failed to read index.txt: {e}")
            texts = []
    _CACHED_INDEX_PATH = index_dir
    _CACHED_INDEX = index
    _CACHED_TEXTS = texts
    return index, texts


def embed_query(query: str):
    if not FORCE_OFFLINE and API_KEY and GoogleGenerativeAIEmbeddings is not None:
        try:
            emb = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)
            v = emb.embed_query(query)
            return np.array(v).astype('float32')
        except Exception as e:
            logging.warning(f"Embedding API failed, using offline fallback: {e}")
    # Dry-run fallback: deterministic pseudo-random vector so searches work without API
    rng = np.random.default_rng(12345)
    return rng.normal(size=(1536,)).astype('float32')


def search_similar(query: str, index, texts: list[str], k: int = 4):
    qvec = embed_query(query)
    # Ensure query vector matches index dimension to avoid FAISS errors
    dim = getattr(index, 'd', None)
    if dim is not None and qvec.shape[0] != dim:
        if qvec.shape[0] > dim:
            qvec = qvec[:dim]
        else:
            pad = np.zeros((dim - qvec.shape[0],), dtype=qvec.dtype)
            qvec = np.concatenate([qvec, pad], axis=0)
    D, I = index.search(np.expand_dims(qvec, axis=0), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        text = texts[idx] if idx < len(texts) else ""
        results.append({"text": text, "score": float(score)})
    return results


def synthesize_answer(query: str, results: list[dict]) -> str:
    """Synthesize a short answer from the retrieved results.

    If GEMINI_API_KEY is present, try calling the Generative AI API. Otherwise fall back
    to a simple heuristic summary (first 2 sentences from the top results).
    """
    prompt = (
        "You are an assistant that answers questions using the provided context. "
        "Context:\n" + "\n---\n".join([r['text'] for r in results[:5]]) + "\n\n"
        f"Question: {query}\nAnswer concisely:"
    )

    if not FORCE_OFFLINE and API_KEY and genai is not None:
        try:
            # Use GenerativeModel for text generation (google-generativeai >= 0.3)
            model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')
            model = genai.GenerativeModel(model_name, generation_config={"temperature": float(os.getenv("LLM_TEMPERATURE", "0.4"))})
            resp = model.generate_content(prompt)
            if hasattr(resp, 'text') and resp.text:
                return resp.text
            # Fallback if SDK returns a different structure
            return str(resp)
        except Exception as e:
            logging.warning(f"LLM call failed: {e}. Falling back to heuristic summary.")

    # Heuristic fallback: take first 2 sentences from top 3 results
    import re
    sentences = []
    for r in results[:3]:
        text = r.get('text', '')
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        for s in parts:
            if s and len(sentences) < 4:
                sentences.append(s.strip())
            if len(sentences) >= 4:
                break
        if len(sentences) >= 4:
            break
    return " ".join(sentences)


def load_persona_prompt(persona: str) -> str:
    """Load the appropriate persona prompt template from prompts/ folder.
    
    Args:
        persona: One of 'saas_guru', 'deep_tech_skeptic', 'early_stage_angel', 
                 'growth_investor', or 'generic_baseline'
    
    Returns:
        The prompt template string with {context} and {history} placeholders
    """
    here = Path(__file__).resolve().parent.parent
    prompts_dir = here / "prompts"
    
    persona_files = {
        "saas_guru": "saas_guru_prompt.txt",
        "deep_tech_skeptic": "deep_tech_skeptic_prompt.txt",
        "early_stage_angel": "early_stage_angel_prompt.txt",
        "growth_investor": "growth_investor_prompt.txt",
        "generic_baseline": "generic_baseline_prompt.txt"
    }
    
    filename = persona_files.get(persona, "generic_baseline_prompt.txt")
    prompt_path = prompts_dir / filename
    
    try:
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception as e:
        logging.warning(f"Failed to load persona prompt {filename}: {e}")
    
    # Fallback generic prompt
    return (
        "You are an experienced VC. Given the pitch deck context: {context}\n\n"
        "And the conversation history so far: {history}\n\n"
        "Generate a single, challenging follow-up question that probes deeper into the business model."
    )


def generate_question(persona: str = "generic_baseline", context: str = "", history: str = "") -> str:
    """Generate a question using Gemini based on the persona and context.

    Args:
        persona: The VC persona to use for question generation
        context: The pitch deck context from RAG retrieval
        history: Conversation history so far
    
    Returns:
        Generated question string
    """
    # Load and format persona-specific prompt
    prompt_template = load_persona_prompt(persona)
    prompt = prompt_template.format(context=context[:2000], history=history[:1000])
    
    if not FORCE_OFFLINE and API_KEY and genai is not None:
        try:
            model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')
            model = genai.GenerativeModel(model_name, generation_config={"temperature": float(os.getenv("LLM_TEMPERATURE", "0.4"))})
            resp = model.generate_content(prompt)
            if hasattr(resp, 'text') and resp.text:
                return resp.text.strip()
            return str(resp).strip()
        except Exception as e:
            logging.warning(f"LLM call for question generation failed: {e}. Falling back to generic question.")
    # Fallback: a generic challenging question
    return "What is the biggest risk to your business model, and how do you plan to mitigate it?"


def evaluate_answer(question: str, answer: str, context_text: str = "") -> dict:
    """Evaluate user's answer and suggest an improved version.

    Returns: { score: 0-10, notes: str, suggestion: str }
    Uses LLM when available; otherwise a heuristic fallback.
    """
    # Heuristic baseline
    def heuristic_eval(q: str, a: str) -> dict:
        a_l = a.lower()
        score = 5.0
        notes = []
        if any(tok in a_l for tok in ["$", "%", "mrr", "arr", "cac", "ltv", "churn", "runway", "margin"]):
            score += 2
        if len(a.split()) > 30:
            score += 1
        if any(k in a_l for k in ["because", "therefore", "so that", "we plan"]):
            score += 1
        score = max(1.0, min(10.0, score))
        suggestion = a
        if score < 7.5:
            suggestion = (
                "Be specific and quantitative. Include 1-2 metrics (e.g., CAC, LTV, churn, growth), "
                "the time horizon, and a mitigation/next step."
            )
        return {"score": score, "notes": "; ".join(notes) if notes else "", "suggestion": suggestion}

    if FORCE_OFFLINE or not (API_KEY and genai is not None):
        return heuristic_eval(question, answer)

    try:
        model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')
        model = genai.GenerativeModel(model_name, generation_config={"temperature": float(os.getenv("LLM_TEMPERATURE", "0.4"))})
        prompt = (
            "You are a precise VC coach. Evaluate the founder's answer to the question and suggest a better one.\n"
            "Return JSON with keys: score (0-10), notes (string), suggestion (string).\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Context (optional): {context_text[:1500]}\n"
        )
        resp = model.generate_content(prompt)
        txt = getattr(resp, 'text', '') or str(resp)
        # Try to extract JSON
        import json as _json, re as _re
        m = _re.search(r"\{[\s\S]*\}", txt)
        if m:
            try:
                data = _json.loads(m.group(0))
                # clip and defaults
                data['score'] = float(max(0, min(10, data.get('score', 0))))
                data['notes'] = str(data.get('notes', ''))
                data['suggestion'] = str(data.get('suggestion', ''))
                return data
            except Exception:
                pass
        # Fallback: wrap as suggestion
        return {"score": 7.0, "notes": "LLM eval", "suggestion": txt[:600]}
    except Exception:
        return heuristic_eval(question, answer)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    index_dir = here.parent.joinpath("input", "faiss_index")

    index, texts = load_index_and_texts(str(index_dir))
    if index is None or texts is None or len(texts) == 0:
        logging.error("Index or texts could not be loaded. Run embedding first.")
    else:
        try:
            while True:
                q = input("Enter a query: ")
                if not q.strip():
                    continue
                results = search_similar(q, index, texts, k=5)
                for i, r in enumerate(results, 1):
                    print(f"\nResult {i} (score={r['score']}):\n{r['text'][:500]}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
