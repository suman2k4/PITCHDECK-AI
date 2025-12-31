import faiss
import numpy as np
from typing import Any
from functools import lru_cache
import hashlib
import re
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

# Caching for embeddings and LLM responses to reduce API calls
_EMBED_CACHE = {}
_LLM_RESPONSE_CACHE = {}


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
    # Check cache first to save API calls
    cache_key = hashlib.md5(query.encode()).hexdigest()
    if cache_key in _EMBED_CACHE:
        logging.info(f"Using cached embedding for query: {query[:50]}...")
        return _EMBED_CACHE[cache_key]
    
    if not FORCE_OFFLINE and API_KEY and GoogleGenerativeAIEmbeddings is not None:
        try:
            emb = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)
            v = emb.embed_query(query)
            result = np.array(v).astype('float32')
            # Cache the result
            _EMBED_CACHE[cache_key] = result
            # Limit cache size to prevent memory issues
            if len(_EMBED_CACHE) > 500:
                # Remove oldest entry
                _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)))
            return result
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
    
    # Check cache to save API calls
    cache_key = hashlib.md5(f"{persona}:{context[:500]}:{history[:500]}".encode()).hexdigest()
    if cache_key in _LLM_RESPONSE_CACHE:
        logging.info(f"Using cached question for persona: {persona}")
        return _LLM_RESPONSE_CACHE[cache_key]
    
    if not FORCE_OFFLINE and API_KEY and genai is not None:
        try:
            model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')
            model = genai.GenerativeModel(model_name, generation_config={"temperature": float(os.getenv("LLM_TEMPERATURE", "0.4"))})
            resp = model.generate_content(prompt)
            if hasattr(resp, 'text') and resp.text:
                question = resp.text.strip()
                # Cache successful response
                _LLM_RESPONSE_CACHE[cache_key] = question
                if len(_LLM_RESPONSE_CACHE) > 200:
                    _LLM_RESPONSE_CACHE.pop(next(iter(_LLM_RESPONSE_CACHE)))
                return question
            return str(resp).strip()
        except Exception as e:
            logging.warning(f"LLM call for question generation failed: {e}. Using intelligent fallback.")
    
    # Intelligent fallback based on conversation history and persona
    return _generate_intelligent_fallback_question(persona, context, history)


def _generate_intelligent_fallback_question(persona: str, context: str, history: str) -> str:
    """Generate contextual questions based on conversation history when API is unavailable."""
    
    # Analyze the conversation history
    history_lower = history.lower()
    
    # Extract last answer to build on
    last_answer_match = re.search(r'A:\s*(.+?)(?=\nQ:|$)', history, re.DOTALL)
    last_answer = last_answer_match.group(1).strip() if last_answer_match else ""
    last_answer_lower = last_answer.lower()
    
    # Count how many questions have been asked
    question_count = history.count('Q:')
    
    # Persona-specific intelligent question sequences
    persona_questions = {
        'saas_guru': [
            "Let's start with the basics - what's your current MRR or ARR, and what's driving that growth?",
            "How do you acquire customers? Walk me through your CAC and how it compares to LTV.",
            "Tell me about your churn rate. What percentage of customers leave each month, and why?",
            "What's your pricing strategy? Have you tested different pricing tiers or models?",
            "How does your product create network effects or switching costs that lock in customers?"
        ],
        'deep_tech_skeptic': [
            "What's the core technical innovation here? Convince me this isn't just an incremental improvement.",
            "Who are the technical experts or institutions validating your approach? Any partnerships?",
            "What are the biggest technical risks that could derail this project, and how are you mitigating them?",
            "How defensible is your technology? Do you have patents filed or trade secrets?",
            "What regulatory hurdles do you face, and what's your realistic timeline to market?"
        ],
        'early_stage_angel': [
            "What traction have you achieved so far? Even if it's early, what validates that people want this?",
            "How much runway do you have, and what specific milestones will you hit before needing more capital?",
            "Why is your team uniquely positioned to win? What's your unfair advantage?",
            "What keeps you up at night? What's the biggest risk to your business right now?",
            "If I gave you an extra $50K today, what would you spend it on to move the needle most?"
        ],
        'growth_investor': [
            "Walk me through your go-to-market strategy. What channels are working best?",
            "How do you plan to scale from where you are to $10M ARR? What needs to change?",
            "What are your unit economics at scale? When do you become profitable?",
            "Who are your top 3 competitors, and why will you win market share from them?",
            "What operational bottlenecks will you hit as you scale, and how are you preparing?"
        ],
        'generic_baseline': [
            "Tell me about your target market. Who exactly are your ideal customers?",
            "What specific problem are you solving, and how painful is it for your customers?",
            "How do you make money? Walk me through your business model.",
            "What's your competitive advantage? Why will you win this market?",
            "What are the key risks to your business, and how are you managing them?"
        ]
    }
    
    # Get persona-specific questions
    questions = persona_questions.get(persona, persona_questions['generic_baseline'])
    
    # Adaptive follow-up based on last answer content
    if "don't know" in last_answer_lower or "not sure" in last_answer_lower or "no idea" in last_answer_lower:
        return "That's totally fine - let's break it down. What data or metrics ARE you tracking closely right now, and what do they tell you?"
    
    elif "revenue" in last_answer_lower or "mrr" in last_answer_lower or "arr" in last_answer_lower:
        if "margin" not in history_lower:
            return "Good! Now tell me about your margins. What are your gross margins, and how do they improve as you scale?"
        else:
            return "Great context on revenue. How predictable is this? What percentage of revenue is recurring vs. one-time?"
    
    elif "customer" in last_answer_lower or "user" in last_answer_lower:
        if "retention" not in history_lower and "churn" not in history_lower:
            return "Interesting. Tell me about retention - what percentage of customers are still with you after 6 months?"
        else:
            return "Got it. What do your best customers have in common? Any patterns in who succeeds with your product?"
    
    elif "team" in last_answer_lower or "founder" in last_answer_lower:
        return "Your team sounds solid. What key roles are you still hiring for, and why are those critical right now?"
    
    elif "market" in last_answer_lower or "tam" in last_answer_lower:
        if "competition" not in history_lower:
            return "Market sizing is helpful. Now tell me about competition - who else is going after this market and how are you different?"
        else:
            return "Good market analysis. How much of that addressable market can you realistically capture in 3 years?"
    
    elif "product" in last_answer_lower or "feature" in last_answer_lower:
        return "Tell me about your roadmap. What's the most important feature you're building next, and why?"
    
    elif "competitor" in last_answer_lower or "competition" in last_answer_lower:
        return "Understood. What would make a customer choose you over the incumbent? What's your wedge into the market?"
    
    elif "fundraise" in last_answer_lower or "capital" in last_answer_lower or "investment" in last_answer_lower:
        return "Let's talk about deployment. If you raise this round, what are your milestones for the next 12-18 months?"
    
    elif len(last_answer) < 20:
        return "Can you elaborate on that? I'd love more details to understand your thinking."
    
    # Default: use question sequence based on conversation depth
    question_index = min(question_count, len(questions) - 1)
    return questions[question_index]


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
