import os
import sys
import logging
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, status, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, User, Session as DbSession, PitchFile, Feedback, Answer, CoachingFeedback
import bcrypt
try:
    import pypdf
except ImportError:
    pypdf = None
try:
    import pptx
except ImportError:
    pptx = None

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rag_engine import query_handler

# Load environment variables early
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Minimal logging to confirm key presence without exposing it
if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    logging.getLogger(__name__).info("LLM API key detected via env (.env or system).")
else:
    logging.getLogger(__name__).info("No LLM API key found; running in offline fallback mode.")

DATABASE_URL = f"sqlite:///./pitchdeck.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

app = FastAPI()
# --- Root and Static ---
@app.get("/")
def root():
    return {"status": "ok", "app": "Pitchdeck AI"}

_web_dir = ROOT.joinpath('web')
app.mount("/static", StaticFiles(directory=_web_dir, html=True), name="static")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_answer_to_db(session_id, question, answer, db):
    answer_obj = Answer(session_id=session_id, question=question, answer=answer)
    db.add(answer_obj)
    db.commit()

def save_feedback_to_db(session_id, clarity, realism, relevance, actionability, improvement, recommendations, db):
    feedback_obj = Feedback(
        session_id=session_id,
        clarity_score=str(clarity),
        realism_score=str(realism),
        relevance_score=str(relevance),
        actionability_score=str(actionability),
        improvement=improvement,
        recommendations=recommendations
    )
    db.add(feedback_obj)
    db.commit()

class FeedbackRequest(BaseModel):
    session_id: int
    transcript: List[Dict]

class FeedbackResponse(BaseModel):
    session_id: int
    clarity_score: float
    realism_score: float
    relevance_score: float
    actionability_score: float
    improvement: str
    recommendations: str

class EvaluateRequest(BaseModel):
    session_id: int

def score_transcript(transcript):
    """Heuristic, robust scoring for clarity, realism, relevance, actionability.
    Works offline without LLM; attempts LLM-based enhancement if available.
    """
    if not transcript:
        return 5.0, 5.0, 5.0, 5.0

    # Heuristics
    questions = [t.get('question', '') for t in transcript]
    answers = [t.get('answer', '') for t in transcript if t.get('answer')]
    q_text = "\n".join(questions)
    a_text = "\n".join(answers)

    def clip(x):
        return max(1.0, min(10.0, float(x)))

    # Clarity: shorter sentences, fewer filler words, punctuation balance
    total_words = len(a_text.split()) or 1
    avg_len = total_words / max(1, len(answers))
    filler = sum(a_text.lower().count(w) for w in ["like", "umm", "you know", "sort of"])
    # Penalize "I don't know" patterns lightly
    import re as _re
    unknown_hits = len(_re.findall(r"\b(i\s*do(n'?| n)?t\s*know|idk|not\s+sure|no\s+idea)\b", a_text, flags=_re.I))
    clarity = clip(10 - (avg_len / 40.0) * 3 - min(3, filler * 0.5) - min(2.0, unknown_hits * 0.6))

    # Realism: presence of tough VC cues in questions (why, how, unit economics, runway)
    realism_terms = ["why", "how", "unit", "cac", "ltv", "runway", "churn", "cohort", "defensible", "moat"]
    realism_hits = sum(q_text.lower().count(t) for t in realism_terms)
    realism = clip(4 + min(6, realism_hits * 0.8))

    # Relevance: overlap between questions and pitch-like terms in answers
    pitch_terms = ["market", "customer", "revenue", "growth", "margin", "retention", "pricing", "competition", "roadmap"]
    relevance_hits = sum(a_text.lower().count(t) for t in pitch_terms)
    relevance = clip(4 + min(6, relevance_hits * 0.4))

    # Actionability: presence of numbers, dates, metrics in answers
    digits = sum(c.isdigit() for c in a_text)
    has_pct = a_text.count('%')
    has_dollar = a_text.count('$')
    actionability = clip(3 + min(7, (digits / 50.0) + has_pct * 1.0 + has_dollar * 0.5) - min(2.0, unknown_hits * 0.5))

    # Optional: attempt LLM-based refinement if an evaluator exists
    try:
        if hasattr(query_handler, 'evaluate_transcript'):
            eval_data = query_handler.evaluate_transcript(transcript)  # expected dict
            clarity = clip(eval_data.get('clarity', clarity))
            realism = clip(eval_data.get('realism', realism))
            relevance = clip(eval_data.get('relevance', relevance))
            actionability = clip(eval_data.get('actionability', actionability))
    except Exception:
        pass

    return clarity, realism, relevance, actionability

def generate_feedback(transcript):
    """Deterministic feedback with optional LLM enhancement if available."""
    # Heuristic suggestions
    all_answers = "\n".join([t.get('answer', '') for t in transcript if t.get('answer')])
    needs_numbers = (sum(c.isdigit() for c in all_answers) < 10)
    missing_market = (all_answers.lower().count('market') < 2)
    missing_gtm = (all_answers.lower().count('go-to-market') + all_answers.lower().count('gtm')) == 0
    improvement_parts = []
    if needs_numbers:
        improvement_parts.append("Include concrete numbers (CAC, LTV, churn, conversion, runway).")
    if missing_market:
        improvement_parts.append("Reference market sizing (TAM/SAM/SOM) and evidence.")
    if missing_gtm:
        improvement_parts.append("Outline your go-to-market motion and channels.")
    improvement = " ".join(improvement_parts) or "Clarify your answers and back claims with data and sources."
    recommendations = (
        "Add a slide or appendix with KPIs (CAC, LTV, churn, cohorts). "
        "Prepare a one-minute crisp value prop and a detailed GTM funnel with assumptions."
    )

    # Optional LLM enhancement
    try:
        if hasattr(query_handler, 'improve_feedback'):
            fb = query_handler.improve_feedback(transcript)
            if isinstance(fb, dict):
                improvement = fb.get('improvement', improvement)
                recommendations = fb.get('recommendations', recommendations)
    except Exception:
        pass

    return improvement, recommendations

# --- Expert review helpers ---
def _load_transcript_from_db(db, session_id: int):
    rows = db.query(Answer).filter(Answer.session_id == session_id).order_by(Answer.created_at.asc()).all()
    transcript = []
    for r in rows:
        d = {"question": r.question or ""}
        if r.answer:
            d["answer"] = r.answer
        transcript.append(d)
    return transcript

@app.post('/api/feedback')
def api_feedback(req: FeedbackRequest, db=Depends(get_db)):
    clarity, realism, relevance, actionability = score_transcript(req.transcript)
    improvement, recommendations = generate_feedback(req.transcript)
    save_feedback_to_db(req.session_id, clarity, realism, relevance, actionability, improvement, recommendations, db)
    return FeedbackResponse(
        session_id=req.session_id,
        clarity_score=clarity,
        realism_score=realism,
        relevance_score=relevance,
        actionability_score=actionability,
        improvement=improvement,
        recommendations=recommendations
    )

@app.get('/api/sessions/{session_id}/transcript')
def api_get_transcript(session_id: int, db=Depends(get_db)):
    # Validate session
    if not db.query(DbSession).filter(DbSession.id == session_id).first():
        raise HTTPException(status_code=404, detail='Session not found')
    transcript = _load_transcript_from_db(db, session_id)
    return {"session_id": session_id, "count": len(transcript), "items": transcript}

@app.get('/api/sessions/{session_id}/feedback')
def api_get_feedback(session_id: int, db=Depends(get_db)):
    fb = db.query(Feedback).filter(Feedback.session_id == session_id).first()
    if not fb:
        raise HTTPException(status_code=404, detail='Feedback not found')
    return FeedbackResponse(
        session_id=session_id,
        clarity_score=float(fb.clarity_score or 0),
        realism_score=float(fb.realism_score or 0),
        relevance_score=float(fb.relevance_score or 0),
        actionability_score=float(fb.actionability_score or 0),
        improvement=fb.improvement or "",
        recommendations=fb.recommendations or ""
    )

@app.post('/api/evaluate')
def api_evaluate(req: EvaluateRequest, db=Depends(get_db)):
    if not db.query(DbSession).filter(DbSession.id == req.session_id).first():
        raise HTTPException(status_code=404, detail='Session not found')
    transcript = _load_transcript_from_db(db, req.session_id)
    clarity, realism, relevance, actionability = score_transcript(transcript)
    improvement, recommendations = generate_feedback(transcript)
    save_feedback_to_db(req.session_id, clarity, realism, relevance, actionability, improvement, recommendations, db)
    return FeedbackResponse(
        session_id=req.session_id,
        clarity_score=clarity,
        realism_score=realism,
        relevance_score=relevance,
        actionability_score=actionability,
        improvement=improvement,
        recommendations=recommendations
    )
class QAStartRequest(BaseModel):
    session_id: int
    persona: str
    context: Dict
    hints: Optional[bool] = True

class QAQuestionRequest(BaseModel):
    session_id: int
    history: List[Dict]
    hints: Optional[bool] = True

class QAResponse(BaseModel):
    session_id: int
    question: str
    followup: bool
    assistant_suggestion: Optional[str] = None
    sources: Optional[List[Dict]] = None

def generate_initial_question(persona, context):
    # Placeholder logic: use persona and context to ask a tough question
    weaknesses = context.get('weaknesses', [])
    persona = persona.lower()
    if persona == 'generic':
        if weaknesses:
            return f"What is your response to this weakness: {weaknesses[0]}?"
        return "What is the biggest risk in your business?"
    elif persona == 'saas guru':
        if weaknesses:
            return f"As a SaaS Guru, I noticed: {weaknesses[0]}. How will you address this in your SaaS business model?"
        return "As a SaaS Guru, what is your customer acquisition cost and how do you plan to scale ARR?"
    elif persona == 'deep tech skeptic':
        if weaknesses:
            return f"As a Deep Tech Skeptic, I see: {weaknesses[0]}. Can you justify the technical feasibility and market need?"
        return "As a Deep Tech Skeptic, what is the biggest technical risk and how do you mitigate it?"
    elif persona == 'early-stage angel':
        if weaknesses:
            return f"As an Early-Stage Angel, I noticed: {weaknesses[0]}. How will you convince early adopters?"
        return "As an Early-Stage Angel, what traction have you achieved and what is your burn rate?"
    elif persona == 'growth investor':
        if weaknesses:
            return f"As a Growth Investor, I see: {weaknesses[0]}. How will you overcome this to achieve scale?"
        return "As a Growth Investor, what is your go-to-market strategy and how do you plan to reach $10M ARR?"
    else:
        if weaknesses:
            return f"As a {persona.title()}, I noticed: {weaknesses[0]}. Can you address this?"
        return f"As a {persona.title()}, what is the biggest risk in your business?"

def generate_followup_question(persona, context, history):
    # Placeholder: probe last answer
    last_answer = history[-1]['answer'] if history else ''
    persona = persona.lower()
    # If the founder said they don't know, ask a guiding follow-up instead of generic probe
    import re as _re
    if _re.search(r"\b(i\s*do(n'?| n)?t\s*know|idk|not\s+sure|no\s+idea)\b", (last_answer or ''), flags=_re.I):
        prompts = {
            'generic': "No problem. Let's break it down: what is your current monthly revenue (even an estimate) and top 1-2 acquisition channels?",
            'saas guru': "That's okay. Start with MRR/ARR and your CAC payback. If unknown, estimate based on current spend and signups.",
            'deep tech skeptic': "Understood. What specific milestone can validate feasibility in the next 8-12 weeks (prototype, benchmark, or third-party test)?",
            'early-stage angel': "Got it. Share your next 2 milestones and the scrappy GTM steps to reach them.",
            'growth investor': "Okay. Give a ballpark of current unit economics (gross margin, CAC, NRR) and bottleneck to scaling." 
        }
        key = persona if persona in prompts else 'generic'
        return prompts[key]
    if persona == 'generic':
        return f"Based on your last answer: '{last_answer}', can you elaborate further?"
    elif persona == 'saas guru':
        return f"As a SaaS Guru, based on your last answer: '{last_answer}', what metrics will you track to ensure product-market fit?"
    elif persona == 'deep tech skeptic':
        return f"As a Deep Tech Skeptic, your last answer was: '{last_answer}'. Can you provide technical validation or third-party proof?"
    elif persona == 'early-stage angel':
        return f"As an Early-Stage Angel, you said: '{last_answer}'. What is your plan for the next 6 months to reach key milestones?"
    elif persona == 'growth investor':
        return f"As a Growth Investor, you answered: '{last_answer}'. How will you optimize for rapid scale and operational efficiency?"
    else:
        return f"As a {persona.title()}, based on your last answer: '{last_answer}', can you elaborate further?"

# (removed duplicate get_db definition)

# --- Upload & Extraction Models ---
class UploadResponse(BaseModel):
    session_id: int
    file_id: int
    filename: str

class ExtractedTextResponse(BaseModel):
    session_id: int
    slides: List[Dict]

class PersonaRequest(BaseModel):
    session_id: int
    persona: str

# --- File text extraction helpers ---
def extract_pdf_slides(filepath: str):
    if not pypdf:
        raise HTTPException(status_code=500, detail='pypdf not installed')
    reader = pypdf.PdfReader(filepath)
    slides = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ''
        slides.append({'slide': i + 1, 'text': ' '.join(text.split())})
    return slides

def extract_pptx_slides(filepath: str):
    if not pptx:
        raise HTTPException(status_code=500, detail='python-pptx not installed')
    prs = pptx.Presentation(filepath)
    slides = []
    for i, slide in enumerate(prs.slides):
        text = ''
        for shape in slide.shapes:
            if hasattr(shape, 'text'):
                text += shape.text + '\n'
        slides.append({'slide': i + 1, 'text': ' '.join(text.split())})
    return slides

# --- Upload & Extraction Endpoints ---
@app.post('/api/upload')
def api_upload(user_id: int = Form(...), file: UploadFile = File(...), db=Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail='Invalid user')
    ext = (file.filename.split('.')[-1] or '').lower()
    if ext not in ['pdf', 'pptx']:
        raise HTTPException(status_code=400, detail='File must be PDF or PPTX')
    upload_dir = ROOT.joinpath('uploads')
    upload_dir.mkdir(exist_ok=True)
    save_path = upload_dir.joinpath(file.filename)
    with save_path.open('wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    session = DbSession(user_id=user.id)
    db.add(session)
    db.commit()
    db.refresh(session)
    pitch_file = PitchFile(session_id=session.id, filename=file.filename, filetype=ext)
    db.add(pitch_file)
    db.commit()
    db.refresh(pitch_file)
    return UploadResponse(session_id=session.id, file_id=pitch_file.id, filename=file.filename)

@app.post('/api/extract_text')
def api_extract_text(session_id: int, db=Depends(get_db)):
    session = db.query(DbSession).filter(DbSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail='Session not found')
    pitch_file = db.query(PitchFile).filter(PitchFile.session_id == session_id).first()
    if not pitch_file:
        raise HTTPException(status_code=404, detail='Pitch file not found')
    filepath = ROOT.joinpath('uploads', pitch_file.filename)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail='File not found')
    if pitch_file.filetype == 'pdf':
        slides = extract_pdf_slides(str(filepath))
    elif pitch_file.filetype == 'pptx':
        slides = extract_pptx_slides(str(filepath))
    else:
        raise HTTPException(status_code=400, detail='Unsupported file type')
    return ExtractedTextResponse(session_id=session_id, slides=slides)

@app.post('/api/set_persona')
def api_set_persona(req: PersonaRequest, db=Depends(get_db)):
    session = db.query(DbSession).filter(DbSession.id == req.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail='Session not found')
    session.persona = req.persona
    db.commit()
    return {"message": "Persona set", "session_id": req.session_id, "persona": req.persona}
@app.post('/api/qa/start')
def api_qa_start(req: QAStartRequest, db=Depends(get_db)):
    # RAG: Retrieve context from FAISS and deck text
    index_dir = ROOT.joinpath('input', 'faiss_index')
    index, texts = query_handler.load_index_and_texts(str(index_dir))
    try:
        context = dict(req.context or {})
        deck_text = _get_deck_text(db, req.session_id)
        if index and texts:
            results = query_handler.search_similar("business model risk", index, texts, k=3)
            context_text = '\n'.join([r['text'] for r in results])
            context['retrieved'] = context_text
        # Use persona-based question generation with loaded prompt templates
        context_str = context.get('retrieved', deck_text or '')[:2000]
        try:
            question = query_handler.generate_question(
                persona=req.persona,
                context=context_str,
                history=''
            )
        except Exception:
            question = generate_initial_question(req.persona, context)
        # Log initial question to DB
        save_answer_to_db(req.session_id, question, '', db)
        # Optional suggested answer based on deck
        assistant_suggestion = None
        sources_payload: Optional[List[Dict]] = None
        if getattr(req, 'hints', True):
            try:
                base_results = _results_from_text_blob(deck_text)
                if index and texts:
                    base_results = (base_results or []) + query_handler.search_similar(question, index, texts, k=3)
                assistant_suggestion = query_handler.synthesize_answer(question, base_results[:5])
                # Prepare compact sources
                sources_payload = []
                for r in (base_results or [])[:3]:
                    sources_payload.append({
                        "text": (r.get('text','')[:400] if isinstance(r, dict) else str(r)[:400]),
                        "score": float(r.get('score', 0.0)) if isinstance(r, dict) else 0.0
                    })
            except Exception:
                assistant_suggestion = None
                sources_payload = None
        return QAResponse(session_id=req.session_id, question=question, followup=False, assistant_suggestion=assistant_suggestion, sources=sources_payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qa_start_failed: {type(e).__name__}: {e}")

class AnalysisRequest(BaseModel):
    session_id: int
    slides: List[Dict]

class AnalysisResponse(BaseModel):
    session_id: int
    strengths: List[str]
    weaknesses: List[str]
    missing_sections: List[str]

BUSINESS_SECTIONS = [
    "Problem", "Solution", "Market Size", "Team", "Financials", "Go-to-Market", "Ask"
]

def analyze_slides(slides):
    strengths = []
    weaknesses = []
    missing_sections = []
    found_sections = set()
    for slide in slides:
        text = slide.get('text', '').lower()
        for section in BUSINESS_SECTIONS:
            if section.lower() in text:
                found_sections.add(section)
    for section in BUSINESS_SECTIONS:
        if section not in found_sections:
            missing_sections.append(section)
    # Example: strengths/weaknesses based on presence/absence
    if "Problem" in found_sections and "Solution" in found_sections:
        strengths.append("Clear problem and solution presented.")
    if "Team" not in found_sections:
        weaknesses.append("No team slide found.")
    if "Ask" not in found_sections:
        weaknesses.append("No clear ask slide.")
    return strengths, weaknesses, missing_sections

# --- Deck context helpers ---
_DECK_TEXT_CACHE: Dict[int, str] = {}
def _get_deck_text(db, session_id: int) -> str:
    # in-process cache to avoid repeated file parsing in one run
    if session_id in _DECK_TEXT_CACHE:
        return _DECK_TEXT_CACHE[session_id]
    pitch_file = db.query(PitchFile).filter(PitchFile.session_id == session_id).first()
    if not pitch_file:
        return ""
    filepath = ROOT.joinpath('uploads', pitch_file.filename)
    if not filepath.exists():
        return ""
    try:
        if pitch_file.filetype == 'pdf':
            slides = extract_pdf_slides(str(filepath))
        elif pitch_file.filetype == 'pptx':
            slides = extract_pptx_slides(str(filepath))
        else:
            return ""
        text = "\n\n".join([s.get('text', '') for s in slides])
        _DECK_TEXT_CACHE[session_id] = text
        return text
    except Exception:
        return ""

def _results_from_text_blob(text: str, max_chunks: int = 5) -> list[dict]:
    if not text:
        return []
    parts = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    buf = []
    cur = 0
    for p in parts:
        buf.append(p)
        cur += len(p)
        if cur > 600:
            chunks.append(" ".join(buf))
            buf, cur = [], 0
    if buf:
        chunks.append(" ".join(buf))
    return [{"text": c, "score": 0.0} for c in chunks[:max_chunks]]

@app.post('/api/analyze')
def api_analyze(req: AnalysisRequest):
    strengths, weaknesses, missing_sections = analyze_slides(req.slides)
    return AnalysisResponse(
        session_id=req.session_id,
        strengths=strengths,
        weaknesses=weaknesses,
        missing_sections=missing_sections
    )

class CoachRequest(BaseModel):
    session_id: int
    question: str
    answer: str
    persona: Optional[str] = None
    tone: Optional[str] = "brief"  # brief | detailed
    length: Optional[int] = 120     # target words

class CoachResponse(BaseModel):
    score: float
    notes: str
    suggestion: str
    slide_refs: Optional[List[int]] = None

@app.post('/api/coach')
def api_coach(req: CoachRequest, db=Depends(get_db)):
    # Build minimal context from deck and persona rubric
    deck_text = _get_deck_text(db, req.session_id)
    persona = (req.persona or (db.query(DbSession).filter(DbSession.id == req.session_id).first() or DbSession()).persona) or 'generic'
    rubric = {
        'saas': 'Emphasize CAC, LTV, gross margin, churn, NRR, payback, ARR growth.',
        'deep tech skeptic': 'Emphasize feasibility, defensibility, IP, validation, roadmap risks.',
        'early-stage angel': 'Emphasize traction signals, roadmap, team strength, early GTM.',
        'growth investor': 'Emphasize scale, efficiency, unit economics at scale, ops.',
        'generic': 'Emphasize clarity, data, risks and mitigation, and next steps.'
    }
    style = f"Tone: {'investor brief' if (req.tone or 'brief')=='brief' else 'detailed explanation'}; target length ~{int(req.length or 120)} words."
    try:
        context_for_eval = f"Persona: {persona}. Rubric: {rubric.get(persona.lower(), rubric['generic'])}. {style}\n\n{deck_text}"
        data = query_handler.evaluate_answer(req.question, req.answer, context_text=context_for_eval)
        # Try to infer slide references by simple matching (inline references)
        slide_refs: List[int] = []
        try:
            slides = []
            pitch_file = db.query(PitchFile).filter(PitchFile.session_id == req.session_id).first()
            if pitch_file:
                fp = ROOT.joinpath('uploads', pitch_file.filename)
                if pitch_file.filetype == 'pdf':
                    slides = extract_pdf_slides(str(fp))
                elif pitch_file.filetype == 'pptx':
                    slides = extract_pptx_slides(str(fp))
            sug = str(data.get('suggestion', ''))
            for s in slides:
                t = (s.get('text','') or '')
                if t and any(tok in t.lower() for tok in ['cac','ltv','churn','mrr','arr','margin','runway','roadmap','ip','feasible','defensible']):
                    # naive heuristic: mark as reference if keywords align
                    slide_refs.append(int(s.get('slide', 0)))
                if len(slide_refs) >= 3:
                    break
        except Exception:
            slide_refs = []

        # Persist coaching feedback
        try:
            cf = CoachingFeedback(
                session_id=req.session_id,
                question=req.question,
                answer=req.answer,
                score=str(data.get('score', '')),
                notes=str(data.get('notes', '')),
                suggestion=str(data.get('suggestion', '')),
                slide_refs=json.dumps(slide_refs)
            )
            db.add(cf)
            db.commit()
        except Exception:
            pass

        return CoachResponse(
            score=float(data.get('score', 0.0)),
            notes=str(data.get('notes', '')),
            suggestion=str(data.get('suggestion', '')),
            slide_refs=slide_refs or None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"coach_failed: {type(e).__name__}: {e}")


@app.get('/api/coach/report/{session_id}')
def api_coach_report(session_id: int, db=Depends(get_db)):
    rows = db.query(CoachingFeedback).filter(CoachingFeedback.session_id == session_id).order_by(CoachingFeedback.created_at.asc()).all()
    items = []
    for r in rows:
        try:
            refs = json.loads(r.slide_refs or "[]")
        except Exception:
            refs = []
        items.append({
            "question": r.question,
            "answer": r.answer,
            "score": r.score,
            "notes": r.notes,
            "suggestion": r.suggestion,
            "slide_refs": refs,
            "created_at": str(r.created_at)
        })
    return {"session_id": session_id, "count": len(items), "items": items}

@app.post('/api/qa/next')
def api_qa_next(req: QAQuestionRequest, db=Depends(get_db)):
    persona = req.history[0].get('persona', 'VC') if req.history else 'VC'
    context = req.history[0].get('context', {}) if req.history else {}
    # RAG: Retrieve context for followup
    index_dir = ROOT.joinpath('input', 'faiss_index')
    index, texts = query_handler.load_index_and_texts(str(index_dir))
    try:
        session = db.query(DbSession).filter(DbSession.id == req.session_id).first()
        persona = (session.persona if session else None) or 'generic_baseline'
        context = {}
        last_answer = req.history[-1]['answer'] if req.history else ''
        deck_text = _get_deck_text(db, req.session_id)
        if index and texts:
            results = query_handler.search_similar(last_answer or "follow up", index, texts, k=4)
            context_text = '\n'.join([r['text'] for r in results])
            context['retrieved'] = context_text
        # Build conversation history string
        history_str = '\n'.join([
            f"Q: {h.get('question', '')}\nA: {h.get('answer', '')}"
            for h in req.history[-3:]  # Last 3 Q&A pairs
        ])
        context_str = context.get('retrieved', deck_text or '')[:2000]
        try:
            question = query_handler.generate_question(
                persona=persona,
                context=context_str,
                history=history_str
            )
        except Exception:
            question = generate_followup_question(persona, context, req.history)
        # Save last answer to DB if present
        if req.history and 'answer' in req.history[-1]:
            save_answer_to_db(req.session_id, req.history[-1].get('question', ''), req.history[-1]['answer'], db)
        # Log followup question to DB
        save_answer_to_db(req.session_id, question, '', db)
        # Suggested answer
        assistant_suggestion = None
        sources_payload: Optional[List[Dict]] = None
        if getattr(req, 'hints', True):
            try:
                base_results = _results_from_text_blob(deck_text)
                if index and texts:
                    base_results = (base_results or []) + query_handler.search_similar(question, index, texts, k=3)
                assistant_suggestion = query_handler.synthesize_answer(question, base_results[:5])
                sources_payload = []
                for r in (base_results or [])[:3]:
                    sources_payload.append({
                        "text": (r.get('text','')[:400] if isinstance(r, dict) else str(r)[:400]),
                        "score": float(r.get('score', 0.0)) if isinstance(r, dict) else 0.0
                    })
            except Exception:
                assistant_suggestion = None
                sources_payload = None
        return QAResponse(session_id=req.session_id, question=question, followup=True, assistant_suggestion=assistant_suggestion, sources=sources_payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qa_next_failed: {type(e).__name__}: {e}")
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

@app.post('/api/signup')
def api_signup(req: SignupRequest, db=Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail='Email already registered')
    hashed = bcrypt.hashpw(req.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    user = User(name=req.name, email=req.email, password_hash=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "Signup successful"}

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post('/api/login')
def api_login(req: LoginRequest, db=Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not bcrypt.checkpw(req.password.encode('utf-8'), user.password_hash.encode('utf-8')):
        raise HTTPException(status_code=401, detail='Invalid credentials')
    # For demo: return user id (in production, use JWT/session)
    return {"message": "Login successful", "user_id": user.id, "name": user.name}


class QueryRequest(BaseModel):
    query: str
    k: int = 5


@app.post('/api/search')
def api_search(req: QueryRequest):
    index_dir = ROOT.joinpath('input', 'faiss_index')
    index, texts = query_handler.load_index_and_texts(str(index_dir))
    if index is None or not texts:
        raise HTTPException(status_code=500, detail='Index not found; run embedding first')
    results = query_handler.search_similar(req.query, index, texts, k=req.k)
    return {'query': req.query, 'results': results}


@app.post('/api/synthesize')
def api_synthesize(req: QueryRequest):
    index_dir = ROOT.joinpath('input', 'faiss_index')
    index, texts = query_handler.load_index_and_texts(str(index_dir))
    if index is None or not texts:
        raise HTTPException(status_code=500, detail='Index not found; run embedding first')
    results = query_handler.search_similar(req.query, index, texts, k=req.k)
    answer = query_handler.synthesize_answer(req.query, results)
    return {'query': req.query, 'answer': answer, 'sources': results}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='127.0.0.1', port=8000, reload=True)
