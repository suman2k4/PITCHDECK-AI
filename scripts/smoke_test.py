import os
import time
import json
import requests

BASE = os.environ.get("BASE_URL", "http://localhost:8000")

def check_server():
    try:
        # hit root and docs as a simple check
        r = requests.get(f"{BASE}/", timeout=10)
        if r.status_code >= 500:
            raise SystemExit(f"Server error at {BASE}/ -> {r.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        raise SystemExit(
            "Cannot connect to API. Start the server first or set BASE_URL.\n"
            f"Tried: {BASE}\nError: {e}"
        )

def post_json(path, payload=None, files=None, data=None, expect=200):
    url = f"{BASE}{path}"
    try:
        if files is not None:
            resp = requests.post(url, data=(data or {}), files=files, timeout=60)
        else:
            resp = requests.post(url, json=(payload or {}), timeout=30)
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"POST {path} failed to connect: {e}")
    if resp.status_code != expect:
        raise SystemExit(f"POST {path} -> {resp.status_code} {resp.text}")
    return resp.json() if resp.text else {}


def run():
    print(f"Using BASE_URL={BASE}")
    print("[0/7] Checking server availability...")
    check_server()
    print("   -> server reachable")
    print("[1/7] Signup (ignore if exists)...")
    email = f"demo_{int(time.time())}@example.com"
    password = "test1234"
    name = "Demo User"
    try:
        post_json("/api/signup", {"name": name, "email": email, "password": password})
    except SystemExit as e:
        # allow already-exists error and continue
        if "400" not in str(e):
            raise
    print("   -> ok")

    print("[2/7] Login...")
    data = post_json("/api/login", {"email": email, "password": password})
    user_id = data.get("user_id")
    assert user_id, "missing user_id"
    print("   -> user_id", user_id)

    print("[3/7] Upload dummy PDF...")
    # Create a tiny in-memory PDF-like bytes. If server strictly parses PDF, replace with a real small PDF file path.
    # Here we fallback to a .pptx path if provided via env DUMMY_FILE.
    dummy_path = os.environ.get("DUMMY_FILE")
    if dummy_path and os.path.exists(dummy_path):
        filename = os.path.basename(dummy_path)
        with open(dummy_path, "rb") as f:
            files = {"file": (filename, f, "application/octet-stream")}
            up = post_json("/api/upload", files=files, data={"user_id": str(user_id)})
    else:
        # minimal valid PDF header to pass basic readers; may still fail with strict parsers
        pdf_bytes = b"%PDF-1.1\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"
    files = {"file": ("dummy.pdf", pdf_bytes, "application/pdf")}
    up = post_json("/api/upload", files=files, data={"user_id": str(user_id)})
    session_id = up.get("session_id")
    assert session_id, "missing session_id"
    print("   -> session_id", session_id)

    print("[4/7] Extract text...")
    try:
        ext = requests.post(f"{BASE}/api/extract_text", params={"session_id": session_id}, timeout=30)
    except requests.exceptions.RequestException as e:
        print("   !! extract request failed:", e)
        ext = None
    if not ext or ext.status_code != 200:
        # OK for dummy
        print("   !! extract failed (ok for dummy)", getattr(ext, 'status_code', 'n/a'), getattr(ext, 'text', ''))
    else:
        print("   -> extracted slides", len(ext.json().get("slides", [])))

    print("[5/7] Set persona...")
    sp = post_json("/api/set_persona", {"session_id": session_id, "persona": "saas"})
    print("   -> persona set")

    print("[6/7] Start QA...")
    qa0 = post_json("/api/qa/start", {"session_id": session_id, "persona": "saas", "context": {}})
    q1 = qa0.get("question")
    assert q1, "missing first question"
    print("   -> Q1:", q1[:80], "...")

    print("[7/7] Next QA + Feedback...")
    qa1 = post_json("/api/qa/next", {"session_id": session_id, "history": [{"question": q1, "answer": "We project $1M ARR in 18 months."}]})
    q2 = qa1.get("question")
    print("   -> Q2:", (q2 or "")[0:80], "...")

    fb = post_json(
        "/api/feedback",
        {
            "session_id": session_id,
            "transcript": [
                {"question": q1, "answer": "We project $1M ARR in 18 months."},
                {"question": q2 or "", "answer": "Our CAC is $250"},
            ],
        },
    )
    # The API returns clarity_score and improvement as string; handle both new/old shapes gracefully
    clarity_score = fb.get("clarity_score") if isinstance(fb, dict) else None
    clarity_score = clarity_score if clarity_score is not None else fb.get("clarity") if isinstance(fb, dict) else None
    improvement = fb.get("improvement") if isinstance(fb, dict) else None
    if isinstance(improvement, dict):
        improvement_keys = list(improvement.keys())
    elif isinstance(improvement, str):
        # Derive pseudo-keys from sentences for display
        improvement_keys = [s.strip()[:20] for s in improvement.split('.') if s.strip()][:3]
    else:
        improvement_keys = []
    print("   -> Feedback clarity:", clarity_score, "improvement keys:", improvement_keys)

    print("All steps completed.")


if __name__ == "__main__":
    run()
