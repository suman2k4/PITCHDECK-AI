// --- Upload Flow ---
window.uploadPitchDeck = async function uploadPitchDeck(userId, file) {
  const formData = new FormData();
  formData.append('user_id', userId);
  formData.append('file', file);
  const res = await fetch('/api/upload', {
    method: 'POST',
    body: formData
  });
  if (!res.ok) throw new Error('Upload failed');
  return await res.json();
}

// --- Persona Selection ---
window.setPersona = async function setPersona(sessionId, persona) {
  const res = await fetch('/api/set_persona', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({session_id: sessionId, persona})
  });
  if (!res.ok) throw new Error('Persona set failed');
  return await res.json();
}

// --- Text Extraction ---
window.extractText = async function extractText(sessionId) {
  const res = await fetch(`/api/extract_text?session_id=${sessionId}`, {
    method: 'POST'
  });
  if (!res.ok) throw new Error('Text extraction failed');
  return await res.json();
}

// --- Dual-Axis Analysis ---
window.analyzeSlides = async function analyzeSlides(sessionId, slides) {
  const res = await fetch('/api/analyze', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({session_id: sessionId, slides})
  });
  if (!res.ok) throw new Error('Analysis failed');
  return await res.json();
}

// --- Q&A Engine ---
window.startQA = async function startQA(sessionId, persona, context, opts={}) {
  const res = await fetch('/api/qa/start', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({session_id: sessionId, persona, context, hints: opts.hints !== false})
  });
  if (!res.ok) throw new Error('Q&A start failed');
  return await res.json();
}

window.nextQA = async function nextQA(sessionId, history, opts={}) {
  const res = await fetch('/api/qa/next', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({session_id: sessionId, history, hints: opts.hints !== false})
  });
  if (!res.ok) throw new Error('Q&A next failed');
  return await res.json();
}

// --- Feedback Report ---
window.getFeedback = async function getFeedback(sessionId, transcript) {
  const res = await fetch('/api/feedback', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({session_id: sessionId, transcript})
  });
  if (!res.ok) throw new Error('Feedback failed');
  return await res.json();
}

// --- Coaching ---
window.getCoach = async function getCoach(sessionId, question, answer, options={}) {
  const res = await fetch('/api/coach', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({session_id: sessionId, question, answer, persona: options.persona, tone: options.tone, length: options.length})
  });
  if (!res.ok) throw new Error('Coach failed');
  return await res.json();
}
window.postJson = async function postJson(url, body) {
  const resultsEl = document.getElementById('results');
  if (resultsEl) resultsEl.innerHTML = '<em>Loading...</em>';
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    if (!res.ok) throw new Error('API error: ' + res.status);
    const data = await res.json();
    return data;
  } catch (e) {
    const el = document.getElementById('results');
    if (el) el.innerHTML = `<span style='color:red'>${e}</span>`;
    throw e;
  }
}

function renderResults(data) {
  if (!data || !data.results) return;
  let html = '<ol>';
  for (const r of data.results) {
    html += `<li><b>Score:</b> ${r.score.toFixed(2)}<br/><pre>${r.text.slice(0,400)}</pre></li>`;
  }
  html += '</ol>';
  const el = document.getElementById('results');
  if (el) el.innerHTML = html;
}

function renderSynth(data) {
  if (!data || !data.answer) return;
  let html = `<div style='background:#e0ffe0;padding:10px;border-radius:6px'><b>Answer:</b><br/>${data.answer}</div>`;
  if (data.sources) {
    html += '<h3>Sources</h3><ol>';
    for (const r of data.sources) {
      html += `<li><b>Score:</b> ${r.score.toFixed(2)}<br/><pre>${r.text.slice(0,400)}</pre></li>`;
    }
    html += '</ol>';
  }
  const el = document.getElementById('results');
  if (el) el.innerHTML = html;
}

const searchBtn = document.getElementById('search');
if (searchBtn) {
  searchBtn.addEventListener('click', async ()=>{
    const qEl = document.getElementById('q');
    const q = qEl ? qEl.value : '';
    try {
      const data = await postJson('/api/search', {query:q, k:5});
      renderResults(data);
    } catch(e) {}
  });
}

const synthBtn = document.getElementById('synth');
if (synthBtn) {
  synthBtn.addEventListener('click', async ()=>{
    const qEl = document.getElementById('q');
    const q = qEl ? qEl.value : '';
    try {
      const data = await postJson('/api/synthesize', {query:q, k:5});
      renderSynth(data);
    } catch(e) {}
  });
}
