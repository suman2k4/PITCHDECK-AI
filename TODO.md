# TODO List for Pitchdeck AI Project

## Completed Tasks
- [x] Update README.md with Pitchdeck AI details, APBI algorithm, features, setup, and usage
- [x] Clean up irrelevant files (duplicates, unused samples like sample_pitch.pdf, outputs/, typos in prompts, old prompts, empty files)
- [x] Extract pitch deck text from input/NVCPitchDeckTemplate.pdf to input/parsed_text.txt
- [x] Chunk the extracted text into 6 chunks in input/output_chunks/
- [x] Build FAISS index with mock embeddings (due to API quota)
- [x] Start web server on http://127.0.0.1:8000
- [x] Replace old index.html with new home page featuring Pitchdeck AI branding and links to login/signup
- [x] Test web app: Home page loads correctly, login.html and signup.html accessible
- [x] Fix server.py issues: Remove duplicate imports, ensure API endpoints are correctly defined
- [x] Test API endpoints: /api/signup, /api/login, /api/search, /api/synthesize work correctly

## Remaining Tasks
- [x] Implement database saving for answers and feedback in Q&A simulation
- [x] Add /api/analyze endpoint for pitch deck analysis
- [x] Test remaining API endpoints: /api/upload, /api/extract_text, /api/set_persona, /api/analyze, /api/qa/start, /api/qa/next, /api/feedback
- [ ] Ensure RAG retrieval works with FAISS index for context-aware questions
- [ ] Add JavaScript functionality to web pages for full user flow
- [ ] Test end-to-end: Sign up, log in, upload pitch deck, select persona, practice Q&A, view feedback
- [ ] Optimize prompts for better APBI algorithm performance
- [ ] Add error handling and user feedback in web UI
