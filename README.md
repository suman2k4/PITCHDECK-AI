# Pitchdeck AI — Adaptive Persona-Based Interrogation (APBI) for VC Pitch Practice

This repository implements Pitchdeck AI, a web application that simulates realistic Venture Capital (VC) Q&A sessions for founders to practice their pitch decks. It uses the Adaptive Persona-Based Interrogation (APBI) algorithm, powered by Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs), to generate persona-driven, challenging questions based on the pitch deck content and conversation history.

## Features

- **Persona-Driven Q&A Simulation**: Choose from VC personas (e.g., SaaS Guru, Deep Tech Skeptic, Early Stage Angel, Growth Investor) or use a generic baseline for comparison.
- **RAG-Powered Insights**: Embed pitch deck content into a FAISS vector database for context-aware question generation and answer synthesis.
- **Web UI**: Full-stack web application with user authentication, pitch deck upload, persona selection, interactive Q&A sessions, and AI-generated feedback.
- **LLM Integration**: Uses Google Gemini for embeddings, question generation, and feedback evaluation.
- **Extraction Pipeline**: Automated parsing and chunking of pitch decks (PDF, DOCX, PPTX, TXT).

## Project Structure

- `extraction/`: Jupyter notebooks for parsing pitch decks and splitting text into chunks.
- `input/`: Folder for pitch deck files, parsed text, chunks, and FAISS index.
- `prompts/`: Persona-specific prompt files for LLM question generation.
- `rag_engine/`: Core RAG components for embedding, querying, and question generation.
- `web/`: Frontend HTML/JS files for the web application.
- `server.py`: FastAPI backend server with endpoints for upload, Q&A, feedback, etc.
- `models.py`: SQLAlchemy models for users, sessions, files, and feedback.
- `run_pipeline.py`: CLI for running embedding and query steps.

## Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key (get from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Setup

1. Clone the repository and navigate to the project directory.

2. Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-pro
GEMINI_EMBEDDING_MODEL=models/embedding-001
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Prepare Pitch Deck Data

1. Place your pitch deck file (e.g., `NVCPitchDeckTemplate.pdf`) in the `input/` folder.

2. Run text extraction:

   Execute the code in `extraction/parse_pitch.ipynb` to parse the pitch deck into `input/parsed_text.txt`.

3. Run text chunking:

   Execute the code in `extraction/split_chunks.ipynb` to split the text into chunks in `input/output_chunks/`.

### Build RAG Index

Run the embedding pipeline to create FAISS index:

```bash
python run_pipeline.py --embed
```

This generates embeddings using Gemini and stores the FAISS index in `input/faiss_index/`.

### Run the Web Application

1. Start the backend server:

```bash
python server.py
```

The server runs on http://127.0.0.1:8000.

2. Open your browser and navigate to http://127.0.0.1:8000.

3. Use the web app:
   - Sign up or log in.
   - Upload your pitch deck.
   - Select a VC persona (or toggle to generic baseline).
   - Engage in an interactive Q&A session.
   - View AI-generated feedback on your performance.

### Optional: CLI Querying

For command-line interaction with the RAG index:

```bash
python run_pipeline.py --query
```

## APBI Algorithm Overview

The Adaptive Persona-Based Interrogation (APBI) algorithm generates realistic VC questions by:

1. Retrieving relevant pitch deck context using RAG.
2. Incorporating persona-specific traits and conversation history.
3. Using LLM prompts tailored to each persona for challenging, domain-specific questions.
4. Providing baseline generic mode for comparison against state-of-the-art models.

Personas include:
- **SaaS Guru**: Focuses on unit economics, churn, and scalability.
- **Deep Tech Skeptic**: Probes technical risks, IP, and defensibility.
- **Early Stage Angel**: Emphasizes team, market validation, and milestones.
- **Growth Investor**: Questions growth metrics, competition, and exit potential.
- **Generic Baseline**: Standard questions for comparison.

## API Endpoints

- `POST /api/signup`: User registration.
- `POST /api/login`: User authentication.
- `POST /api/upload`: Upload pitch deck file.
- `POST /api/set_persona`: Set VC persona for session.
- `POST /api/qa/start`: Start Q&A session.
- `POST /api/qa/next`: Generate next question.
- `POST /api/feedback`: Generate session feedback.
- `POST /api/search`: Search pitch deck content.
- `POST /api/synthesize`: Synthesize answers from context.

## Notes

- The project uses Google Gemini for embeddings and LLM calls. Ensure `GEMINI_API_KEY` is set.
- If the FAISS index exists, embedding steps can be skipped.
- For production, consider using a proper database (e.g., PostgreSQL) instead of SQLite.
- The web UI is served statically; for production, use a proper web server.

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements, especially around persona prompts or feedback evaluation.

