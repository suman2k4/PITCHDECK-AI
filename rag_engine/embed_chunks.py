import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

# Load .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
if not API_KEY:
    logging.warning("GEMINI_API_KEY not set. Embedding calls will fail without it.")
else:
    genai.configure(api_key=API_KEY)


def load_chunks_from_folder(folder_path: str) -> list[str]:
    folder = Path(folder_path)
    chunks = []
    if not folder.exists():
        logging.error(f"Chunks folder not found: {folder}")
        return chunks
    for filename in sorted(folder.iterdir()):
        if filename.suffix == ".txt":
            chunks.append(filename.read_text(encoding="utf-8"))
    return chunks


def create_embeddings(chunks: list[str]) -> np.ndarray:
    if GoogleGenerativeAIEmbeddings is None:
        raise RuntimeError("GoogleGenerativeAIEmbeddings not available")
    emb = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)
    vectors = [e for e in emb.embed_documents(chunks)]
    return np.array(vectors).astype('float32')


def build_faiss_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def save_index(index, out_dir: str, chunks: list[str]):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out.joinpath('index.faiss')))
    # save texts
    with open(out.joinpath('index.txt'), 'w', encoding='utf-8') as f:
        for c in chunks:
            f.write(c.replace('\n', ' ') + '\n---\n')


def main():
    here = Path(__file__).resolve().parent
    chunks_dir = here.parent.joinpath("input", "output_chunks")
    index_dir = here.parent.joinpath("input", "faiss_index")

    chunks = load_chunks_from_folder(str(chunks_dir))
    if not chunks:
        logging.error("No chunks to index. Exiting.")
        return

    logging.info(f"Loaded {len(chunks)} chunks.")
    if not API_KEY or GoogleGenerativeAIEmbeddings is None:
        logging.warning("GEMINI_API_KEY not set or embeddings library missing — creating a mock index for dry-run.")
        # create deterministic random vectors for dry-run
        rng = np.random.default_rng(12345)
        vectors = rng.normal(size=(len(chunks), 1536)).astype('float32')
        index = build_faiss_index(vectors)
        save_index(index, str(index_dir), chunks)
        logging.info(f"Stored mock index with {len(chunks)} vectors in FAISS at {index_dir}.")
        return

    logging.info(f"Creating real embeddings using model {EMBEDDING_MODEL}...")
    try:
        vectors = create_embeddings(chunks)
        index = build_faiss_index(vectors)
        save_index(index, str(index_dir), chunks)
        logging.info(f"Stored {len(chunks)} chunks in FAISS at {index_dir}.")
    except Exception as e:
        logging.warning(f"Embedding failed: {e}. Falling back to mock index.")
        rng = np.random.default_rng(12345)
        vectors = rng.normal(size=(len(chunks), 1536)).astype('float32')
        index = build_faiss_index(vectors)
        save_index(index, str(index_dir), chunks)
        logging.info(f"Stored mock index with {len(chunks)} vectors in FAISS at {index_dir}.")


if __name__ == "__main__":
    main()
