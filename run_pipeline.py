"""Simple CLI to run embedding and query steps for the pitchdrill pipeline."""
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run_embed():
    script = str(ROOT.joinpath('rag_engine', 'embed_chunks.py'))
    print(f"Running {script}...")
    res = subprocess.run([sys.executable, "-u", script], check=False)
    return res.returncode == 0


def run_query():
    script = str(ROOT.joinpath('rag_engine', 'query_handler.py'))
    print(f"Starting interactive {script} (ctrl-c to exit)")
    res = subprocess.run([sys.executable, "-u", script], check=False)
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", action="store_true", help="Run embedding step")
    parser.add_argument("--query", action="store_true", help="Run query handler")
    args = parser.parse_args()

    if not args.embed and not args.query:
        parser.print_help()
        return

    if args.embed:
        if not run_embed():
            print("Embedding failed.")
            return

    if args.query:
        run_query()


if __name__ == "__main__":
    main()
