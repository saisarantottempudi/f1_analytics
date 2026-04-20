"""Generate race narratives + embed them into Chroma.

Usage:
    python scripts/build_rag_index.py
    python scripts/build_rag_index.py --no-reset    # append only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.corpus_builder import build_corpus, dump_corpus_to_disk  # noqa: E402
from rag.index import build_index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-reset", dest="reset", action="store_false")
    args = ap.parse_args()

    print("Building corpus…")
    docs = build_corpus()
    print(f"  {len(docs)} race narratives generated")
    jsonl = dump_corpus_to_disk(docs)
    print(f"  dumped → {jsonl.relative_to(ROOT)}")

    print("Embedding into Chroma…")
    n = build_index(docs, reset=args.reset)
    print(f"  collection now holds {n} documents")

    if docs:
        print("\nSample narrative:")
        print(f"  {docs[0].text}")


if __name__ == "__main__":
    main()
