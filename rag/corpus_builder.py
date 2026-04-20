"""Generate one-paragraph narratives per race from results + features.

Produces ~1 document per race: a compact description that captures the
winner, podium, pole, notable retirements, biggest gains/losses, and
weather context. Those documents are what the Chroma index indexes and
what the RAG layer retrieves for explanation prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ERGAST = ROOT / "data" / "raw" / "ergast"
CORPUS_DIR = ROOT / "rag" / "corpus"


@dataclass
class RaceDoc:
    doc_id: str           # e.g. "2024-15-zandvoort"
    text: str
    metadata: dict        # season, round, circuit_id, winner, pole


def _classify_status(s: str) -> str:
    if s == "Finished" or s.startswith("+"):
        return "classified"
    return "retired"


def _fmt_driver(row: pd.Series) -> str:
    return f"{row['driver_name']} ({row['constructor_id']})"


def _race_doc(race_rows: pd.DataFrame) -> RaceDoc | None:
    r = race_rows.sort_values("position", na_position="last")
    if r.empty:
        return None
    season = int(r.iloc[0]["season"])
    rnd = int(r.iloc[0]["round"])
    race_name = r.iloc[0]["race_name"]
    circuit_id = r.iloc[0]["circuit_id"]
    date = r.iloc[0]["date"]

    finishers = r[r["position"].notna()].sort_values("position")
    retirees = r[r["position"].isna()]
    pole_row = r[r["grid"] == 1]
    pole = _fmt_driver(pole_row.iloc[0]) if not pole_row.empty else "unknown"

    if finishers.empty:
        return None

    winner = finishers.iloc[0]
    podium = finishers.head(3)
    podium_str = ", ".join(
        f"P{int(row['position'])} {_fmt_driver(row)}" for _, row in podium.iterrows()
    )

    # Biggest gainer / biggest loser among classified finishers.
    cls = finishers.copy()
    cls["delta"] = cls["grid"] - cls["position"]
    cls = cls[cls["grid"] > 0]
    gainer_note = loser_note = ""
    if not cls.empty:
        g = cls.loc[cls["delta"].idxmax()]
        l = cls.loc[cls["delta"].idxmin()]
        if g["delta"] >= 3:
            gainer_note = (
                f" Biggest move was {_fmt_driver(g)} gaining {int(g['delta'])} "
                f"positions (P{int(g['grid'])} → P{int(g['position'])})."
            )
        if l["delta"] <= -3:
            loser_note = (
                f" {_fmt_driver(l)} lost {int(-l['delta'])} positions "
                f"(P{int(l['grid'])} → P{int(l['position'])})."
            )

    retire_note = ""
    if not retirees.empty:
        retire_names = ", ".join(_fmt_driver(row) for _, row in retirees.iterrows())
        retire_note = f" {len(retirees)} driver(s) retired: {retire_names}."

    text = (
        f"{season} {race_name} at {circuit_id} ({date}). "
        f"{_fmt_driver(winner)} won from grid P{int(winner['grid'])}. "
        f"Podium: {podium_str}. "
        f"Pole went to {pole}."
        f"{gainer_note}{loser_note}{retire_note}"
    )

    meta = {
        "season": season,
        "round": rnd,
        "circuit_id": str(circuit_id),
        "race_name": str(race_name),
        "winner_id": str(winner["driver_id"]),
        "pole_id": str(pole_row.iloc[0]["driver_id"]) if not pole_row.empty else "",
        "date": str(date),
    }
    return RaceDoc(
        doc_id=f"{season}-{rnd:02d}-{circuit_id}",
        text=text,
        metadata=meta,
    )


def build_corpus(results_path: Path = ERGAST / "results.parquet") -> list[RaceDoc]:
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} — run `scripts/ingest_ergast.py` first")
    results = pd.read_parquet(results_path)
    docs: list[RaceDoc] = []
    for (season, rnd), grp in results.groupby(["season", "round"], sort=True):
        doc = _race_doc(grp)
        if doc is not None:
            docs.append(doc)
    return docs


def dump_corpus_to_disk(docs: list[RaceDoc], out_dir: Path = CORPUS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # One JSONL file, easier to diff / inspect than N markdowns.
    path = out_dir / "races.jsonl"
    with path.open("w") as f:
        import json
        for d in docs:
            f.write(json.dumps({"id": d.doc_id, "text": d.text, **d.metadata}) + "\n")
    return path
