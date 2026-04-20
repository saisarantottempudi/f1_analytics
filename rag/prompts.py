"""Prompt templates for the three modes.

Each template takes the ML output (a dict) and a list of retrieved race
summaries, and produces:
  (a) a deterministic narrative (template-only), and
  (b) a user-message string for optional LLM refinement.

The system prompt (SYSTEM_EXPLAINER) is shared across modes and cacheable.
"""

from __future__ import annotations

SYSTEM_EXPLAINER = (
    "You are an F1 race analyst. You write tight, confident, 2-4 sentence "
    "explanations of ML model outputs. You ground claims in the historical "
    "context provided. You never invent specific facts (lap counts, retirement "
    "reasons, specific gaps) that are not in the context. When the model output "
    "is uncertain, you say so. No emojis unless asked."
)


def _fmt_context(retrieved: list[dict], k: int = 3) -> str:
    if not retrieved:
        return "(no historical context retrieved)"
    return "\n".join(f"- {r['text']}" for r in retrieved[:k])


def prediction_prompt(ml_output: dict, retrieved: list[dict]) -> tuple[str, str]:
    """Returns (deterministic_narrative, llm_user_message)."""
    top3 = sorted(ml_output["drivers"], key=lambda d: -d["p_win"])[:3]
    winner = top3[0]
    pole = ml_output["pole_driver_id"]
    sc_pct = ml_output["sc_probability"] * 100

    # Deterministic narrative — always available.
    lines = [
        f"Predicted winner: {winner['driver_id']} ({winner['p_win']*100:.1f}% P(win), "
        f"{winner['p_podium']*100:.0f}% P(podium)).",
        f"Pole: {pole}.",
        f"Top 3 by win probability: "
        + ", ".join(f"{d['driver_id']} ({d['p_win']*100:.1f}%)" for d in top3) + ".",
        f"Safety Car probability: {sc_pct:.0f}%.",
    ]
    narrative = " ".join(lines)

    context_block = _fmt_context(retrieved)
    top_table = "\n".join(
        f"  {d['driver_id']:<20} grid=P{d['grid_position']:<2} "
        f"mean_finish={d['mean_finish']:.2f} P(win)={d['p_win']*100:.1f}% "
        f"P(podium)={d['p_podium']*100:.1f}% P(points)={d['p_points']*100:.1f}%"
        for d in sorted(ml_output["drivers"], key=lambda d: d["mean_finish"])[:8]
    )
    llm_msg = (
        "Race prediction for circuit "
        f"'{ml_output['circuit_id']}', {ml_output['n_sims']} Monte Carlo sims.\n\n"
        f"Top-8 predicted order:\n{top_table}\n\n"
        f"Pole: {pole}. Predicted winner by P(win): {winner['driver_id']}. "
        f"Race SC probability: {sc_pct:.0f}%.\n\n"
        f"Historical context (most similar past races from the index):\n{context_block}\n\n"
        "Write 3-4 sentences: who's likely to win and why, the main threat, and one "
        "context-grounded nuance (weather, track character, recent form). "
        "Do not invent facts."
    )
    return narrative, llm_msg


def simulate_prompt(sim_output: dict, retrieved: list[dict]) -> tuple[str, str]:
    final = sim_output["final_standings"]
    events = sim_output["events"]
    sc_n = sum(1 for e in events if e["kind"] == "SC")
    vsc_n = sum(1 for e in events if e["kind"] == "VSC")
    dnf_n = sum(1 for e in events if e["kind"] == "DNF")
    winner = final[0]["driver_id"]
    podium = [s["driver_id"] for s in final[:3]]

    narrative = (
        f"Race winner: {winner}. "
        f"Podium: {', '.join(podium)}. "
        f"Events: {sc_n} Safety Car, {vsc_n} VSC, {dnf_n} retirements."
    )

    context_block = _fmt_context(retrieved)
    top_lines = "\n".join(
        f"  P{row['position']:<2} {row['driver_id']}"
        + (f" (DNF lap {row['retired_lap']})" if row.get("retired_lap") else "")
        for row in final[:8]
    )
    event_lines = "\n".join(
        f"  lap {e['lap']:<3} {e['kind']:<4} {e.get('driver_id') or ''} {e.get('detail') or ''}"
        for e in events[:12]
    )
    llm_msg = (
        f"Race simulation for circuit '{sim_output['circuit_id']}' "
        f"over {sim_output['total_laps']} laps (ran {sim_output['laps_ran']}).\n\n"
        f"Final top 8:\n{top_lines}\n\n"
        f"Notable events:\n{event_lines}\n\n"
        f"Historical context:\n{context_block}\n\n"
        "Write 3-4 sentences of race-commentary style narrative: how the race unfolded, "
        "who benefited from SCs/retirements, and the key moment that decided the win. "
        "Ground any comparisons in the historical context above."
    )
    return narrative, llm_msg


def h2h_prompt(h2h_output: dict, retrieved: list[dict]) -> tuple[str, str]:
    a, b = h2h_output["driver_a"], h2h_output["driver_b"]
    winner = h2h_output["overall_winner"]
    edge = h2h_output["overall_edge_pct"]
    shared = h2h_output["shared_races"]

    winner_label = {"A": a, "B": b, "TIE": "tied"}[winner]
    narrative = (
        f"Head-to-head over {shared} shared races: {winner_label} "
        + (f"leads by {edge:.0f}% edge." if winner != "TIE" else "— too close to call.")
    )

    rows = "\n".join(
        f"  {s['label']:<45} A={s['a_value']}  B={s['b_value']}  winner={s['winner']}"
        for s in h2h_output["sections"]
    )
    context_block = _fmt_context(retrieved)
    llm_msg = (
        f"Driver head-to-head: {a} vs {b}. Shared races: {shared}.\n\n"
        f"Comparison breakdown:\n{rows}\n\n"
        f"Historical context (races featuring either driver):\n{context_block}\n\n"
        f"Write 3-4 sentences: who has the edge overall and on which dimensions, "
        "where the other driver pushes back, and one data-grounded observation. "
        "No invented stats."
    )
    return narrative, llm_msg
