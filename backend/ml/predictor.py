"""LSTM race-finish predictor.

Per-driver sequence model: feeds a chronological window of engineered
features into a stacked LSTM and regresses the finish position for the
next race. MC-dropout at inference gives a predictive distribution
instead of a point estimate, which downstream code turns into finish
probabilities.

Training is driven by `scripts/train_predictor.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "lstm_predictor.pt"

# Feature set — numeric columns pulled from features.parquet.
# Categorical features (driver/team/circuit) are embedded separately.
NUMERIC_FEATURES: list[str] = [
    "grid",
    "drv_rollN_avg_finish_5",
    "drv_rollN_dnf_rate_5",
    "drv_rollN_points_avg_5",
    "drv_circ_avg_finish_prior",
    "drv_circ_races_prior",
    "drv_season_finish_std",
    "drv_season_finish_iqr",
    "team_rollN_avg_finish_3",
    "tmate_quali_win_rate_season",
    "tmate_race_win_rate_season",
    "circ_pole_to_win_rate",
    "circ_dnf_rate",
    "circ_grid_finish_corr",
]
CAT_FEATURES: list[str] = ["driver_id", "constructor_id", "circuit_id"]
TARGET = "position"


# ---------- data prep ----------

@dataclass
class Encoders:
    """Category → integer-id maps, shared between train and inference."""
    maps: dict[str, dict[str, int]]

    def encode(self, col: str, values: pd.Series) -> np.ndarray:
        m = self.maps[col]
        # 0 is reserved for unknown.
        return values.map(lambda v: m.get(v, 0)).to_numpy(dtype=np.int64)

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Encoders":
        maps: dict[str, dict[str, int]] = {}
        for c in CAT_FEATURES:
            uniq = sorted(df[c].dropna().unique().tolist())
            maps[c] = {v: i + 1 for i, v in enumerate(uniq)}  # reserve 0
        return cls(maps)

    def cardinalities(self) -> dict[str, int]:
        return {c: len(m) + 1 for c, m in self.maps.items()}


class DriverSequenceDataset(Dataset):
    """Each item = one race row; sequence = that driver's previous `seq_len` races."""

    def __init__(self, df: pd.DataFrame, encoders: Encoders, seq_len: int = 5):
        df = df.sort_values(["driver_id", "date"]).copy()
        # Impute numeric nulls with per-column median (causal-safe: computed over the
        # whole table, but nulls mostly represent "no history yet" which is the neutral
        # prior anyway).
        for c in NUMERIC_FEATURES:
            df[c] = df[c].fillna(df[c].median())
        df = df.dropna(subset=[TARGET])  # drop DNFs for regression
        df[TARGET] = df[TARGET].astype(float)

        self.encoders = encoders
        self.seq_len = seq_len
        self.numeric = df[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
        self.cats = {c: encoders.encode(c, df[c]) for c in CAT_FEATURES}
        self.targets = df[TARGET].to_numpy(dtype=np.float32)
        self.driver = df["driver_id"].to_numpy()

        # Per-driver index into rows, used to build history windows.
        self._row_idx: list[int] = []
        for drv, grp in df.groupby("driver_id", sort=False).indices.items():
            # `grp` is already in chronological order due to the sort above.
            self._row_idx.extend(grp.tolist())

        # Precompute padding vector once.
        self._pad_num = np.zeros(len(NUMERIC_FEATURES), dtype=np.float32)

        # Map each row in sorted order to (driver-group, position within group).
        self._driver_groups: dict[str, list[int]] = {}
        for pos, i in enumerate(self._row_idx):
            drv = self.driver[i]
            self._driver_groups.setdefault(drv, []).append(i)

    def __len__(self) -> int:
        return len(self._row_idx)

    def __getitem__(self, idx: int):
        i = self._row_idx[idx]
        drv = self.driver[i]
        hist = self._driver_groups[drv]
        pos = hist.index(i)
        window_start = max(0, pos - self.seq_len + 1)
        window = hist[window_start : pos + 1]

        num = np.stack([self.numeric[j] for j in window])
        if len(window) < self.seq_len:
            pad = np.tile(self._pad_num, (self.seq_len - len(window), 1))
            num = np.vstack([pad, num])

        # Categorical features stay the same across the sequence (same driver & team &
        # circuit for the target race — that's the context we condition on).
        cats = {c: self.cats[c][i] for c in CAT_FEATURES}
        return (
            torch.from_numpy(num),
            {c: torch.tensor(v, dtype=torch.long) for c, v in cats.items()},
            torch.tensor(self.targets[i], dtype=torch.float32),
        )


# ---------- model ----------

class LSTMPredictor(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: dict[str, int],
        embed_dim: int = 8,
        hidden: int = 48,
        n_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embeds = nn.ModuleDict({
            c: nn.Embedding(n, embed_dim, padding_idx=0)
            for c, n in cat_cardinalities.items()
        })
        lstm_in = n_numeric
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        head_in = hidden + embed_dim * len(cat_cardinalities)
        self.head = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, numeric: torch.Tensor, cats: dict[str, torch.Tensor]) -> torch.Tensor:
        _, (h, _) = self.lstm(numeric)
        h_last = h[-1]
        emb = torch.cat([self.embeds[c](cats[c]) for c in self.embeds], dim=-1)
        fused = torch.cat([h_last, emb], dim=-1)
        return self.head(fused).squeeze(-1)


# ---------- checkpoint I/O ----------

def save_checkpoint(model: LSTMPredictor, encoders: Encoders, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "encoders": encoders.maps,
            "cat_cardinalities": encoders.cardinalities(),
            "numeric_features": NUMERIC_FEATURES,
        },
        path,
    )
    return path


def load_checkpoint(path: Path = MODEL_PATH) -> tuple[LSTMPredictor, Encoders]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    enc = Encoders(maps=blob["encoders"])
    model = LSTMPredictor(
        n_numeric=len(blob["numeric_features"]),
        cat_cardinalities=blob["cat_cardinalities"],
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, enc


# ---------- inference with MC-dropout ----------

@torch.no_grad()
def predict_distribution(
    model: LSTMPredictor,
    numeric: torch.Tensor,
    cats: dict[str, torch.Tensor],
    n_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """MC-dropout — run the model N times with dropout ON, return per-row mean + std."""
    model.train()  # enable dropout
    preds: list[np.ndarray] = []
    for _ in range(n_samples):
        preds.append(model(numeric, cats).cpu().numpy())
    model.eval()
    arr = np.stack(preds, axis=0)
    return arr.mean(0), arr.std(0)


def position_probabilities(mu: np.ndarray, sigma: np.ndarray, n_positions: int = 20) -> np.ndarray:
    """Soft-rank: per-driver P(finish = k) from Gaussian (mu, sigma) discretised to [1..n]."""
    from scipy.stats import norm
    positions = np.arange(1, n_positions + 1)[None, :]  # (1, n)
    mu = mu[:, None]
    sigma = np.clip(sigma, 0.5, None)[:, None]
    logits = -0.5 * ((positions - mu) / sigma) ** 2
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    return probs
