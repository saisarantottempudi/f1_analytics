"""Train the LSTM race-finish predictor on features.parquet.

Usage:
    python scripts/train_predictor.py                 # default 30 epochs
    python scripts/train_predictor.py --epochs 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ml.predictor import (  # noqa: E402
    CAT_FEATURES, NUMERIC_FEATURES,
    DriverSequenceDataset, Encoders, LSTMPredictor,
    save_checkpoint,
)

FEATURES = ROOT / "data" / "processed" / "features.parquet"


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val-seasons", type=int, default=1,
                    help="Hold out the last N seasons for validation.")
    args = ap.parse_args()

    df = pd.read_parquet(FEATURES)
    seasons = sorted(df["season"].unique())
    if args.val_seasons <= 0 or args.val_seasons >= len(seasons):
        # No holdout — train and val share the dataset (ok for the single-season slice).
        train_seasons = val_seasons = seasons
    else:
        train_seasons = seasons[: -args.val_seasons]
        val_seasons = seasons[-args.val_seasons:]
    train_df = df[df["season"].isin(train_seasons)].copy()
    val_df = df[df["season"].isin(val_seasons)].copy()
    print(f"train: {len(train_df)} rows · seasons {train_seasons}")
    print(f"val:   {len(val_df)} rows · seasons {val_seasons}")

    encoders = Encoders.fit(df)
    train_ds = DriverSequenceDataset(train_df, encoders, seq_len=args.seq_len)
    val_ds = DriverSequenceDataset(val_df, encoders, seq_len=args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = _device()
    model = LSTMPredictor(
        n_numeric=len(NUMERIC_FEATURES),
        cat_cardinalities=encoders.cardinalities(),
        hidden=args.hidden,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    def move(batch):
        num, cats, y = batch
        num = num.to(device)
        cats = {c: v.to(device) for c, v in cats.items()}
        y = y.to(device)
        return num, cats, y

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            num, cats, y = move(batch)
            opt.zero_grad()
            pred = model(num, cats)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        train_mse = total / max(len(train_ds), 1)

        model.eval()
        with torch.no_grad():
            vtotal = 0.0
            for batch in val_loader:
                num, cats, y = move(batch)
                pred = model(num, cats)
                vtotal += loss_fn(pred, y).item() * len(y)
            val_mse = vtotal / max(len(val_ds), 1)

        marker = "  ★" if val_mse < best_val else ""
        best_val = min(best_val, val_mse)
        print(f"epoch {epoch+1:3d}/{args.epochs}  train MSE {train_mse:6.3f}  "
              f"val MSE {val_mse:6.3f}  (RMSE {val_mse**0.5:.2f}){marker}")

    path = save_checkpoint(model.cpu(), encoders)
    print(f"\nSaved → {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
