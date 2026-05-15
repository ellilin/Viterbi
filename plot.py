from __future__ import annotations

import csv
import os
from pathlib import Path

cache_dir = Path("results/.cache").resolve()
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str((cache_dir / "matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def main() -> None:
    csv_path = Path("results/ber.csv")
    if not csv_path.exists():
        raise SystemExit("results/ber.csv not found. Run `make run` first.")

    channel_p: list[float] = []
    decoded_ber: list[float] = []
    raw_ber: list[float] = []

    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            channel_p.append(float(row["channel_error_probability"]))
            decoded_ber.append(float(row["decoded_bit_error_rate"]))
            raw_ber.append(float(row["raw_channel_bit_error_rate"]))

    Path("results").mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(channel_p, decoded_ber, "o-", label="После декодера Витерби")
    plt.plot(channel_p, raw_ber, "s--", label="В канале без декодирования", alpha=0.75)
    plt.xlabel("Вероятность ошибки в двоичном симметричном канале")
    plt.ylabel("Вероятность ошибки на бит")
    plt.title("BER после декодирования Витерби")
    plt.grid(True, which="both", linestyle=":", linewidth=0.8)
    plt.yscale("log")
    plt.ylim(bottom=max(1e-6, min(decoded_ber + raw_ber) / 2))
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/ber.png", dpi=180)
    print("Saved results/ber.png")


if __name__ == "__main__":
    main()
