"""Record a CPU energy diagnostic for explicit spin configurations."""

from itertools import product
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from py_func import energy_ising3d


def main() -> None:
    """Save calculated energies, clearly separate from a training benchmark."""
    root = Path(__file__).resolve().parents[1]
    length, beta = 3, 0.22
    spins = torch.ones(2, 1, length, length, length)
    spins[1, 0, 0, 0, 0] = -1
    energies = energy_ising3d(
        2, length, spins, list(product(range(length), repeat=3)), beta
    ).tolist()
    (root / "results").mkdir(exist_ok=True)
    (root / "results/diagnostic.json").write_text(
        json.dumps(
            {
                "kind": "CPU diagnostic, no training",
                "L": length,
                "beta": beta,
                "energies": energies,
            },
            indent=2,
        )
        + "\n"
    )
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    bars = ax.bar(
        ["All aligned", "One flipped spin"], energies, color=["#176b60", "#547da8"]
    )
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(
        ylabel="Energy including beta",
        title="Explicit configurations · CPU diagnostic · no training",
    )
    ax.spines[["top", "right"]].set_visible(False)
    (root / "assets").mkdir(exist_ok=True)
    fig.savefig(root / "assets/results.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
