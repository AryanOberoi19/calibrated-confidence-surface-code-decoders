"""M0 pipeline check: reproduce a surface-code threshold plot in simulation.

Usage:  python scripts/m0_threshold.py [--workers 8] [--max-shots 200000]

Rotated memory-Z circuits with rounds = d and uniform circuit-level depolarizing noise,
decoded with PyMatching via sinter. Writes results/m0_threshold.csv (resumable) and
results/m0_threshold.png. Expected: curves for d = 3, 5, 7 cross near p = 0.6%.
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sinter
import stim

REPO = pathlib.Path(__file__).resolve().parents[1]
DISTANCES = (3, 5, 7)
NOISE = (0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010)


def tasks():
    for d in DISTANCES:
        for p in NOISE:
            c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                       after_clifford_depolarization=p, before_round_data_depolarization=p,
                                       before_measure_flip_probability=p, after_reset_flip_probability=p)
            yield sinter.Task(circuit=c, json_metadata={"d": d, "p": p})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-shots", type=int, default=200_000)
    ap.add_argument("--max-errors", type=int, default=300)
    args = ap.parse_args()
    out = REPO / "results"
    out.mkdir(exist_ok=True)
    stats = sinter.collect(num_workers=args.workers, tasks=tasks(), decoders=["pymatching"],
                           max_shots=args.max_shots, max_errors=args.max_errors,
                           save_resume_filepath=str(out / "m0_threshold.csv"), print_progress=True)
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    sinter.plot_error_rate(ax=ax, stats=stats, x_func=lambda s: s.json_metadata["p"],
                           group_func=lambda s: f"d = {s.json_metadata['d']}")
    ax.loglog()
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("logical error rate per shot")
    ax.grid(True, which="major", color="#e1e0d9", linewidth=0.6)
    ax.legend()
    fig.savefig(out / "m0_threshold.png", dpi=160, bbox_inches="tight")
    print(f"wrote {out / 'm0_threshold.png'}")


if __name__ == "__main__":   # required: sinter uses worker processes (spawn on macOS)
    main()
