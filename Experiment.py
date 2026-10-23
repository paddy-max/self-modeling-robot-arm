"""Run one experiment and save everything automatically.

Example:
    python experiment.py --name pool-2000 --seeds 10 --pool 2000

Each run creates a folder:  experiments/<date>_<name>/
    config.json   <- the exact settings you used
    results.csv   <- every number (open it in Excel if you like)
    summary.md    <- a ready-made results table
    plot.png      <- error vs number of moves, random vs guided
"""
import argparse, csv, copy, json, platform, time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lab import Arm, Ensemble, rand_angles, adapt

p = argparse.ArgumentParser()
p.add_argument("--name", default="active-vs-random")
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--damaged_l2", type=float, default=0.4, help="new length of link 2 (healthy = 0.8)")
p.add_argument("--pool", type=int, default=400, help="candidate moves the guided strategy chooses from")
p.add_argument("--per_round", type=int, default=5)
p.add_argument("--rounds", type=int, default=16)
args = p.parse_args()

out = Path("experiments") / f"{datetime.now():%Y-%m-%d}_{args.name}"
out.mkdir(parents=True, exist_ok=True)
(out / "config.json").write_text(json.dumps(
    {**vars(args), "started": datetime.now().isoformat(timespec="seconds"),
     "python": platform.python_version(), "numpy": np.__version__}, indent=2))

rows, t0 = [], time.time()
for s in range(args.seeds):
    rng = np.random.default_rng(s)
    healthy, damaged = Arm(), Arm((1.0, args.damaged_l2, 0.5))
    test = rand_angles(rng, 1500)
    ens = Ensemble(rng); ens.train(healthy, rng, 3000, 3e-3)           # learn the healthy body
    for strat in ("random", "active"):
        curve, _ = adapt(copy.deepcopy(ens), damaged, strat, np.random.default_rng(100 + s),
                         test, rounds=args.rounds, per_round=args.per_round, pool=args.pool)
        rows += [(s, strat, (i + 1) * args.per_round, e) for i, e in enumerate(curve)]
    print(f"seed {s + 1}/{args.seeds} done ({time.time() - t0:.0f}s)", flush=True)

with open(out / "results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["seed", "strategy", "moves", "error"]); w.writerows(rows)

moves = sorted({r[2] for r in rows}); stats = {}
for strat in ("random", "active"):
    for m in moves:
        v = np.array([r[3] for r in rows if r[1] == strat and r[2] == m])
        stats[(strat, m)] = (v.mean(), v.std())

lines = [f"# {args.name}", "", f"Seeds: {args.seeds}, damaged link 2 = {args.damaged_l2}, pool = {args.pool}", "",
         "| Moves | Random (mean ± sd) | Guided (mean ± sd) |", "|---|---|---|"]
for m in moves:
    (rm, rs), (am, a_s) = stats[("random", m)], stats[("active", m)]
    lines.append(f"| {m} | {rm:.3f} ± {rs:.3f} | {am:.3f} ± {a_s:.3f} |")
(out / "summary.md").write_text("\n".join(lines) + "\n")

plt.figure(figsize=(6, 4))
for strat, label in (("random", "random moves"), ("active", "uncertainty-guided")):
    mu = np.array([stats[(strat, m)][0] for m in moves]); sd = np.array([stats[(strat, m)][1] for m in moves])
    plt.plot(moves, mu, label=label); plt.fill_between(moves, mu - sd, mu + sd, alpha=0.2)
plt.xlabel("moves after damage"); plt.ylabel("body error (arm-length units)")
plt.title(f"Recovery after damage ({args.seeds} seeds)"); plt.legend(); plt.tight_layout()
plt.savefig(out / "plot.png", dpi=150)
print(f"\nSaved to {out}/"); print("\n".join(lines[4:]))