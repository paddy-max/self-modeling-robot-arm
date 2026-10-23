"""Watch the arm learn its body, get damaged, and relearn, with live graphs.

    python visualize.py                  # live window on your laptop
    python visualize.py --quick          # shorter run, good for a first look
    python visualize.py --save demo.gif  # save an animation instead (use this on Google Colab)

Left:         the simulated arm (a simple 2D simulation drawn with matplotlib).
              Green dot = real hand. Red cross = where the arm's self-model THINKS its hand is.
              Red circle = how much the 5 networks disagree (uncertainty, drawn 5x bigger).
Top right:    error on 400 test poses (lower is better) and the uncertainty at the current pose.
Bottom right: accuracy = % of test poses where the guess lands within 0.05 of the real hand.
Phases:       1) learns its body from random practice  2) healthy and accurate
              3) DAMAGED (segment 2 gets shorter) and relearns from its own moves.
In the live window, speed depends on how fast your laptop can train the networks.
"""
import argparse
import numpy as np
import matplotlib
ap = argparse.ArgumentParser()
ap.add_argument("--save", default=None, help="save to this .gif instead of opening a window")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--quick", action="store_true", help="shorter run for a fast first look")
args = ap.parse_args()
if args.save:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from lab import Arm, Ensemble, rand_angles

rng = np.random.default_rng(args.seed)
healthy, damaged = Arm(), Arm((1.0, 0.4, 0.5))
ens = Ensemble(rng)
if args.quick:
    N_TRAIN, N_HEALTHY, N_RELEARN, TRAIN_STEPS = 10, 15, 40, 150
else:
    N_TRAIN, N_HEALTHY, N_RELEARN, TRAIN_STEPS = 40, 80, 240, 75
T1, T2 = N_TRAIN, N_TRAIN + N_HEALTHY            # the damage happens at frame T2
TOTAL = T2 + N_RELEARN
GOOD = 0.05                                       # "accurate" = within 0.05 of the real hand
test = rand_angles(np.random.default_rng(999), 400)
truth = {"h": healthy.hand(test), "d": damaged.hand(test)}

def pose(t):                                      # a smooth, repeatable movement for the display
    return np.array([[0.9*np.sin(0.06*t) + 0.2, 1.1*np.sin(0.09*t), 1.0*np.sin(0.13*t)]])

def joint_points(arm, a):
    c = np.cumsum(a[0])
    steps = np.stack([arm.L*np.cos(c), arm.L*np.sin(c)], 1)
    return np.vstack([[0, 0], np.cumsum(steps, axis=0)])

def metrics(key):
    d = np.linalg.norm(ens.mean(test) - truth[key], axis=1)
    return float(np.sqrt(np.mean(d**2))), float(np.mean(d < GOOD) * 100)

fig = plt.figure(figsize=(11.5, 5.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15])
ax, ax_e, ax_a = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.set_aspect("equal"); ax.grid(alpha=.3)
(links,) = ax.plot([], [], "o-", lw=5, color="#555", markersize=8)
(trail_line,) = ax.plot([], [], "-", color="#2a9d4a", alpha=.25, lw=1)
(true_dot,) = ax.plot([], [], "o", color="#2a9d4a", markersize=12, label="real hand")
(pred_dot,) = ax.plot([], [], "x", color="#d62828", markersize=12, mew=3, label="model's guess")
ring = Circle((0, 0), 0.02, fill=False, color="#d62828", ls="--"); ax.add_patch(ring)
ax.legend(loc="upper right", fontsize=8)
title = ax.set_title("phase line\nerror and accuracy line", fontsize=10)

ax_e.set_yscale("log"); ax_e.set_xlim(0, TOTAL); ax_e.set_ylim(0.003, 8)
ax_e.set_ylabel("error (arm-length units)"); ax_e.grid(alpha=.3)
(err_line,) = ax_e.plot([], [], color="#d62828", label="error on test poses")
(unc_line,) = ax_e.plot([], [], color="#1d6fa5", label="uncertainty (networks disagree)")
ax_e.legend(loc="upper right", fontsize=8)
ax_a.set_xlim(0, TOTAL); ax_a.set_ylim(0, 105); ax_a.grid(alpha=.3)
ax_a.set_ylabel(f"accuracy (% within {GOOD})"); ax_a.set_xlabel("time (frames)")
(acc_line,) = ax_a.plot([], [], color="#2a9d4a")
for a_ in (ax_e, ax_a):
    a_.axvline(T1, color="k", ls=":"); a_.axvline(T2, color="k", ls=":")
ax_e.text(1, 4.5, "learns body", fontsize=8); ax_e.text(T1 + 1, 4.5, "healthy", fontsize=8); ax_e.text(T2 + 1, 4.5, "damage", fontsize=8)
fig.tight_layout()

X, Y = np.empty((0, 3)), np.empty((0, 2))
errs, uncs, accs, trail, done = {}, {}, {}, [], set()
SNAPS = {1, T1 - 1, T2 - 1, T2 + 2, TOTAL - 1}

def update(f):
    global X, Y
    arm = healthy if f < T2 else damaged
    key = "h" if f < T2 else "d"
    a = pose(f); true = arm.hand(a)[0]
    if f not in done:
        done.add(f)
        if f < T1:                                          # phase 1: learn the body from random practice
            ens.train(healthy, rng, TRAIN_STEPS, 3e-3)
        elif f >= T2:                                       # phase 3: relearn from its own moves
            new = np.vstack([a, rand_angles(rng, 3)])
            X = np.vstack([X, new]); Y = np.vstack([Y, arm.hand(new)])
            ens.train(arm, rng, 20, 2e-3, X, Y)
        errs[f], accs[f] = metrics(key)
        trail.append(true)
    P = ens.preds(a)[:, 0, :]; pred = P.mean(0); unc = float(np.sqrt(P.var(0).sum()))
    uncs[f] = unc
    pts = joint_points(arm, a)
    links.set_data(pts[:, 0], pts[:, 1])
    true_dot.set_data([true[0]], [true[1]]); pred_dot.set_data([pred[0]], [pred[1]])
    tr = np.array(trail[-60:]); trail_line.set_data(tr[:, 0], tr[:, 1])
    ring.center = (pred[0], pred[1]); ring.set_radius(max(5 * unc, 0.02))
    phase = ("1) learning its body from random practice" if f < T1 else
             "2) healthy arm, model is accurate" if f < T2 else
             "3) DAMAGED: segment 2 shortened, relearning")
    title.set_text(f"{phase}\nerror {errs[f]:.3f}    accuracy {accs[f]:.0f}%")
    ks = sorted(errs)
    err_line.set_data(ks, [errs[k] for k in ks]); acc_line.set_data(ks, [accs[k] for k in ks])
    ku = sorted(uncs); unc_line.set_data(ku, [max(uncs[k], 1e-4) for k in ku])
    if args.save and f in SNAPS:
        fig.savefig(f"frame_{f}.png", dpi=80)
    return links, true_dot, pred_dot

anim = FuncAnimation(fig, update, frames=TOTAL, interval=1, repeat=False, blit=False)
if args.save:
    anim.save(args.save, writer=PillowWriter(fps=20), dpi=70)
    m = lambda d, lo, hi: np.mean([d[k] for k in range(max(lo, 0), hi)])
    print(f"error:       learning start {errs[0]:.3f} | healthy {m(errs, T2-10, T2):.3f} | just after damage {m(errs, T2, T2+5):.3f} | end {m(errs, TOTAL-20, TOTAL):.3f}")
    print(f"accuracy %:  healthy {m(accs, T2-10, T2):.0f} | just after damage {m(accs, T2, T2+5):.0f} | end {m(accs, TOTAL-20, TOTAL):.0f}")
    print(f"uncertainty: healthy {m(uncs, T2-10, T2):.4f} | just after damage {m(uncs, T2, T2+5):.4f}")
    print("saved", args.save)
else:
    plt.show()