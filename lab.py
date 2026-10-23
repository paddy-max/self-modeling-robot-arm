import copy, sys, time
import numpy as np

# ---------------- the real body ----------------
class Arm:
    def __init__(self, lengths=(1.0, 0.8, 0.5)):
        self.L = np.array(lengths, float)
    def hand(self, a):                        # a: (n, joints) angles in radians
        c = np.cumsum(a, axis=1)
        return np.stack([(self.L*np.cos(c)).sum(1), (self.L*np.sin(c)).sum(1)], 1)

def rand_angles(rng, n, j=3):
    return rng.uniform(-np.pi, np.pi, (n, j))

# ---------------- one self-model (small neural network) ----------------
class MLP:
    def __init__(self, rng, j=3, h=64):
        self.W = [rng.normal(0, .4, (2*j, h)), rng.normal(0, .4, (h, h)), rng.normal(0, .4, (h, 2))]
        self.b = [np.zeros(h), np.zeros(h), np.zeros(2)]
        self.p = [self.W[0], self.b[0], self.W[1], self.b[1], self.W[2], self.b[2]]
        self.m = [np.zeros_like(q) for q in self.p]; self.v = [np.zeros_like(q) for q in self.p]; self.t = 0
    def fwd(self, a):
        self.x = np.concatenate([np.sin(a), np.cos(a)], 1)
        self.h1 = np.tanh(self.x @ self.W[0] + self.b[0])
        self.h2 = np.tanh(self.h1 @ self.W[1] + self.b[1])
        return self.h2 @ self.W[2] + self.b[2]
    def step(self, a, y, lr):
        out = self.fwd(a); d = 2*(out-y)/len(a)
        g3 = self.h2.T @ d; gb3 = d.sum(0)
        d2 = (d @ self.W[2].T)*(1-self.h2**2); g2 = self.h1.T @ d2; gb2 = d2.sum(0)
        d1 = (d2 @ self.W[1].T)*(1-self.h1**2); g1 = self.x.T @ d1; gb1 = d1.sum(0)
        self.t += 1
        for i, (q, g) in enumerate(zip(self.p, [g1, gb1, g2, gb2, g3, gb3])):
            self.m[i] = .9*self.m[i]+.1*g; self.v[i] = .999*self.v[i]+.001*g*g
            q -= lr*(self.m[i]/(1-.9**self.t))/(np.sqrt(self.v[i]/(1-.999**self.t))+1e-8)

# ---------------- an ensemble: several self-models that can disagree ----------------
class Ensemble:
    def __init__(self, rng, k=5): self.models = [MLP(rng) for _ in range(k)]
    def preds(self, a): return np.stack([m.fwd(a) for m in self.models])   # (k, n, 2)
    def mean(self, a): return self.preds(a).mean(0)
    def uncertainty(self, a): return self.preds(a).var(0).sum(1)           # disagreement
    def train(self, arm, rng, steps, lr, X=None, Y=None):
        for m in self.models:
            for _ in range(steps):
                if X is None:
                    a = rand_angles(rng, 128); m.step(a, arm.hand(a), lr)
                else:
                    m.step(X, Y, lr)

def body_error(ens, arm, test):               # how wrong is the self-model about the body?
    return np.sqrt(np.mean(np.sum((ens.mean(test) - arm.hand(test))**2, 1)))

# ---------------- using the self-model: plan a reach, then check on the real arm ----------------
def plan(ens, target, rng):
    best = rand_angles(rng, 3000); s = np.sum((ens.mean(best)-target)**2, 1)
    cur = best[np.argsort(s)[:20]]
    sigma = 0.5
    for _ in range(25):
        cand = np.vstack([cur, (cur[rng.integers(0, len(cur), 200)] + rng.normal(0, sigma, (200, 3)))])
        s = np.sum((ens.mean(cand)-target)**2, 1); cur = cand[np.argsort(s)[:20]]; sigma *= .8
    return cur[0]

def reach_error(ens, arm, rng, n=30):
    tg = arm.hand(rand_angles(rng, n))        # targets that are really reachable
    return np.mean([np.linalg.norm(arm.hand(plan(ens, t, rng)[None])[0] - t) for t in tg])

# ---------------- adapt after damage: random moves vs uncertainty-guided moves ----------------
def adapt(ens, arm, strategy, rng, test, rounds=16, per_round=5, steps=150, pool=400):
    X = np.empty((0, 3)); Y = np.empty((0, 2)); curve = []; snap = None
    for r in range(rounds):
        if strategy == "random":
            new = rand_angles(rng, per_round)
        else:
            cand = rand_angles(rng, pool)
            new = cand[np.argsort(ens.uncertainty(cand))[-per_round:]]
        X = np.vstack([X, new]); Y = np.vstack([Y, arm.hand(new)])
        ens.train(arm, rng, steps, 2e-3, X, Y)
        curve.append(body_error(ens, arm, test))
        if (r+1)*per_round == 40: snap = copy.deepcopy(ens)
    return curve, snap

if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    res = {"random": [], "active": []}; reach = {k: [] for k in ["healthy", "damaged_old", "random_40", "active_40"]}
    base_err, dmg_err = [], []
    t0 = time.time()
    for s in range(seeds):
        rng = np.random.default_rng(s)
        healthy, damaged = Arm(), Arm((1.0, 0.4, 0.5))          # damage: link 2 is now half as long
        test_h, test_d = rand_angles(rng, 1500), rand_angles(rng, 1500)
        ens = Ensemble(rng); ens.train(healthy, rng, 3000, 3e-3)
        base_err.append(body_error(ens, healthy, test_h)); dmg_err.append(body_error(ens, damaged, test_d))
        reach["healthy"].append(reach_error(ens, healthy, rng)); reach["damaged_old"].append(reach_error(ens, damaged, rng))
        for strat in ("random", "active"):
            curve, snap = adapt(copy.deepcopy(ens), damaged, strat, np.random.default_rng(100+s), test_d)
            res[strat].append(curve); reach[strat+"_40"].append(reach_error(snap, damaged, rng))
        print(f"seed {s} done, {time.time()-t0:.0f}s", flush=True)
    print("\nbody error, healthy arm learned:", np.round(np.mean(base_err), 3))
    print("body error right after damage   :", np.round(np.mean(dmg_err), 3))
    print("\nmoves  random(mean±sd)   active(mean±sd)")
    for i, m in enumerate(range(5, 85, 5)):
        r = np.array(res["random"])[:, i]; a = np.array(res["active"])[:, i]
        if m in (10, 20, 40, 80): print(f"{m:>5}  {r.mean():.3f}±{r.std():.3f}    {a.mean():.3f}±{a.std():.3f}")
    print("\nreach error (distance missed by the real hand):")
    for k, v in reach.items(): print(f"  {k:12s} {np.mean(v):.3f}±{np.std(v):.3f}")