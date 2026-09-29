#!/usr/bin/env python3
"""
Sparse (maximum hands-off) scheduling of a grid-connected BESS under
time-varying conditions.  Numerical study for ISA Transactions submission.

Plant (DT-LTV):  E_{k+1} = a_k E_k - dt*eta_k*u_k   (E[kWh] state, u[kW] ctrl)
Task: deliver a target arbitrage revenue R over a day.
  * L0 (exact, maximum hands-off):  min #actions s.t. revenue>=R
  * L1 (convex relaxation, min fuel): min sum|u_k| s.t. revenue>=R
Compares action counts (equivalence), revenue/cycle trade-off, baselines,
and robustness over random market/condition profiles.

Dependencies: numpy, scipy, matplotlib.
The exact L0 (minimum-action) solver picks the first backend that works:
  1) scipy.optimize.milp            (SciPy >= 1.9)
  2) PuLP                             (pip install pulp)
  3) built-in branch-and-bound        (only needs scipy.optimize.linprog)
Force a specific backend with the env var SBESS_FORCE_BB=1 (built-in B&B),
or SBESS_BACKEND in {scipy, pulp, bb}.
"""
import os
import sys
import json
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, vstack as spvstack
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- portable paths (works as script AND in Jupyter/Colab) ---
# In a notebook __file__ is undefined; fall back to the current directory.
# Override the output root with the env var SBESS_OUT if you like.
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # running inside Jupyter/IPython
    HERE = os.getcwd()
OUT = os.environ.get("SBESS_OUT") or (
    os.path.dirname(HERE) if os.path.basename(HERE) == "code" else HERE)
FIG = os.path.join(OUT, "figures")
DATA = os.path.join(OUT, "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

# ---------------- exact-L0 backend selection ----------
_FORCE = os.environ.get("SBESS_BACKEND")
if os.environ.get("SBESS_FORCE_BB") == "1":
    _FORCE = "bb"

_HAVE_SCIPY_MILP = False
if _FORCE in (None, "scipy"):
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        _HAVE_SCIPY_MILP = True
    except Exception:
        if _FORCE == "scipy":
            raise

pulp = None
if (not _HAVE_SCIPY_MILP) and _FORCE in (None, "pulp"):
    try:
        import pulp as _pulp  # noqa
        pulp = _pulp
    except Exception:
        pulp = None


def _backend_name():
    if _HAVE_SCIPY_MILP:
        return "scipy.optimize.milp"
    if pulp is not None:
        return "PuLP"
    return "built-in branch-and-bound (linprog)"


# ---------------- LP solver: pick a method the installed SciPy supports ----
# "highs" needs SciPy >= 1.6; older releases use "revised simplex"/"simplex".
# Instead of probing once, we try methods lazily at call time and cache the
# first one that works, so every SciPy version is handled automatically.
_LP_ORDER = ["highs", "revised simplex", "simplex", "interior-point"]
_LP_METHOD = os.environ.get("SBESS_LP_METHOD")   # None -> auto-detect
_LP_ANNOUNCED = False


def _lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """linprog wrapper; prefers a simplex method so vertex (sparse) solutions
    are returned -- essential for the min-fuel L1 relaxation."""
    global _LP_METHOD, _LP_ANNOUNCED
    order = ([_LP_METHOD] if _LP_METHOD else []) + \
            [m for m in _LP_ORDER if m != _LP_METHOD]
    last = None
    for m in order:
        try:
            r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method=m)
        except Exception as e:           # e.g. ValueError: Unknown solver
            last = e
            continue
        _LP_METHOD = m                   # this method exists -> use it
        if not _LP_ANNOUNCED:
            print("[sim_bess] linear-programming method (auto-detected):", m)
            _LP_ANNOUNCED = True
        return r
    if _LP_METHOD is None:               # last resort: let SciPy choose
        return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    raise RuntimeError("no usable linprog method; last error: %r" % (last,))


# ---------------- nominal data ----------------
N, dt = 24, 1.0
E_cap = 1000.0
E_min, E_max = 0.05 * E_cap, 0.95 * E_cap
u_max = 250.0                       # 0.25C -> 4h battery
E_init = E_f = 0.30 * E_cap

k = np.arange(N)
eta_nom = 0.925 + 0.045 * np.sin(2 * np.pi * (k - 6) / 24.0)
a_nom = 1.0 - (0.0006 + 0.0002 * np.sin(2 * np.pi * (k - 3) / 24.0))
price_nom = np.array([28, 25, 23, 22, 24, 30, 42, 55, 61, 48, 44, 41, 43, 47, 52, 58, 63,
                      88, 104, 118, 96, 78, 52, 38], dtype=float)


# ---------------- solvers ----------------
def _dyn_rows(nv, eta, a, off_e):
    A = lil_matrix((N, nv))
    for i in range(N):
        A[i, i] = dt * eta[i]; A[i, off_e + i] = -a[i]; A[i, off_e + i + 1] = 1.0
    return A.tocsr()


def solve_LP_maxrev(price, eta, a):
    pc = price * dt / 1000.0
    nv = 2 * N + (N + 1)
    c = np.zeros(nv); c[0:N] = -pc
    A_eq = _dyn_rows(nv, eta, a, 2 * N); b_eq = np.zeros(N)
    A_ub = lil_matrix((2 * N, nv)); b_ub = np.zeros(2 * N)
    for i in range(N):
        A_ub[2 * i, N + i] = -1.0; A_ub[2 * i, i] = 1.0
        A_ub[2 * i + 1, N + i] = -1.0; A_ub[2 * i + 1, i] = -1.0
    b = [(-u_max, u_max)] * N + [(0.0, None)] * N + [(E_min, E_max)] * (N + 1)
    b[2 * N] = (E_init, E_init); b[2 * N + N] = (E_f, E_f)
    r = _lp(c, A_ub=A_ub.toarray(), b_ub=b_ub, A_eq=A_eq.toarray(), b_eq=b_eq, bounds=b)
    return r.x[0:N].copy(), r.x[2 * N:2 * N + N + 1].copy()


def solve_LP_minfuel(price, eta, a, R):
    """L1 relaxation: min sum|u| s.t. revenue>=R."""
    pc = price * dt / 1000.0
    nv = 2 * N + (N + 1)
    c = np.zeros(nv); c[N:2 * N] = 1.0
    A_eq = _dyn_rows(nv, eta, a, 2 * N); b_eq = np.zeros(N)
    A_ub = lil_matrix((2 * N + 1, nv)); b_ub = np.zeros(2 * N + 1)
    for i in range(N):
        A_ub[2 * i, N + i] = -1.0; A_ub[2 * i, i] = 1.0
        A_ub[2 * i + 1, N + i] = -1.0; A_ub[2 * i + 1, i] = -1.0
    A_ub[2 * N, 0:N] = -pc; b_ub[2 * N] = -R
    b = [(-u_max, u_max)] * N + [(0.0, None)] * N + [(E_min, E_max)] * (N + 1)
    b[2 * N] = (E_init, E_init); b[2 * N + N] = (E_f, E_f)
    r = _lp(c, A_ub=A_ub.toarray(), b_ub=b_ub, A_eq=A_eq.toarray(), b_eq=b_eq, bounds=b)
    return r.x[0:N].copy(), r.x[2 * N:2 * N + N + 1].copy()


def _milp_data(price, eta, a, R):
    """Assemble the (u, z, E) MILP: min sum z  s.t. u_k <= u_max z_k, -u_k <= u_max z_k."""
    pc = price * dt / 1000.0
    nv = 3 * N + 1
    c = np.zeros(nv); c[N:2 * N] = 1.0                     # cost = number of active hours
    A_eq = lil_matrix((N, nv)); b_eq = np.zeros(N)
    for i in range(N):
        A_eq[i, i] = dt * eta[i]; A_eq[i, 2 * N + i] = -a[i]; A_eq[i, 2 * N + i + 1] = 1.0
    A_ub = lil_matrix((2 * N + 1, nv)); b_ub = np.zeros(2 * N + 1)
    for i in range(N):
        A_ub[2 * i, i] = 1.0; A_ub[2 * i, N + i] = -u_max
        A_ub[2 * i + 1, i] = -1.0; A_ub[2 * i + 1, N + i] = -u_max
    A_ub[2 * N, 0:N] = -pc; b_ub[2 * N] = -R
    lb = np.empty(nv); ub = np.empty(nv)
    lb[0:N], ub[0:N] = -u_max, u_max
    lb[N:2 * N], ub[N:2 * N] = 0.0, 1.0
    lb[2 * N:], ub[2 * N:] = E_min, E_max
    lb[2 * N] = ub[2 * N] = E_init
    lb[2 * N + N] = ub[2 * N + N] = E_f
    return pc, nv, c, A_eq.tocsr(), A_ub.tocsr(), b_ub, lb, ub


def _milp_scipy(price, eta, a, R):
    """Backend 1: scipy.optimize.milp (SciPy >= 1.9)."""
    pc, nv, c, A_eq, A_ub, b_ub, lb, ub = _milp_data(price, eta, a, R)
    integ = np.zeros(nv); integ[N:2 * N] = 1
    A = spvstack([A_eq, A_ub]).tocsr()
    lower = np.concatenate([np.zeros(N), np.full(2 * N + 1, -np.inf)])
    upper = np.concatenate([np.zeros(N), b_ub])
    r = milp(c, integrality=integ, bounds=Bounds(lb, ub),
             constraints=LinearConstraint(A, lower, upper),
             options={"time_limit": 60, "mip_rel_gap": 1e-4})
    if not r.success:
        return None
    return r.x[0:N].copy(), r.x[2 * N:].copy()


def _milp_pulp(price, eta, a, R):
    """Backend 2: PuLP (any installed CBC)."""
    import pulp as pl
    pc = price * dt / 1000.0
    prob = pl.LpProblem("min_actions", pl.LpMinimize)
    u = [pl.LpVariable("u%d" % i, -u_max, u_max) for i in range(N)]
    z = [pl.LpVariable("z%d" % i, 0, 1, cat="Binary") for i in range(N)]
    E = [pl.LpVariable("E%d" % i, E_min, E_max) for i in range(N + 1)]
    prob += pl.lpSum(z)
    for i in range(N):
        prob += E[i + 1] == a[i] * E[i] - dt * eta[i] * u[i]
        prob += u[i] <= u_max * z[i]
        prob += -u[i] <= u_max * z[i]
    prob += pl.lpSum(pc[i] * u[i] for i in range(N)) >= R
    prob += E[0] == E_init
    prob += E[N] == E_f
    prob.solve(pl.PULP_CBC_CMD(msg=0))
    if pl.LpStatus[prob.status] != "Optimal":
        return None
    uu = np.array([pl.value(v) for v in u])
    EE = np.array([pl.value(v) for v in E])
    return uu, EE


def _milp_bb(price, eta, a, R, node_limit=200000, incumbent=None):
    """Backend 3: built-in branch-and-bound over the z_k, using linprog only."""
    pc, nv, c, A_eq, A_ub, b_ub, lb0, ub0 = _milp_data(price, eta, a, R)
    best_fun, best_u, best_E = np.inf, None, None
    if incumbent is not None:                 # seed with a feasible L1 solution
        u0, E0 = incumbent
        best_fun = float(np.sum(np.abs(u0) > 1e-6))
        best_u, best_E = np.asarray(u0, float).copy(), np.asarray(E0, float).copy()
    stack = [(np.zeros(N), np.ones(N))]
    nodes = 0
    while stack:
        zlb, zub = stack.pop()
        nodes += 1
        if nodes > node_limit:
            break
        lb = lb0.copy(); ub = ub0.copy()
        lb[N:2 * N] = zlb; ub[N:2 * N] = zub
        r = _lp(c, A_ub=A_ub.toarray(), b_ub=b_ub, A_eq=A_eq.toarray(), b_eq=np.zeros(N),
                bounds=list(zip(lb.tolist(), ub.tolist())))
        if not r.success or r.fun >= best_fun - 1e-9:
            continue
        z = r.x[N:2 * N]
        frac = np.where((z > 1e-6) & (z < 1.0 - 1e-6))[0]
        if len(frac) == 0:                                  # integral -> incumbent
            best_fun = r.fun
            best_u = r.x[0:N].copy()
            best_E = r.x[2 * N:].copy()
            continue
        j = int(frac[np.argmin(np.abs(z[frac] - 0.5))])     # branch on most fractional
        for val in (1.0, 0.0):                              # try z_j=1 first (fewer actions)
            zl = zlb.copy(); zu = zub.copy()
            zl[j] = zu[j] = val
            stack.append((zl, zu))
    if best_u is None:
        return None
    return best_u, best_E


def solve_MILP_minact(price, eta, a, R):
    """Exact L0: min #actions s.t. revenue>=R (dispatches to an available backend)."""
    if _HAVE_SCIPY_MILP:
        return _milp_scipy(price, eta, a, R)
    if pulp is not None:
        return _milp_pulp(price, eta, a, R)
    u1, E1 = solve_LP_minfuel(price, eta, a, R)   # feasible seed -> strong bound
    return _milp_bb(price, eta, a, R, incumbent=(u1, E1))


# ---------------- metrics ----------------
def rainflow(turning):
    stack, cycles = [], []
    for x in turning:
        stack.append(x)
        while len(stack) >= 3:
            A, B, C = stack[-3], stack[-2], stack[-1]
            if (B - A) * (C - B) < 0:
                if abs(B - A) <= abs(C - B):
                    cycles.append(abs(B - A)); del stack[-2]; del stack[-2]
                else:
                    break
            else:
                break
    for i in range(len(stack) - 1):
        cycles.append(abs(stack[i + 1] - stack[i]))
    return cycles


def turning_points(series):
    s = [series[0]]
    for x in series[1:]:
        if abs(x - s[-1]) > 1e-9:
            s.append(x)
    if len(s) < 3:
        return s + [series[-1]]
    tp = [s[0]]
    for i in range(1, len(s) - 1):
        if (s[i] - s[i - 1]) * (s[i + 1] - s[i]) < 0:
            tp.append(s[i])
    tp.append(s[-1])
    return tp


def metrics(u, E, price, eps=1.0):
    act = np.where(np.abs(u) > eps)[0]
    n_act = int(len(act))
    n_sw = int(np.sum(np.sign(u[act])[1:] != np.sign(u[act])[:-1])) if n_act > 1 else 0
    revenue = float(np.sum(price * u * dt) / 1000.0)
    throughput = float(np.sum(np.abs(u)) * dt) / 1000.0
    efc = throughput / (2.0 * (E_cap / 1000.0))
    cyc = rainflow(turning_points(E))
    dod = np.array([c / E_cap for c in cyc])
    damage = float(np.sum(5.24e-4 * dod ** 2.03)) if len(dod) else 0.0
    return dict(n_actions=n_act, n_switches=n_sw, revenue=round(revenue, 3),
                throughput_MWh=round(throughput, 3), efc=round(efc, 3),
                damage=round(damage, 5))


def greedy_tou(price, eta):
    """TOU heuristic, terminal-feasible (E_N = E_init)."""
    u = np.zeros(N); E = E_init
    order = np.argsort(price)
    for i in sorted(order[:6].tolist()):                     # charge 6 cheapest
        p = min(u_max, (E_max - E) / (dt * eta[i])); u[i] = -p; E += dt * eta[i] * p
    for i in sorted(order[-9:].tolist(), key=lambda j: -price[j]):  # discharge to E_init
        if E <= E_init + 1e-9:
            break
        if u[i] != 0:
            continue
        p = min(u_max, (E - E_init) / (dt * eta[i])); u[i] = p; E -= dt * eta[i] * p
    Eg = [E_init]; E = E_init
    for i in range(N):
        E = E - dt * eta[i] * u[i]; Eg.append(E)
    return u, np.array(Eg)


# ---------------- nominal study ----------------
print("[sim_bess] exact-L0 backend:", _backend_name())
print("[sim_bess] linear-programming method:", _LP_METHOD)
res = {}
u_d, E_d = solve_LP_maxrev(price_nom, eta_nom, a_nom)
m_d = metrics(u_d, E_d, price_nom); res["dense"] = m_d
rev_d = m_d["revenue"]

u_g, E_g = greedy_tou(price_nom, eta_nom); res["greedy"] = metrics(u_g, E_g, price_nom)

fracs = [0.99, 0.97, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
front = []
for f in fracs:
    R = f * rev_d
    o0 = solve_MILP_minact(price_nom, eta_nom, a_nom, R)
    o1 = solve_LP_minfuel(price_nom, eta_nom, a_nom, R)
    row = dict(frac=f, R=round(R, 3))
    if o0 is not None:
        row.update({("L0_" + kk): vv for kk, vv in metrics(*o0, price_nom).items()})
    row.update({("L1_" + kk): vv for kk, vv in metrics(*o1, price_nom).items()})
    front.append(row)
res["front"] = front

# operating point: 90% revenue, L1 (min-fuel) schedule
R_op = 0.90 * rev_d
u_op, E_op = solve_LP_minfuel(price_nom, eta_nom, a_nom, R_op)
res["op"] = metrics(u_op, E_op, price_nom)

# ---------------- robustness over random days ----------------
rng = np.random.default_rng(0)
rob = []
for s in range(24):
    pr = price_nom.copy()
    pr = pr * (0.8 + 0.4 * rng.random())                 # level shift
    pk = rng.integers(16, 21)                             # peak hour
    pr = pr + 25 * np.exp(-0.5 * ((k - pk) / 1.5) ** 2)   # peak boost
    pr = np.clip(pr, 10, None)
    ph = rng.integers(0, 6)
    et = 0.925 + 0.045 * np.sin(2 * np.pi * (k - ph) / 24.0)
    aa = 1.0 - (0.0006 + 0.0002 * np.sin(2 * np.pi * (k - rng.integers(0, 24)) / 24.0))
    ud, Ed = solve_LP_maxrev(pr, et, aa); rd = metrics(ud, Ed, pr)["revenue"]
    o0 = solve_MILP_minact(pr, et, aa, 0.90 * rd)
    o1 = solve_LP_minfuel(pr, et, aa, 0.90 * rd)
    if o0 is None:
        continue
    ma0 = metrics(*o0, pr); ma1 = metrics(*o1, pr)
    rob.append(dict(rev_dense=round(rd, 2), L0_act=ma0["n_actions"], L1_act=ma1["n_actions"],
                    gap=ma1["n_actions"] - ma0["n_actions"]))
res["robustness"] = rob

print(json.dumps(res, indent=2))

np.savez(os.path.join(DATA, "schedules.npz"), u_op=u_op, E_op=E_op, u_dense=u_d,
         E_dense=E_d, u_greedy=u_g, E_greedy=E_g, price=price_nom, eta=eta_nom, a=a_nom)
with open(os.path.join(DATA, "results.json"), "w") as f:
    json.dump(res, f, indent=2)

# ---------------- figures ----------------
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "ps.fonttype": 42, "pdf.fonttype": 42})


def _save(fig, name):
    for ext in ("pdf", "png", "eps"):
        fig.savefig(os.path.join(FIG, name + "." + ext))


fig, ax1 = plt.subplots(figsize=(7, 3.0))
ax1.step(k, price_nom, where="mid", color="C0", lw=1.8); ax1.set_xlabel("time [h]")
ax1.set_ylabel("price [\\$/MWh]", color="C0"); ax1.tick_params(axis="y", labelcolor="C0")
ax2 = ax1.twinx(); ax2.grid(False); ax2.plot(k, eta_nom, color="C3", lw=1.8, marker="o", ms=3)
ax2.set_ylabel("efficiency $\\eta_k$", color="C3"); ax2.tick_params(axis="y", labelcolor="C3")
fig.tight_layout(); _save(fig, "fig_profiles"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 4))
ax.plot([r["L0_n_actions"] for r in front], [r["L0_revenue"] for r in front],
        "o-", color="C0", ms=6, label="exact $L_0$ (MILP)")
ax.plot([r["L1_n_actions"] for r in front], [r["L1_revenue"] for r in front],
        "x", color="C3", ms=8, label="$L_1$ relaxation (LP)")
ax.axhline(rev_d, ls="--", color="gray", lw=1, label="unconstrained optimum")
ax.set_xlabel("number of control actions"); ax.set_ylabel("arbitrage revenue [\\$]")
ax.legend(fontsize=9); fig.tight_layout()
_save(fig, "fig_front"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.step(k, price_nom, where="mid", color="0.75", lw=1.2, label="price")
ax.bar(k - 0.2, u_d, width=0.4, color="C0", alpha=0.7, label="dense (max revenue)")
ax.bar(k + 0.2, u_op, width=0.4, color="C3", alpha=0.9, label="sparse (proposed)")
ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("time [h]"); ax.set_ylabel("battery power $u_k$ [kW]")
ax.legend(ncol=3, fontsize=9); fig.tight_layout()
_save(fig, "fig_dispatch"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.step(range(N + 1), E_d, where="post", color="C0", lw=1.8, label="dense")
ax.step(range(N + 1), E_op, where="post", color="C3", lw=1.8, label="sparse (proposed)")
ax.step(range(N + 1), E_g, where="post", color="C2", lw=1.3, ls="--", label="TOU greedy")
ax.axhline(E_max, color="0.6", ls=":", lw=1); ax.axhline(E_min, color="0.6", ls=":", lw=1)
ax.set_xlabel("time [h]"); ax.set_ylabel("stored energy $E_k$ [kWh]")
ax.legend(fontsize=9); fig.tight_layout()
_save(fig, "fig_soc"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.2))
gap = [r["gap"] for r in rob]
ax.hist(gap, bins=np.arange(min(gap) - 0.5, max(gap) + 1.5), color="C0", alpha=0.8, edgecolor="k")
ax.set_xlabel("$L_1$ actions $-$ $L_0$ actions"); ax.set_ylabel("# instances")
fig.tight_layout(); _save(fig, "fig_robust"); plt.close(fig)
print("DONE")
