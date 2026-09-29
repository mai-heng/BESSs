#!/usr/bin/env python3
"""LTV vs frozen-average-efficiency experiment; merges into results.json.

Run sim_bess.py first (it produces data/results.json).
Only needs numpy + scipy.optimize.linprog (works on any SciPy version).
"""
import os
import sys
import json
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

try:                                     # __file__ is undefined in notebooks
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
OUT = os.environ.get("SBESS_OUT") or (
    os.path.dirname(HERE) if os.path.basename(HERE) == "code" else HERE)
DATA = os.path.join(OUT, "data")
RES = os.path.join(DATA, "results.json")

# ---------------- LP solver: auto-detect a method the SciPy supports ------
_LP_ORDER = ["highs", "revised simplex", "simplex", "interior-point"]
_LP_METHOD = os.environ.get("SBESS_LP_METHOD")   # None -> auto-detect


def _lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    global _LP_METHOD
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
        return r
    if _LP_METHOD is None:               # last resort: let SciPy choose
        return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    raise RuntimeError("no usable linprog method; last error: %r" % (last,))

N, dt = 24, 1.0
E_cap = 1000.0
E_min, E_max = 0.05 * E_cap, 0.95 * E_cap
u_max = 250.0
E_init = E_f = 0.30 * E_cap
k = np.arange(N)
eta_nom = 0.925 + 0.045 * np.sin(2 * np.pi * (k - 6) / 24.0)
a_nom = 1.0 - (0.0006 + 0.0002 * np.sin(2 * np.pi * (k - 3) / 24.0))
price_nom = np.array([28, 25, 23, 22, 24, 30, 42, 55, 61, 48, 44, 41, 43, 47, 52, 58, 63,
                      88, 104, 118, 96, 78, 52, 38], dtype=float)


def solve_LP_minfuel(price, eta, R):
    pc = price * dt / 1000.0
    nv = 2 * N + (N + 1)
    c = np.zeros(nv); c[N:2 * N] = 1.0
    A_eq = lil_matrix((N, nv)); b_eq = np.zeros(N)
    for i in range(N):
        A_eq[i, i] = dt * eta[i]; A_eq[i, 2 * N + i] = -a_nom[i]; A_eq[i, 2 * N + i + 1] = 1.0
    A_ub = lil_matrix((2 * N + 1, nv)); b_ub = np.zeros(2 * N + 1)
    for i in range(N):
        A_ub[2 * i, N + i] = -1.0; A_ub[2 * i, i] = 1.0
        A_ub[2 * i + 1, N + i] = -1.0; A_ub[2 * i + 1, i] = -1.0
    A_ub[2 * N, 0:N] = -pc; b_ub[2 * N] = -R
    b = [(-u_max, u_max)] * N + [(0.0, None)] * N + [(E_min, E_max)] * (N + 1)
    b[2 * N] = (E_init, E_init); b[2 * N + N] = (E_f, E_f)
    r = _lp(c, A_ub=A_ub.toarray(), b_ub=b_ub, A_eq=A_eq.toarray(), b_eq=b_eq, bounds=b)
    return r.x[0:N].copy()


if not os.path.exists(RES):
    sys.exit("[ltv_check] %s not found.\n"
             "  Run sim_bess.py first (it writes data/results.json)." % RES)

r = json.load(open(RES))
R_op = 0.90 * r["dense"]["revenue"]

u_ltv = solve_LP_minfuel(price_nom, eta_nom, R_op)
eta_bar = np.full(N, float(eta_nom.mean()))
u_avg = solve_LP_minfuel(price_nom, eta_bar, R_op)


def eval_on_true(u):
    E = E_init; feas = True; rev = 0.0; Eminv = E_init; Emaxv = E_init
    for i in range(N):
        E = a_nom[i] * E - dt * eta_nom[i] * u[i]
        Eminv = min(Eminv, E); Emaxv = max(Emaxv, E)
        if E < E_min - 1e-6 or E > E_max + 1e-6:
            feas = False
        rev += price_nom[i] * u[i] * dt / 1000.0
    return dict(revenue=round(rev, 3), feasible=bool(feas),
                terminal=round(float(E), 2), min_E=round(float(Eminv), 1), max_E=round(float(Emaxv), 1))


r["ltv_experiment"] = {
    "R_op": round(float(R_op), 3),
    "ltv_schedule_on_true": eval_on_true(u_ltv),
    "avg_schedule_on_true": eval_on_true(u_avg),
    "terminal_ref": E_f,
}
json.dump(r, open(RES, "w"), indent=2)
print(json.dumps(r["ltv_experiment"], indent=2))
