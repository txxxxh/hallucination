#!/usr/bin/env python3
"""Evaluate the user-requested approximately 150-dimensional PCA budget."""
import importlib

E = importlib.import_module("186_pca_budget_sparse_confirmation")
E.PER_LAYER = (18,)
E.OUT = E.M.RUNS / "188_pca_148d_confirmation.json"

def select_scalars(x, residual, limit=4):
    ranked = []
    for j, name in enumerate(E.M.N):
        rho = E.np.corrcoef(x[:, j], residual)[0, 1]
        if E.np.isfinite(rho): ranked.append((abs(rho), float(rho), j, name))
    ranked.sort(reverse=True)
    chosen = []
    for z in ranked:
        if all(abs(E.np.corrcoef(x[:, z[2]], x[:, w[2]])[0, 1]) < .8 for w in chosen):
            chosen.append(z)
            if len(chosen) == limit: break
    return chosen

E.select_scalars = select_scalars
E.main()
