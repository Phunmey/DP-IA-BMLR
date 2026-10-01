import os
import sys
import arviz as az

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from model_dp_ia_bmlr import DPIABMLR


model_specs = {"dp_ia_bmlr": dict(H_trunc=8, weighted=True), "dp_bmlr": dict(H_trunc=8, weighted=False),
               "ia_bmlr": dict(H_trunc=1, weighted=True), "u_bmlr": dict(H_trunc=1, weighted=False)}


def fit_and_extract(X, y, X_test, model_name, sampling, fixed_alpha=None, alpha_prior_a=2.0, alpha_prior_b=2.0, seed=0):
    if model_name not in model_specs:
        raise ValueError(f"unknown model_name {model_name}; valid: {list(model_specs)}")
    mspec = model_specs[model_name]
    H_trunc = mspec["H_trunc"]
    weighted = mspec["weighted"]
    np.random.seed(seed)

    model = DPIABMLR(H_trunc=H_trunc, weighted=weighted, fixed_alpha=fixed_alpha, alpha_prior_a=alpha_prior_a,
                     alpha_prior_b=alpha_prior_b, **sampling)
    model.fit(X, y, verbose=False, save_trace=False)

    proba_test_samples = model._get_posterior_samples(X_test, use_training_cache=False)
    proba_test_mean = proba_test_samples.mean(axis=0)

    diag = model.cluster_diagnostics()
    conv = _convergence_diagnostics(getattr(model, "trace", None))

    converged = None
    if conv["max_rhat"] is not None:
        converged = bool(conv["max_rhat"] < 1.01 and (conv["min_ess_bulk"] is None or conv["min_ess_bulk"] > 400)
                         and (conv["n_divergences"] or 0) == 0)

    beta_draws = None
    betaI_draws = None
    try:
        post = model.trace.posterior
        if "beta" in post and "betaI" in post:
            bI = post["betaI"].values
            bB = post["beta"].values
            betaI_draws = bI.reshape((-1,) + bI.shape[2:])
            beta_draws = bB.reshape((-1,) + bB.shape[2:])
    except Exception:
        pass

    return {"proba_test_samples": proba_test_samples, "proba_test_mean": proba_test_mean, "diagnostics": diag,
            "learned_gamma": model.learned_gamma, "learned_alpha": model.learned_alpha, "converged": converged,
            "convergence": conv, "betaI_draws": betaI_draws, "beta_draws": beta_draws}


def _convergence_diagnostics(trace):
    out = {"max_rhat": None, "min_ess_bulk": None, "min_ess_tail": None, "n_divergences": None, "params_checked": [],
           "omega_max_rhat": None, "omega_min_ess_bulk": None}
    if trace is None:
        return out
    try:
        post = trace.posterior
        primary = [v for v in ("alpha", "gamma") if v in post]
        out["params_checked"] = primary
        if primary:
            sp = az.summary(trace, var_names=primary, round_to=None)
            if "r_hat" in sp:
                out["max_rhat"] = float(sp["r_hat"].max())
            if "ess_bulk" in sp:
                out["min_ess_bulk"] = float(sp["ess_bulk"].min())
            if "ess_tail" in sp:
                out["min_ess_tail"] = float(sp["ess_tail"].min())
        if "omega" in post:
            so = az.summary(trace, var_names=["omega"], round_to=None)
            if "r_hat" in so:
                out["omega_max_rhat"] = float(so["r_hat"].max())
            if "ess_bulk" in so:
                out["omega_min_ess_bulk"] = float(so["ess_bulk"].min())
        if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
            out["n_divergences"] = int(trace.sample_stats["diverging"].values.sum())
    except Exception as e:
        out["error"] = str(e)
    return out
