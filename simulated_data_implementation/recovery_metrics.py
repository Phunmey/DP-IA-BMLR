import numpy as np


def hdi_interval(samples, prob=0.95):
    x = np.sort(np.asarray(samples, float))
    n = x.size
    if n == 0:
        return (np.nan, np.nan)
    n_in = max(1, int(np.floor(prob * n)))
    if n_in >= n:
        return (x[0], x[-1])
    widths = x[n_in:] - x[: n - n_in]
    j = int(np.argmin(widths))
    return (x[j], x[j + n_in])


def predictive_errors(proba_test_mean, pi_test):
    d = np.asarray(proba_test_mean, float) - np.asarray(pi_test, float)
    return {"predictive_rmse": float(np.sqrt(np.mean(d ** 2))), "predictive_mae": float(np.mean(np.abs(d)))}


def predictive_coverage(proba_test_samples, pi_test, prob=0.95):
    MC, N, K = proba_test_samples.shape
    covered = 0
    total = N * K
    width_sum = 0.0
    for n in range(N):
        for k in range(K):
            lo, hi = hdi_interval(proba_test_samples[:, n, k], prob)
            width_sum += (hi - lo)
            if lo <= pi_test[n, k] <= hi:
                covered += 1
    return {"predictive_coverage": covered / total, "mean_hdi_width": width_sum / total, "nominal": prob}


def interval_score(proba_test_samples, pi_test, prob=0.95):
    MC, N, K = proba_test_samples.shape
    alpha = 1.0 - prob
    total = N * K
    s = 0.0
    for n in range(N):
        for k in range(K):
            lo, hi = hdi_interval(proba_test_samples[:, n, k], prob)
            y = pi_test[n, k]
            score = (hi - lo)
            if y < lo:
                score += (2.0 / alpha) * (lo - y)
            elif y > hi:
                score += (2.0 / alpha) * (y - hi)
            s += score
    return {"interval_score": s / total}


def total_variation(proba_test_mean, pi_test):
    P = np.asarray(proba_test_mean, float)
    Q = np.asarray(pi_test, float)
    tv = 0.5 * np.sum(np.abs(P - Q), axis=1)
    return {"total_variation": float(np.mean(tv))}


def gmean_from_proba(proba_mean, y_true, K):
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(proba_mean, float).argmax(axis=1)
    recalls = []
    for k in range(K):
        mask = (y_true == k)
        if mask.sum() == 0:
            continue
        recalls.append((y_pred[mask] == k).mean())
    if not recalls:
        return float("nan")
    recalls = np.asarray(recalls, float)
    if np.any(recalls == 0):
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


def score_replicate(artifacts, pi_test, C_true, prob=0.95, y_test=None, K=None, datagen=None):
    out = {}
    out.update(predictive_errors(artifacts["proba_test_mean"], pi_test))
    out.update(predictive_coverage(artifacts["proba_test_samples"], pi_test, prob))
    out.update(interval_score(artifacts["proba_test_samples"], pi_test, prob))
    out.update(total_variation(artifacts["proba_test_mean"], pi_test))

    if y_test is not None and K is not None:
        out["gmean"] = gmean_from_proba(artifacts["proba_test_mean"], y_test, K)

    diag = artifacts["diagnostics"]
    out["weight_perplexity"] = diag["weight_perplexity"]
    out["n_occupied"] = diag["n_occupied"]
    out["tail_mass"] = diag["tail_mass"]
    out["H_true"] = C_true
    out["occupied_matches_true"] = int(diag["n_occupied"] == C_true)
    out["perplexity_rounds_to_true"] = int(round(diag["weight_perplexity"]) == C_true)

    out["learned_gamma"] = artifacts["learned_gamma"]
    out["learned_alpha"] = artifacts["learned_alpha"]
    out["converged"] = artifacts["converged"]
    conv = artifacts.get("convergence", {}) or {}
    out["max_rhat"] = conv.get("max_rhat")
    out["min_ess_bulk"] = conv.get("min_ess_bulk")
    out["min_ess_tail"] = conv.get("min_ess_tail")
    out["n_divergences"] = conv.get("n_divergences")
    out["aligned_coef_rmse"] = None
    out["matched_pairs"] = None
    if (datagen is not None and artifacts.get("beta_draws") is not None
            and artifacts.get("betaI_draws") is not None):
        try:
            occ_mask = None
            diag = artifacts.get("diagnostics", {})
            wts = diag.get("regime_weights")
            if wts is not None:
                wts = np.asarray(wts, float)
                occ_mask = wts > (0.05 / len(wts))
            res = aligned_coefficient_rmse(
                artifacts["betaI_draws"], artifacts["beta_draws"],
                datagen, ref_class=0, occupied_mask=occ_mask)
            if res is not None:
                out["aligned_coef_rmse"] = res["aligned_coef_rmse"]
                out["matched_pairs"] = res["matched_pairs"]
        except Exception:
            pass

    return out


def aligned_coefficient_rmse(model_trace_betaI, model_trace_beta, datagen, ref_class=0, occupied_mask=None):
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return None

    H = model_trace_betaI.shape[1]
    H_true = datagen.H_true
    K = datagen.K
    bI = model_trace_betaI.mean(0)
    bB = model_trace_beta.mean(0)

    fitted = list(range(H))
    if occupied_mask is not None:
        fitted = [h for h in range(H) if occupied_mask[h]]
        if not fitted:
            return {"aligned_coef_rmse": float("nan"), "matched_pairs": 0}

    def full_vec(h):
        fI = np.insert(bI[h], ref_class, 0.0)
        fB = np.insert(bB[h], ref_class, 0.0, axis=0)
        return np.concatenate([fI.ravel(), fB.ravel()])

    def true_vec(hp):
        return np.concatenate([datagen.intercepts[hp].ravel(),
                               datagen.slopes[hp].ravel()])

    cost = np.zeros((len(fitted), H_true))
    for i, h in enumerate(fitted):
        fv = full_vec(h)
        for hp in range(H_true):
            cost[i, hp] = np.sqrt(np.mean((fv - true_vec(hp)) ** 2))
    rows, cols = linear_sum_assignment(cost)

    sq = []
    for i, hp in zip(rows, cols):
        h = fitted[i]
        sq.append(np.mean((full_vec(h) - true_vec(hp)) ** 2))
    rmse = float(np.sqrt(np.mean(sq))) if sq else float("nan")
    return {"aligned_coef_rmse": rmse, "matched_pairs": len(rows)}


def aligned_coefficient_coverage(model_trace_betaI, model_trace_beta, datagen, ref_class=0, prob=0.95):
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return None

    H = model_trace_betaI.shape[1]
    C_true = datagen.C_true
    bI = model_trace_betaI.mean(0)
    bB = model_trace_beta.mean(0)

    K = datagen.K
    cost = np.zeros((H, C_true))
    for h in range(H):
        fI = np.insert(bI[h], ref_class, 0.0)            # (K,)
        fB = np.insert(bB[h], ref_class, 0.0, axis=0)    # (K, p)
        for hp in range(C_true):
            d = np.concatenate([(fI - datagen.intercepts[hp]).ravel(), (fB - datagen.slopes[hp]).ravel()])
            cost[h, hp] = np.sqrt(np.sum(d ** 2))
    rows, cols = linear_sum_assignment(cost)

    covered, total = 0, 0
    for h, hp in zip(rows, cols):
        for k in range(K):
            if k == ref_class:
                continue
            kk = k - (1 if k > ref_class else 0)
            lo, hi = hdi_interval(model_trace_betaI[:, h, kk], prob)
            if lo <= datagen.intercepts[hp, k] <= hi:
                covered += 1
            total += 1
            for j in range(datagen.p):
                lo, hi = hdi_interval(model_trace_beta[:, h, kk, j], prob)
                if lo <= datagen.slopes[hp, k, j] <= hi:
                    covered += 1
                total += 1
    return {"aligned_coef_coverage": covered / total if total else np.nan, "matched_pairs": len(rows)}
