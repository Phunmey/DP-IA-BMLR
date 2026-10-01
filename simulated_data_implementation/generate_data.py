import numpy as np


def _softmax(eta):
    eta = eta - eta.max(axis=-1, keepdims=True)
    e = np.exp(eta)
    return e / e.sum(axis=-1, keepdims=True)


class Datagen:
    def __init__(self, omega, intercepts, slopes, K, p, gate=None):
        self.omega = np.asarray(omega, float)
        self.intercepts = np.asarray(intercepts, float)
        self.slopes = np.asarray(slopes, float)
        self.K = K
        self.p = p
        self.C_true = len(self.omega)
        self.gate = gate

    def cluster_proba(self, X):
        X = np.asarray(X, float)
        eta = self.intercepts[None, :, :] + np.einsum("np,hkp--nhk", X, self.slopes)
        return _softmax(eta)

    def mixing_weights(self, X):
        X = np.asarray(X, float)
        N = X.shape[0]
        if self.gate is None:
            return np.tile(self.omega, (N, 1))
        a, b = self.gate
        return _softmax(a[None, :] + X @ b.T)

    def mixture_proba(self, X):
        pc = self.cluster_proba(X)
        w = self.mixing_weights(X)
        return np.einsum("nh,nhk--nk", w, pc)


def make_truth(K, p, C_true, separation, imbalance, rng):
    intercepts = np.zeros((C_true, K))
    slopes = np.zeros((C_true, K, p))
    for h in range(C_true):
        for k in range(1, K):                          # k = 0 is the reference
            intercepts[h, k] = -imbalance + rng.normal(0, 0.5)
            slopes[h, k, :] = rng.normal(0, separation, size=p)
    return intercepts, slopes


def make_gate(C_true, p, gate_strength, rng):
    a = np.zeros(C_true)
    b = np.zeros((C_true, p))
    for h in range(1, C_true):
        a[h] = rng.normal(0, 0.5)
        b[h] = rng.normal(0, gate_strength, size=p)
    return a, b


def _draw(spec, seed):
    rng = np.random.default_rng(seed)
    K, p, C_true = spec["K"], spec["p"], spec["C_true"]
    N = spec["n_train"]
    mixing = spec.get("mixing", "fixed")

    omega = np.asarray(spec["omega_true"], float)
    omega = omega / omega.sum()
    intercepts, slopes = make_truth(K, p, C_true, spec["separation"], spec["imbalance"], rng)

    if mixing == "covariate":
        gate = make_gate(C_true, p, spec.get("gate_strength", 1.5), rng)
        datagen = Datagen(omega, intercepts, slopes, K, p, gate=gate)
        X = rng.normal(size=(N, p))
        w = datagen.mixing_weights(X)
        c = np.array([rng.choice(C_true, p=w[i]) for i in range(N)])
    else:
        datagen = Datagen(omega, intercepts, slopes, K, p, gate=None)
        c = rng.choice(C_true, size=N, p=omega)
        X = rng.normal(size=(N, p))

    pc = datagen.cluster_proba(X)                           # (N, H, K)
    probs = pc[np.arange(N), c, :]                         # (N, K) row per obs
    y = np.array([rng.choice(K, p=probs[i]) for i in range(N)])

    return X, y, c, datagen, rng


def generate_dataset(spec, seed, n_test_grid=400):
    K, C_true = spec["K"], spec["C_true"]
    X, y, c, datagen, rng = _draw(spec, seed)
    tries = 0
    while len(np.unique(y)) < K and tries < 20:
        tries += 1
        X, y, c, datagen, rng = _draw(spec, seed + 10_000 * (tries + 1))

    X_test = rng.normal(size=(n_test_grid, spec["p"]))
    pi_test = datagen.mixture_proba(X_test)
    rng_lab = np.random.default_rng(seed + 987_654)
    y_test = np.array([rng_lab.choice(K, p=pi_test[i]) for i in range(n_test_grid)])

    counts = np.bincount(y, minlength=K)
    info = {"class_counts": counts.tolist(),
            "imbalance_ratio": float(counts.max() / max(counts.min(), 1)),
            "true_cluster_sizes": np.bincount(c, minlength=C_true).tolist(),
            "C_true": C_true,
            "mixing": spec.get("mixing", "fixed"),
            "gate_strength": (float(spec.get("gate_strength", 0.0)) if spec.get("mixing", "fixed") == "covariate" else 0.0),
            "omega_true": np.asarray(datagen.omega).tolist(),
            "y_test": y_test.tolist()}

    return X, y, datagen, X_test, pi_test, info
