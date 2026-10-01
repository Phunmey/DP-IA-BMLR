import os
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt

from model_ia_bmlr import IABMLR
from les_computation import compute_normalized_les
from weights_computation import compute_class_weights

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class DPIABMLR(IABMLR):
    def __init__(self, H_trunc=10, alpha_prior_a=6.0, alpha_prior_b=1.0, fixed_alpha=None,
                 prior_sigma=1.0, gamma_prior_sigma=1.0, les_neighbors=10,
                 weighted=True, fixed_gamma=None,
                 n_samples=2000, n_tune=1000, n_chains=2, cores=None, target_accept=0.95):
        super().__init__(
            prior_sigma=prior_sigma, gamma_prior_sigma=gamma_prior_sigma,
            les_neighbors=les_neighbors, n_samples=n_samples, n_tune=n_tune,
            n_chains=n_chains, cores=cores, target_accept=target_accept)
        self.H_trunc = H_trunc
        self.alpha_prior_a = alpha_prior_a
        self.alpha_prior_b = alpha_prior_b
        self.fixed_alpha = fixed_alpha       # None -> Gamma prior; float -> alpha held fixed
        self.learned_alpha = None
        self.weighted = weighted
        self.fixed_gamma = fixed_gamma

    def fit(self, X, y, verbose=True, save_trace=True, data_dir="./outputs/results",
            plot_dir="./outputs/plots", filename_prefix="dp_ia_bmlr"):
        X = np.asarray(X)
        y = np.asarray(y)
        self._X_train = X
        self._y_train = y

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]
        N = len(y)

        self.label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self.idx_to_label = {i: c for c, i in self.label_to_idx.items()}
        y_idx = np.array([self.label_to_idx[yi] for yi in y])

        unique, counts = np.unique(y_idx, return_counts=True)
        self.ref_class = unique[np.argmax(counts)]

        self.les_scores, _, _ = compute_normalized_les(X, y, n_neighbors=self.les_neighbors)
        self.class_weights, _ = compute_class_weights(y)
        C = np.eye(self.n_classes, dtype=np.float64)[y_idx]

        self._fit_weighted_bmlr(X, y_idx, C, N, verbose)

        if save_trace:
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(plot_dir, exist_ok=True)
            self._save_results(data_dir, plot_dir, filename_prefix, verbose)

        return self

    def _save_results(self, data_dir, plot_dir, filename_prefix, verbose):
        var_names = []
        if self.weighted and self.fixed_gamma is None and 'gamma' in self.trace.posterior:
            var_names.append('gamma')
        var_names += ['betaI', 'beta']
        if 'alpha' in self.trace.posterior:
            var_names.append('alpha')

        trace_plot = os.path.join(plot_dir, f"trace_{filename_prefix}.pdf")
        summary_file = os.path.join(data_dir, f"summary_{filename_prefix}.csv")
        les_file = os.path.join(data_dir, f"les_scores_{filename_prefix}.csv")

        az.plot_trace(self.trace, var_names=var_names, compact=True)
        plt.tight_layout()
        plt.savefig(trace_plot, bbox_inches='tight')
        plt.close()
        self.convergence_summary.to_csv(summary_file)
        np.savetxt(les_file, self.les_scores, delimiter=',')

    def cluster_diagnostics(self):
        if self.trace is None or 'omega' not in self.trace.posterior:
            return {'weight_perplexity': 1.0, 'n_occupied': 1, 'tail_mass': 0.0, 'omega_mean': [1.0]}
        H = self.H_trunc
        omega = self.trace.posterior['omega'].values.reshape(-1, H).mean(0)
        omega = omega / omega.sum()
        ent = -np.sum(omega * np.log(omega + 1e-12))
        perplexity = float(np.exp(ent))
        occ_mask = omega > 1.0 / (2 * H)
        n_occ = int(occ_mask.sum())
        order = np.argsort(omega)[::-1]
        tail = float(omega[order][n_occ:].sum()) if n_occ < H else 0.0
        return {'weight_perplexity': perplexity, 'n_occupied': n_occ, 'tail_mass': tail, 'omega_mean': omega.tolist()}

    def _fit_weighted_bmlr(self, X, y_idx, C, N_train, verbose=True):
        H = self.H_trunc
        K = self.n_classes
        ref = self.ref_class

        coef_class_names = [f"class_{k}" for k in range(K) if k != ref]
        coords = {'hclusters': [f"h{h}" for h in range(H)], 'classes': coef_class_names,
                  'features': [f"x{j}" for j in range(self.n_features)]}

        with pm.Model(coords=coords) as model:
            X_data = pm.Data("X_data", X)
            y_data = pm.Data("y_data", y_idx)
            H_data = pm.Data("H_data", self.les_scores)
            C_data = pm.Data("C_data", C)

            if not self.weighted:
                weights_data = pt.ones(N_train)
                self.learned_gamma = None
            else:
                if self.fixed_gamma is None:
                    gamma = pm.HalfNormal('gamma', sigma=self.gamma_prior_sigma)
                else:
                    gamma = pt.constant(float(self.fixed_gamma))   # entropy term fixed, not sampled
                h_ent = (1.0 + H_data) ** gamma
                S_k = pt.dot(C_data.T, h_ent)
                S_obs = pt.dot(C_data, S_k)
                w = (N_train / K) * (h_ent / S_obs)
                weights_data = pm.Deterministic("weights_data", w)

            if H > 1:
                if self.fixed_alpha is None:
                    alpha = pm.Gamma('alpha', alpha=self.alpha_prior_a, beta=self.alpha_prior_b)
                else:
                    alpha = pt.constant(float(self.fixed_alpha))   # concentration held fixed
                omega = pm.StickBreakingWeights('omega', alpha=alpha, K=H - 1, dims='hclusters')
            else:
                omega = pt.ones(1)

            betaI = pm.Normal('betaI', mu=0, sigma=self.prior_sigma, dims=('hclusters', 'classes'))
            beta = pm.Normal('beta', mu=0, sigma=self.prior_sigma, dims=('hclusters', 'classes', 'features'))
            nonref = [k for k in range(K) if k != ref]

            betaI_full = pt.zeros((H, K))
            betaI_full = pt.set_subtensor(betaI_full[:, nonref], betaI)

            beta_full = pt.zeros((H, K, self.n_features))
            beta_full = pt.set_subtensor(beta_full[:, nonref, :], beta)

            XB = pt.tensordot(X_data, beta_full, axes=[[1], [2]])
            logits = betaI_full[None, :, :] + XB
            proba = pt.special.softmax(logits, axis=2)

            proba_mix = pt.sum(omega[None, :, None] * proba, axis=1)
            pm.Deterministic("proba", proba_mix)

            n_idx = pt.arange(y_data.shape[0])
            pi_obs = proba[n_idx, :, y_data]
            log_mix = pt.logsumexp(pt.log(omega)[None, :] + pt.log(pi_obs + 1e-10), axis=1)

            weighted_log_lik = pt.sum(weights_data * log_mix)
            pm.Potential("weighted_likelihood", weighted_log_lik)

            trace = pm.sample(draws=self.n_samples, tune=self.n_tune, chains=self.n_chains, cores=self.cores,
                              target_accept=self.target_accept, return_inferencedata=True, progressbar=verbose,
                              init="advi+adapt_diag", idata_kwargs={'log_likelihood': False})

        self.model = model
        self.trace = trace

        proba_xr = trace.posterior['proba']
        samples = proba_xr.stack(sample=('chain', 'draw')).values
        samples = np.moveaxis(samples, -1, 0)
        self._train_proba_samples = samples
        self._train_proba = samples.mean(axis=0)

        if 'alpha' in trace.posterior:
            self.learned_alpha = float(trace.posterior['alpha'].values.mean())
        elif H > 1:
            self.learned_alpha = self.fixed_alpha
        else:
            self.learned_alpha = None

        if self.weighted and self.fixed_gamma is None and 'gamma' in trace.posterior:
            self.learned_gamma = float(trace.posterior['gamma'].values.mean())
        else:
            self.learned_gamma = self.fixed_gamma

        var_names = []
        if self.weighted and self.fixed_gamma is None and 'gamma' in trace.posterior:
            var_names.append('gamma')
        var_names += ['betaI', 'beta']
        if 'alpha' in trace.posterior:
            var_names.append('alpha')
        self.convergence_summary = az.summary(trace, var_names=var_names, hdi_prob=0.95, round_to=3)