import os

sim_results_dir = os.path.join("outputs", "simulation", "runs")     # per-replicate JSON
sim_summary_dir = os.path.join("outputs", "simulation", "summary")  # aggregated tables

sampling = {"quick": dict(n_samples=600, n_tune=600, n_chains=2, cores=1, target_accept=0.95),
            "production": dict(n_samples=5000, n_tune=5000, n_chains=4, cores=1, target_accept=0.99)}

h_trunc_fit = 8
fit_specs = (("unweighted", None), ("weighted", None), ("weighted", 0.5), ("weighted", 1.0),
             ("weighted",   5.0), ("weighted",   10.0))

n_test_grid = 400
k = 4
p = 4
n_reps = 30
sep = 2.0
imb = 2.0

h_true_values = [3, 4, 6]
n_train_values = [300, 600, 1000]

conditions = [
    dict(name=f"logit{h}_n{n}", K=k, p=p,  H_true=h, omega_true=[1.0] * h, separation=sep, imbalance=imb, n_train=n,
         mixing="fixed", n_replicates=n_reps) for h in h_true_values for n in n_train_values]

random_seed_base = 20240601

