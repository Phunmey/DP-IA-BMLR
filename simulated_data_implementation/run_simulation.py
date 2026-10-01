import os
import sys
import json
import time
import argparse
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import sim_config as C
from generate_data import generate_dataset
from fit_models import fit_and_extract
from recovery_metrics import score_replicate


def _save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _np_clean(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        out[k] = v
    return out


def run_cell(cond, rep, model_names, sampling, results_dir):
    seed = C.random_seed_base + 1000 * rep + hash(cond["name"]) % 997
    X, y, datagen, X_test, pi_test, info = generate_dataset(
        cond, seed=seed, n_test_grid=C.n_test_grid)

    for model_name in model_names:
        out_path = os.path.join(
            results_dir, f"{cond['name']}__rep{rep:03d}__{model_name}.json")
        if os.path.exists(out_path):
            continue
        t0 = time.time()
        try:
            art = fit_and_extract(
                X, y, X_test, model_name=model_name, sampling=sampling, seed=seed)
            scores = score_replicate(art, pi_test, C_true=cond["C_true"], y_test=info.get("y_test"), K=cond["K"], datagen=datagen)
            record = {"condition": cond["name"], "replicate": rep, "model": model_name,
                      "spec": {k: cond[k] for k in
                               ("K", "p", "n_clusters", "separation", "cluster_scale", "imbalance", "n_train", "C_true",
                                "generator") if k in cond},
                      "data_info": info, "scores": _np_clean(scores), "elapsed_sec": round(time.time() - t0, 1)}
            _save(record, out_path)
            print(f"{cond['name']} rep{rep:03d} [{model_name:11s}] "
                  f"cov={scores['predictive_coverage']:.3f} "
                  f"rmse={scores['predictive_rmse']:.4f} "
                  f"k_eff={scores['weight_perplexity']:.2f} (true {cond['C_true']}) "
                  f"conv={scores.get('converged')} "
                  f"({record['elapsed_sec']:.0f}s)")
        except Exception as e:
            traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--condition", type=str, default=None, help="condition name; default all")
    ap.add_argument("--replicates", type=int, default=None, help="cap number of replicates")
    ap.add_argument("--models", type=str, nargs="+", default=None,
                    choices=["dp_ia_bmlr", "dp_bmlr", "ia_bmlr", "u_bmlr"],
                    help="restrict to these models (default: all four)")
    ap.add_argument("--array-index", type=int, default=None,
                    help="run only the i-th (condition,replicate) cell across all models")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    sampling = C.sampling["quick" if args.quick else "production"]
    results_dir = args.outdir or C.sim_results_dir

    from fit_models import model_specs
    model_names = args.models if args.models else list(model_specs.keys())

    conditions = C.conditions
    if args.condition:
        conditions = [c for c in conditions if c["name"] == args.condition]
        if not conditions:
            raise SystemExit(f"Unknown condition '{args.condition}'. Options: {[c['name'] for c in C.conditions]}")

    cells = []
    for cond in conditions:
        n_rep = cond["n_replicates"] if args.replicates is None else min(args.replicates, cond["n_replicates"])
        for rep in range(n_rep):
            cells.append((cond, rep))

    print(f"Simulation | models={model_names} "
          f"| sampling={'quick' if args.quick else 'production'} "
          f"| {len(cells)} (condition,replicate) cells | out={results_dir}")

    if args.array_index is not None:
        if not (0 <= args.array_index < len(cells)):
            raise SystemExit(f"array-index out of range [0, {len(cells)})")
        cond, rep = cells[args.array_index]
        run_cell(cond, rep, model_names, sampling, results_dir)
        return

    for cond, rep in cells:
        run_cell(cond, rep, model_names, sampling, results_dir)

    print("\nDone. Aggregate with:  python aggregate_results.py")


if __name__ == "__main__":
    main()
