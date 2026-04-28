"""
run_experiment.py
=================
CLI entry point for config-driven INR time-frequency experiments.

Usage
-----
Single run:
    python run_experiment.py --config configs/marginal_only.yaml

Multi-seed run:
    python run_experiment.py --config configs/marginal_only.yaml \
        --seeds 0 1 2

Parallel multi-seed run:
    python run_experiment.py --config configs/marginal_only.yaml \
        --seeds 0 1 2 --parallel
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an INR time-frequency experiment from a YAML config."
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to run. Overrides the config 'seed' field. "
             "If omitted, the config seed is used.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run seeds in parallel using multiprocessing.Pool.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (-1 = all CPU cores).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override the output root directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.output_root is not None:
        config["output_root"] = args.output_root

    # Import here to keep startup fast
    from inr_tfr.train import run_experiment, run_experiment_multi_seed

    if args.seeds is not None and len(args.seeds) > 1:
        results = run_experiment_multi_seed(
            config,
            seeds=args.seeds,
            parallel=args.parallel,
            n_jobs=args.n_jobs,
        )
        print("\n===== MULTI-SEED SUMMARY =====")
        for r in results:
            m = r["metrics"]
            print(
                f"  seed={m['seed']:3d} | "
                f"rel_err_t={m['rel_err_t']:.4f} | "
                f"rel_err_f={m['rel_err_f']:.4f} | "
                f"symmetry={m['symmetry_err']:.3e} | "
                f"dir={r['output_dir']}"
            )
    else:
        if args.seeds is not None and len(args.seeds) == 1:
            config["seed"] = args.seeds[0]
        result = run_experiment(config)
        print("\n===== RESULT =====")
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
