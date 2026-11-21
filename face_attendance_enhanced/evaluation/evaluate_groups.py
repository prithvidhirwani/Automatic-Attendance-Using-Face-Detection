"""Group-image evaluation.  Run:  python -m evaluation.evaluate_groups [--threshold 70]"""
import argparse

from evaluation.common import run_evaluation


def run(cfg=None, threshold=None, **kw):
    return run_evaluation("group", cfg, threshold, **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=None, help="LBPH distance threshold")
    info = run(threshold=ap.parse_args().threshold)
    print(f"Evaluated {info['images_evaluated']} images ({info['images_skipped']} skipped).")
    for w in info["warnings"]:
        print("  warning:", w)
    print("Wrote results/group_results.csv and results/summary_metrics.csv")
