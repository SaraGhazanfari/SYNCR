"""Confidence intervals and paired tests from saved per-item predictions (lenient scoring).

One model: per-task accuracy with a Wilson 95% interval, and the unweighted task average with a bootstrap 95%
interval (items resampled within each task). Two models on the same items: per-task difference with an exact
McNemar p-value, and the average difference with a paired bootstrap 95% interval.

  python -m eval.stats_report --tag Qwen__Qwen3-VL-8B-Instruct --n 200 --suite syncr
  python -m eval.stats_report --tag qwen3-vl-sft_v3e_all8_answer_only --compare Qwen__Qwen3-VL-8B-Instruct --n 200 --suite syncr
  python -m eval.stats_report --tag gpt-6-astra --root eval_predictions_gpt_fps/hf --n 25 --suite syncr
"""
import argparse
import math
import random

from eval.controls_report import SUITES, find_predictions, lenient_correct


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


def mcnemar_exact(only_a, only_b):
    """Two-sided exact binomial test on the discordant pairs."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(root, tag, stem, n):
    path = find_predictions(root, tag, stem, n)
    if path is None:
        return None
    correct, samples = lenient_correct(path, n)
    items = range(min(n, len(samples)))
    return [bool(correct.get(i, False)) for i in items]


def bootstrap_mean(task_vectors, reps, rng):
    """95% interval of the unweighted mean of per-task accuracies, resampling items within each task."""
    means = []
    for _ in range(reps):
        accs = []
        for vec in task_vectors:
            accs.append(sum(vec[rng.randrange(len(vec))] for _ in range(len(vec))) / len(vec))
        means.append(100 * sum(accs) / len(accs))
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def bootstrap_paired_diff(pairs, reps, rng):
    """95% interval of mean_task(acc_a) - mean_task(acc_b), resampling the same item indices for both models."""
    diffs = []
    for _ in range(reps):
        per_task = []
        for a, b in pairs:
            idx = [rng.randrange(len(a)) for _ in range(len(a))]
            per_task.append((sum(a[i] for i in idx) - sum(b[i] for i in idx)) / len(a))
        diffs.append(100 * sum(per_task) / len(per_task))
    diffs.sort()
    return diffs[int(0.025 * reps)], diffs[int(0.975 * reps) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--root", default="eval_predictions/hf")
    parser.add_argument("--compare", default=None, help="second model tag, scored on the same items")
    parser.add_argument("--compare-root", default=None)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--suite", choices=sorted(SUITES), default="syncr")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    if args.compare is None:
        print(f"\n### {args.tag}, {args.suite}, first {args.n} items (lenient, Wilson 95%)\n")
        print("| Task | n | Accuracy | 95% CI |")
        print("|---|---|---|---|")
        vectors = []
        for name, stem in SUITES[args.suite]:
            vec = load(args.root, args.tag, stem, args.n)
            if vec is None:
                print(f"| {name} | | pending | |")
                continue
            lo, hi = wilson(sum(vec), len(vec))
            vectors.append(vec)
            print(f"| {name} | {len(vec)} | {100 * sum(vec) / len(vec):.1f} | {lo:.1f}–{hi:.1f} |")
        if len(vectors) == len(SUITES[args.suite]):
            avg = 100 * sum(sum(v) / len(v) for v in vectors) / len(vectors)
            lo, hi = bootstrap_mean(vectors, args.reps, rng)
            print(f"| **Average** | | {avg:.1f} | {lo:.1f}–{hi:.1f} (bootstrap) |")
        return

    compare_root = args.compare_root or args.root
    print(f"\n### {args.tag} vs {args.compare}, {args.suite}, first {args.n} items (lenient)\n")
    print("| Task | n | A | B | A − B | A only | B only | McNemar p |")
    print("|---|---|---|---|---|---|---|---|")
    pairs = []
    for name, stem in SUITES[args.suite]:
        a = load(args.root, args.tag, stem, args.n)
        b = load(compare_root, args.compare, stem, args.n)
        if a is None or b is None:
            print(f"| {name} | | pending | | | | | |")
            continue
        m = min(len(a), len(b))
        a, b = a[:m], b[:m]
        only_a = sum(x and not y for x, y in zip(a, b))
        only_b = sum(y and not x for x, y in zip(a, b))
        pairs.append((a, b))
        acc_a, acc_b = 100 * sum(a) / m, 100 * sum(b) / m
        print(f"| {name} | {m} | {acc_a:.1f} | {acc_b:.1f} | {acc_a - acc_b:+.1f} | {only_a} | {only_b} | "
              f"{mcnemar_exact(only_a, only_b):.3g} |")
    if len(pairs) == len(SUITES[args.suite]):
        diff = 100 * sum((sum(a) - sum(b)) / len(a) for a, b in pairs) / len(pairs)
        lo, hi = bootstrap_paired_diff(pairs, args.reps, rng)
        print(f"| **Average** | | | | {diff:+.1f} | | | 95% CI {lo:+.1f} to {hi:+.1f} (paired bootstrap) |")


if __name__ == "__main__":
    main()
