"""Numerical Comparison eval pool from CLEVRER validation videos, no video shared between items.

The original 1000-item benchmark file pairs 1282 distinct videos, one of them in 31
items; its 500-item `_optv3c` subset still repeats a first video up to 13 times.
This script draws candidate 2-video items from the validation split
(get_collision_count_QA, same arguments as the training builder), removes
duplicates, then picks items greedily -- round-robin over (difference bucket,
winner video) -- so that no video appears in two picked items, until every
(bucket, winner) reaches its target. Targets are --slack times the 500-item quota
(ties 1/5; differences 1, 2, 3+ equal; winner video balanced), so build_eval's
nested ordering still has room. Writes rows with the original benchmark keys.

  python -m generation.build.numerical_eval_pool
"""
from utils import DATA_ROOT
import argparse
import collections
import json
import math
import os
import random

from generation.build import tasks

OUT = "json_files/eval/_pools/Clvr_Numerical_Comparison_fullval.jsonl"
KEYS = ["collision_count", "video_list", "reasoning", "question", "options", "answer"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clevrer-root", default=f"{DATA_ROOT}/clvr/")
    parser.add_argument("--val-videos", type=int, default=5000)
    parser.add_argument("--draws-per-video", type=int, default=15)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--slack", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"FATAL: {args.out} exists, not overwriting")

    from generation.clevrer.clvr_comparative_reasoning import CLEVRER_Comparative_Reasoning
    random.seed(f"{args.seed}:numerical_eval_pool")
    ds = CLEVRER_Comparative_Reasoning(root_path=args.clevrer_root, split="validation", sample_num=args.val_videos)
    candidates = []
    for _ in range(args.draws_per_video):
        for i in range(len(ds.data_list)):
            candidates.append(ds.get_collision_count_QA(i, nframes=32, k=2, time_mode="timestamp"))
    kept, dedupe = tasks.dedupe_rows(candidates)
    candidates = [candidates[i] for i in kept]

    weights = tasks.BUCKET_WEIGHTS["numerical"]
    total_w = sum(weights.values())
    groups = collections.defaultdict(list)
    for r in candidates:
        groups[(tasks.numerical_bucket(r), tasks.numerical_winner(r))].append(r)
    targets = {}
    for (b, w) in groups:
        subs = 1 if b == "tie" else 2
        targets[(b, w)] = math.ceil(args.n * weights[b] / total_w / subs * args.slack)
    rng = random.Random(f"{args.seed}:numerical_eval_pool:pick")
    queues = {g: rng.sample(rows, len(rows)) for g, rows in groups.items()}
    used, picked, counts = set(), [], collections.Counter()
    # scarcest (bucket, winner) first, so the rare large differences get their videos before common ones
    order = sorted(queues, key=lambda g: len(queues[g]) / targets[g])
    progress = True
    while progress:
        progress = False
        for g in order:
            if counts[g] >= targets[g]:
                continue
            q = queues[g]
            while q:
                r = q.pop()
                videos = {v["video_path"] for v in r["video_list"]}
                if videos & used:
                    continue
                used |= videos
                picked.append(r)
                counts[g] += 1
                progress = True
                break
    short = {f"{b}|{w}": [counts[(b, w)], targets[(b, w)]] for (b, w) in sorted(targets) if counts[(b, w)] < targets[(b, w)]}
    # Benchmark option layout (tie option + one per video), as reoption_numerical builds it; build_eval
    # refits the options on its 500 but checks that the option count is unchanged.
    from generation.option_balance import reoption_numerical
    built, _ = reoption_numerical([r["collision_count"] for r in picked], rng)
    for r, (options, answer) in zip(picked, built):
        r["options"], r["answer"] = list(options), answer
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in picked:
            f.write(json.dumps({k: r[k] for k in KEYS if k in r}) + "\n")
    uses = collections.Counter(v["video_path"] for r in picked for v in r["video_list"])
    print(f"{args.out}: {len(picked)} items from {len(candidates)} candidates (duplicates {dedupe}); "
          f"per (bucket, winner) {dict(sorted((f'{b}|{w}', c) for (b, w), c in counts.items()))}; "
          f"targets {dict(sorted((f'{b}|{w}', t) for (b, w), t in targets.items()))}; short {short}; "
          f"max uses per video {max(uses.values())}")


if __name__ == "__main__":
    main()
