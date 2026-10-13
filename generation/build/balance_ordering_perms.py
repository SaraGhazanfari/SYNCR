"""Equal answer orders in a Sequential Ordering file by relabelling the segments.

The `_optv3c` gapped Ordering eval file has balanced letters but not balanced
answer orders: picking the order most frequent in training scored 0.349 on its
first 200 items (0.287 on 500). Each item's segments are renumbered so that its
answer becomes a target order: targets cycle through the 24 orders in blocks of
24 (a fresh random order per block), so any prefix -- 200, 500, 2000 -- holds
every order within one of the others. The same renumbering is applied to the
video_list order, every option and every reasoning field, so each item still
shows the same segments with the same option set; answer letters do not change.

  python -m generation.build.balance_ordering_perms IN.jsonl OUT.jsonl
"""
import collections
import itertools
import json
import os
import random
import re
import sys

SEED = 20260913
VIDEO = re.compile(r"Video (\d+)")


def body(option):
    return re.sub(r"^[A-Z]\)\s*", "", option)


def order_of(answer):
    return [int(x) for x in VIDEO.findall(body(answer))]


def relabel(row, target):
    """Renumber so the chronological order reads `target` (list of 1-based video numbers)."""
    current = order_of(row["answer"])
    mapping = {old: new for old, new in zip(current, target)}  # old video number -> new video number
    sub = lambda text: VIDEO.sub(lambda m: f"Video {mapping.get(int(m.group(1)), int(m.group(1)))}", text)
    out = dict(row)
    videos = [None] * len(row["video_list"])
    for old, clip in enumerate(row["video_list"], start=1):
        videos[mapping[old] - 1] = clip
    out["video_list"] = videos
    out["options"] = [sub(o) for o in row["options"]]
    out["answer"] = sub(row["answer"])
    for key in row:
        if key.startswith("reasoning") and isinstance(row[key], str):
            out[key] = sub(row[key])
    return out


def main(src, dst):
    if os.path.exists(dst) or os.path.exists(dst + ".meta.json"):
        raise SystemExit(f"FATAL: {dst} exists, not overwriting")
    rows = [json.loads(line) for line in open(src) if line.strip()]
    k = len(order_of(rows[0]["answer"]))
    orders = [list(p) for p in itertools.permutations(range(1, k + 1))]
    rng = random.Random(f"{SEED}:ordering_perms:{os.path.basename(dst)}")
    targets = []
    while len(targets) < len(rows):
        block = orders[:]
        rng.shuffle(block)
        targets += block
    out = [relabel(r, t) for r, t in zip(rows, targets)]
    for old, new in zip(rows, out):
        assert new["answer"] in new["options"] and len({body(o) for o in new["options"]}) == len(new["options"])
        assert new["answer"][0] == old["answer"][0]
        assert sorted(map(json.dumps, new["video_list"])) == sorted(map(json.dumps, old["video_list"]))
        # chronological order of the clips named by the answer is unchanged
        assert [new["video_list"][i - 1] for i in order_of(new["answer"])] == \
               [old["video_list"][i - 1] for i in order_of(old["answer"])]
    with open(dst, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    counts = lambda rr: collections.Counter(body(r["answer"]) for r in rr)
    stats = {n: [min(counts(out[:n]).values()), max(counts(out[:n]).values()), len(counts(out[:n]))]
             for n in (200, 500, 2000) if n <= len(out)}
    json.dump(dict(script="generation/build/balance_ordering_perms.py", input=src, seed=SEED, size=len(out),
                   answer_order_min_max_distinct=stats,
                   letters=dict(sorted(collections.Counter(r["answer"][0] for r in out).items()))),
              open(dst + ".meta.json", "w"), indent=1)
    print(f"{src} -> {dst}: {len(out)} items; answer orders (min, max, distinct) per prefix {stats}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
