"""Object Counting as "total instances of k objects" (`_optv3e`).

Why: asking for one object's count leaves a blind floor of ~0.35. Most single
counts are 1-2, so the correct option can only be the smallest or
second-smallest. Here an item names k in {2, 3} objects that are valid
counting targets in the same route (route_plan) or scene (general). The answer
is the total number of instances, counted across all videos.

Selection rules:
- Only totals with excess = total - k >= 3. The question implies each named
  object is present (total >= k), and excess >= 3 lets the answer take any of
  the 4 option ranks.
- Options: 4 distinct values in [k, total + window], fitted separately per k
  with fit_balanced_option_sets. This makes the option set carry as little
  information about the answer as possible.
- Name prior: per-object mean counts (training units; cross-fitted by scene
  halves for training items) predict each total. The residual class
  (answer - round(prediction): <=-1, 0, +1, >=+2) is balanced, so "pick the
  option nearest to the typical total" cannot beat chance by much.
- Buckets: (k, excess 3 / 4 / 5+); residual class interleaved inside each.
  Scene round-robin.
- Train: every unit of the training pools. Eval: every unit of the val pools,
  nested 200 / 500.
- Training reasoning: the per-object v3 traces joined, then the sum.

  python -m generation.build.counting_total --split train
  python -m generation.build.counting_total --split val
"""
import argparse
import collections
import glob
import itertools
import json
import os
import random
import re

from generation.option_balance import fit_balanced_option_sets, sample_option_set, shuffled_letter_options
from generation.build import reoption as v2
from generation.build.common import scene_of, write_outputs
from generation.build.nested import balanced_select, nested_order

SEED = 20260913
OBJ = re.compile(r"instances of '(.*)' are present")
TRAIN_POOLS = sorted(glob.glob("json_files/train/_shards/counting_routepool_v3c/shard_*[0-9].jsonl")) + \
    ["json_files/train/Habitat_Object_Counting_trainpool4_v3c.jsonl"]
VAL_POOLS = ["json_files/eval/_pools/Habitat_Object_Counting_route_plan_fullval.jsonl",
             "json_files/eval/_pools/Habitat_Object_Counting_general_fullval.jsonl"]
OUT = {"train": "json_files/train/Habitat_Object_Counting_optv3e.jsonl",
       "val": "json_files/eval/Holistic_Synthesis/Habitat_Object_Counting_optv3e.jsonl"}
WINDOW = 4


def body(text):
    return re.sub(r"^[A-Z]\)\s*", "", text)


def load_units(paths):
    """{video_list json: {"rows": {object: row}, "scene": scene}}"""
    units = {}
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            key = json.dumps(r["video_list"], sort_keys=True)
            u = units.setdefault(key, {"rows": {}, "scene": scene_of(r), "video_list": r["video_list"]})
            u["rows"].setdefault(OBJ.search(r["question"]).group(1), r)
    return units


def name_means(units):
    per = collections.defaultdict(list)
    for u in units.values():
        for o, r in u["rows"].items():
            per[o].append(int(body(r["answer"])))
    return {o: sum(c) / len(c) for o, c in per.items()}


def residual_class(total, predicted):
    d = total - round(predicted)
    return "<=-1" if d <= -1 else ("0" if d == 0 else ("+1" if d == 1 else ">=+2"))


def question(names, n_videos):
    quoted = [f"'{x}'" for x in names]
    listed = " and ".join(quoted) if len(quoted) == 2 else f"{quoted[0]}, {quoted[1]} and {quoted[2]}"
    return (f"Based on the combined exploration in all {n_videos} videos, how many distinct instances of "
            f"{listed} are present in this scene in total?")


def candidates(units, means_for, rng, max_combos):
    items = []
    for key, u in units.items():
        objs = sorted(u["rows"])
        means = means_for(u["scene"])
        for k in (2, 3):
            combos = list(itertools.combinations(objs, k))
            rng.shuffle(combos)
            for names in combos[:max_combos]:
                counts = [int(body(u["rows"][o]["answer"])) for o in names]
                total = sum(counts)
                if total - k < 3:
                    continue
                pred = sum(means.get(o, 1.0) for o in names)
                # excess buckets 3, 4, 5, 6+: a single total then holds at most 1/4 of a k's answers
                items.append(dict(key=key, scene=u["scene"], names=list(names), counts=counts, k=k, total=total,
                                  excess=min(total - k, 6), residual=residual_class(total, pred), predicted=round(pred, 2)))
    return items


def to_row(item, units, split, rng):
    u = units[item["key"]]
    rows = [u["rows"][o] for o in item["names"]]
    row = {"question": question(item["names"], len(u["video_list"])), "video_list": u["video_list"],
           "total": item["total"], "counts": dict(zip(item["names"], item["counts"])), "k": item["k"]}
    if split == "train":
        for field in ("reasoning_cot", "reasoning_cof", "reasoning"):
            if all(field in r for r in rows):
                parts = [f"**Object {i + 1}: '{o}'**\n{r[field]}" for i, (o, r) in enumerate(zip(item["names"], rows))]
                sums = " + ".join(str(c) for c in item["counts"])
                row[field] = "\n\n".join(parts) + f"\n\n**Total:** {sums} = {item['total']}. The answer is {item['total']}."
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--n", type=int, default=None, help="train: 2000; val: 500 (nested 200)")
    parser.add_argument("--max-combos", type=int, default=40, help="k-object combinations sampled per unit and k")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=None, help="Output path (default: the _optv3e file of the split).")
    parser.add_argument("--draws", type=int, default=30, help="Offset-fitted option-set draws per item.")
    parser.add_argument("--total-weight-power", type=float, default=1.0,
                        help="Draw weight = (total-fitted probability) ** power; 0 = offset fit only.")
    args = parser.parse_args()
    n = args.n or (2000 if args.split == "train" else 500)
    out = args.out or OUT[args.split]
    if not args.dry_run and (os.path.exists(out) or os.path.exists(out + ".meta.json")):
        raise SystemExit(f"FATAL: {out} exists, not overwriting")
    rng = random.Random(f"{SEED}:counting_total:{args.split}")

    train_units = load_units(TRAIN_POOLS)
    if args.split == "train":
        units = train_units
        scenes = sorted({u["scene"] for u in units.values()})
        rng.shuffle(scenes)
        half = {s: i % 2 for i, s in enumerate(scenes)}
        fitted = [name_means({k: u for k, u in units.items() if half[u["scene"]] == h}) for h in (0, 1)]
        means_for = lambda scene: fitted[1 - half[scene]]
    else:
        units = load_units(VAL_POOLS)
        overall = name_means(train_units)
        means_for = lambda scene: overall
    items = candidates(units, means_for, rng, args.max_combos)
    # one item per (unit, name set); dedupe identical (question, videos)
    seen, uniq = set(), []
    for it in items:
        sig = (it["key"], tuple(it["names"]))
        if sig not in seen:
            seen.add(sig)
            uniq.append(it)
    items = uniq

    bucket_of = lambda i: f"k{items[i]['k']}|e{items[i]['excess']}"
    sub_of = lambda i: items[i]["residual"]
    scene_fn = lambda i: items[i]["scene"]
    if args.split == "train":
        idx, report = balanced_select(len(items), bucket_of, scene_fn, rng, n, sub_of=sub_of)
    else:
        idx, report = nested_order(len(items), bucket_of, scene_fn, rng, sizes=(200, n), sub_of=sub_of)
    chosen = [items[i] for i in idx]

    # Option sets. A blind model can anchor on the objects' typical total (the sum of per-object mean
    # counts; plain fitted sets left "pick the option nearest to it" at 0.36 on the val file). So the
    # sets are fitted on the offset d = total - anchor, anchor = round(typical total): 4 distinct
    # offsets within +/-WINDOW of d, fitted per k so the offsets say as little as possible about d,
    # then shifted by the anchor. Every value must be >= k (the question implies each named object is
    # present); items with no feasible fitted set fall back to sets fitted on the total itself.
    anchor = lambda it: round(it["predicted"])
    fitted_rel, fitted_abs, option_stats = {}, {}, {}
    for k in (2, 3):
        group = [it for it in chosen if it["k"] == k]
        if not group:
            continue

        def cand_rel(d):
            return [S for S in itertools.combinations(range(d - WINDOW, d + WINDOW + 1), 4) if d in S]

        def cand_abs(v, k=k):
            return [S for S in itertools.combinations(range(k, v + WINDOW + 1), 4) if v in S]
        fitted_rel[k], option_stats[f"k{k}_offset"] = fit_balanced_option_sets(
            collections.Counter(it["total"] - anchor(it) for it in group), cand_rel, collections.Counter)
        fitted_abs[k], option_stats[f"k{k}_total"] = fit_balanced_option_sets(
            collections.Counter(it["total"] for it in group), cand_abs, collections.Counter)
    built, role_hits = [], collections.Counter()
    for it in chosen:
        a, k = anchor(it), it["k"]
        # Offset-fitted draws alone let absolute values leak (the lowest total showed up mostly when it was
        # the answer): among 30 offset-fitted draws, pick one with probability proportional to its weight
        # under the total-fitted distribution, so both views stay close to their fitted optimum.
        draws = [sample_option_set(fitted_rel[k][it["total"] - a], rng, feasible=lambda S: min(S) + a >= k)
                 for _ in range(args.draws)]
        draws = [tuple(sorted(x + a for x in S)) for S in draws if S is not None]
        weights = [fitted_abs[k][it["total"]].get(S, 0.0) ** args.total_weight_power for S in draws]
        if draws and sum(weights) > 0:
            x, S = rng.random() * sum(weights), draws[-1]
            for cand_set, w in zip(draws, weights):
                x -= w
                if x <= 0:
                    S = cand_set
                    break
            role_hits["offset_fit_total_weighted"] += 1
        elif draws:
            S = draws[0]
            role_hits["offset_fit"] += 1
        else:
            S = sample_option_set(fitted_abs[k][it["total"]], rng)
            role_hits["total_fit_fallback"] += 1
        options, answer = shuffled_letter_options(str(it["total"]), [str(v) for v in S if v != it["total"]], rng)
        row = to_row(it, units, args.split, rng)
        row["options"], row["answer"] = list(options), answer
        built.append(row)
    if args.split == "val":
        # keep the nested order; balance letters within the first 200 and the rest separately
        head = v2.balance_answer_letters(built[:200], rng)
        tail = v2.balance_answer_letters(built[200:], rng)
        built = head + tail
    else:
        built = v2.balance_answer_letters(built, rng)

    stats = {
        "size": len(built), "candidates": len(items), "units": len(units),
        "scenes": len({it["scene"] for it in chosen}), "candidate_scenes": len({it["scene"] for it in items}),
        "buckets": dict(sorted(collections.Counter(f"k{it['k']}|e{it['excess']}" for it in chosen).items())),
        "residual": dict(sorted(collections.Counter(it["residual"] for it in chosen).items())),
        "totals": dict(sorted(collections.Counter(it["total"] for it in chosen).items())),
        "select_report": report, "option_stats": option_stats, "name_role_draws": dict(role_hits),
        "letters": dict(sorted(collections.Counter(r["answer"][0] for r in built).items())),
    }
    print(json.dumps(stats, default=str)[:3000])
    if not args.dry_run:
        write_outputs(out, built, dict(task="counting_total", script="generation/build/counting_total.py", seed=SEED,
                                       split=args.split, pools=TRAIN_POOLS if args.split == "train" else VAL_POOLS,
                                       stats=stats))


if __name__ == "__main__":
    main()
