"""Object Counting pools from every Habitat route (route_plan) or scene (general), sharded.

The single-process pool run (generate_counting_pool.SBATCH) scans the general
section first and saves only after each section, so a cancelled run kept no
route_plan items; the training files covered a few hundred of the 3452
route_plan/train routes. This script processes one shard of the sorted unit
list (route_plan: routes, general: scenes), appending each unit's items as it
goes, and a merge step builds one 4-option file:
  train route_plan: json_files/train/Habitat_Object_Counting_routepool_v3c.jsonl
  other (split, section): json_files/eval/_pools/Habitat_Object_Counting_<section>_full<split>.jsonl
Every row keeps the per-object count it was generated for; the rows of one unit
share a video_list, so they also give the unit's full object -> count map.

  python -m generation.build.counting_routepool --shard 3 --num-shards 40
  python -m generation.build.counting_routepool --merge --num-shards 40
  python -m generation.build.counting_routepool --split val --section general --shard 0 --num-shards 12
"""
import argparse
import collections
import json
import os
import random
import time

from generation.habitat.habitat_holistic_aggregation import (
    DEFAULT_ROOT_PATH, Habitat_Dataset, generate_single_object_counting_qa,
)
from generation.option_balance import reoption_counting


def paths(split, section):
    if split == "train" and section == "route_plan":
        return "json_files/train/_shards/counting_routepool_v3c", "json_files/train/Habitat_Object_Counting_routepool_v3c.jsonl"
    return (f"json_files/eval/_pools/_shards/counting_{section}_{split}" if split == "val"
            else f"json_files/train/_shards/counting_{section}_{split}",
            f"json_files/eval/_pools/Habitat_Object_Counting_{section}_full{split}.jsonl" if split == "val"
            else f"json_files/train/Habitat_Object_Counting_{section}pool_v3d.jsonl")


def unit_list(data_path, section):
    units = []
    for scene in sorted(os.listdir(data_path)):
        scene_path = os.path.join(data_path, scene)
        if not os.path.isdir(scene_path):
            continue
        if section == "general":
            units.append((scene, None))
            continue
        for route in sorted(os.listdir(scene_path)):
            if os.path.isdir(os.path.join(scene_path, route)):
                units.append((scene, route))
    return units


def run_shard(args):
    shard_dir, _ = paths(args.split, args.section)
    os.makedirs(shard_dir, exist_ok=True)
    out = os.path.join(shard_dir, f"shard_{args.shard:03d}.jsonl")
    if os.path.exists(out + ".done"):
        print(f"{out} already done")
        return
    habitat = Habitat_Dataset(root_path=args.root_path, section=args.section, split=args.split)
    units = unit_list(habitat.data_path, args.section)
    mine = [u for i, u in enumerate(units) if i % args.num_shards == args.shard]
    random.seed(args.seed * 1000 + args.shard)
    print(f"{args.section}/{args.split} shard {args.shard}/{args.num_shards}: {len(mine)} of {len(units)} units", flush=True)
    n_items = 0
    start = time.time()
    with open(out, "w") as f:
        for k, (scene, route) in enumerate(mine):
            t = time.time()
            try:
                rows = generate_single_object_counting_qa(habitat, scene, route_idx=route,
                                                          include_reasoning=args.split == "train")
            except (IndexError, ValueError) as e:  # e.g. an empty semantic video for one camera
                print(f"[{k + 1}/{len(mine)}] {scene}/{route or ''}: SKIPPED ({type(e).__name__}: {e})", flush=True)
                continue
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
            n_items += len(rows)
            print(f"[{k + 1}/{len(mine)}] {scene}/{route or ''}: {len(rows)} items in {time.time() - t:.1f}s "
                  f"(total {n_items}, {time.time() - start:.0f}s)", flush=True)
    open(out + ".done", "w").write(f"{len(mine)} units, {n_items} items\n")


def run_merge(args):
    shard_dir, output = paths(args.split, args.section)
    if os.path.exists(output):
        raise SystemExit(f"FATAL: {output} exists, not overwriting")
    rows = []
    for shard in range(args.num_shards):
        path = os.path.join(shard_dir, f"shard_{shard:03d}.jsonl")
        if not os.path.exists(path + ".done"):
            raise SystemExit(f"FATAL: {path} not finished")
        rows += [json.loads(line) for line in open(path) if line.strip()]
    # Shards run per route, so the per-scene cap is applied to each scene's full set of
    # generated items here: 3+ items first (the scarce bucket), then 2, then 1, and within a
    # bucket round-robin over objects so one object counted along many routes does not fill the cap.
    generated = len(rows)
    cap_rng = random.Random(f"{args.seed}:counting_routepool:cap")
    by_scene = collections.defaultdict(list)
    for row in rows:
        by_scene[row["video_list"][0]["video_path"].split("/")[-4 if args.section == "route_plan" else -3]].append(row)
    rows = []
    for scene in sorted(by_scene):
        order = []
        for bucket in ("3+", "2", "1"):
            by_object = collections.defaultdict(list)
            for row in by_scene[scene]:
                value = int(row["answer"][3:])
                if ("3+" if value >= 3 else str(value)) == bucket:
                    by_object[row["question"]].append(row)
            queues = [cap_rng.sample(q, len(q)) for _, q in sorted(by_object.items())]
            while any(queues):
                for q in queues:
                    if q:
                        order.append(q.pop())
        rows += order[:args.per_scene_cap] if args.per_scene_cap > 0 else order
    print(f"per-scene cap {args.per_scene_cap}: kept {len(rows)}/{generated} items")
    rebuilt, stats = reoption_counting([int(row["answer"][3:]) for row in rows],
                                       random.Random(f"{args.seed}:counting_routepool"))
    for row, (options, answer) in zip(rows, rebuilt):
        row["options"], row["answer"] = options, answer
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    values = collections.Counter(int(row["answer"][3:]) for row in rows)
    buckets = collections.Counter("3+" if v >= 3 else str(v) for v in values.elements())
    scenes = len(by_scene)
    print(f"{output}: {len(rows)} items from {scenes} scenes; buckets {dict(buckets)}; "
          f"answers {dict(sorted(values.items()))}; options {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=40)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--section", default="route_plan", choices=["route_plan", "general"])
    parser.add_argument("--root-path", default=DEFAULT_ROOT_PATH)
    parser.add_argument("--seed", type=int, default=20260913)
    # 0 = keep every item; build_train's balanced_select spreads the selection evenly over scenes.
    parser.add_argument("--per-scene-cap", type=int, default=0)
    args = parser.parse_args()
    run_merge(args) if args.merge else run_shard(args)
