"""ReID and Route pools from every Habitat scene of a split, sharded.

The training files were generated with --total-num 2000, which stops the scan in
directory order: ReID covered 34 of 132 general/train scenes and Route 37 of 123
route_plan/train scenes. Each shard processes every num_shards-th scene of the
sorted scene list (Route: all routes of the scene). shard_NNN.raw.jsonl holds
every generated item of those scenes; shard_NNN.jsonl holds at most
--per-scene-cap items per scene (drawn round-robin over answer buckets). The
merge step reads the raw files by default, rebuilds options over the whole list
the same way the original generators do and writes one pool file:
  train: json_files/train/<Task>_scenepool_v3c.jsonl (build_train.py --pool-files)
  val:   json_files/eval/_pools/<Task>_fullval.jsonl (every val scene; no reasoning traces)

  python -m generation.build.habitat_pool_shards --task reid --shard 3 --num-shards 33
  python -m generation.build.habitat_pool_shards --task reid --merge --num-shards 33
  python -m generation.build.habitat_pool_shards --task route --split val --shard 0 --num-shards 10
"""
from utils import DATA_ROOT
import argparse
import collections
import json
import os
import random
import re
import time

from generation.habitat import habitat_entity_tracking as reid_gen
from generation.habitat import habitat_holistic_aggregation as route_gen
from generation.option_balance import reoption_reid, reoption_route_v3

ROOT = f"{DATA_ROOT}/habitat_data"
TASKS = {
    "reid": dict(section="general", name="Habitat_Object_Re_identification", shard="reid_scenepool_v3c"),
    "route": dict(section="route_plan", name="Habitat_Route_Planning", shard="route_scenepool_v3c"),
}


def paths(task, split):
    cfg = TASKS[task]
    if split == "train":
        return f"json_files/train/_shards/{cfg['shard']}", f"json_files/train/{cfg['name']}_scenepool_v3c.jsonl"
    return f"json_files/eval/_pools/_shards/{cfg['shard']}_{split}", f"json_files/eval/_pools/{cfg['name']}_full{split}.jsonl"


def answer_bucket(task, row):
    body = re.sub(r"^[A-Z]\)\s*", "", row["answer"])
    return ("2" if body.count("->") <= 2 else "3+") if task == "route" else "all"


def cap_per_scene(rows, cap, bucket_of, rng):
    """At most cap rows, taken round-robin over answer buckets (each bucket shuffled)."""
    if len(rows) <= cap:
        return list(rows)
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[bucket_of(row)].append(row)
    queues = [rng.sample(b, len(b)) for _, b in sorted(buckets.items())]
    kept = []
    while len(kept) < cap:
        for q in queues:
            if q and len(kept) < cap:
                kept.append(q.pop())
    return kept


def reid_scene_items(habitat, scene_idx, include_reasoning):
    """The per-scene body of generate_object_reidentification_qa (no cap)."""
    rows = []
    common_obj_ids = habitat.get_common_object_ids(scene_idx=scene_idx)
    frame_counts = habitat.get_video_frame_counts(scene_idx=scene_idx)
    for obj_id in common_obj_ids:
        full_object_name = habitat.get_object_name(scene_idx=scene_idx, object_id=obj_id)
        if full_object_name == "Unknown Object":
            continue
        object_name = " ".join(full_object_name.split(" ")[1:-1])
        if object_name in reid_gen.objects_to_filter:
            continue
        presence_data = reid_gen.filter_presence_data(
            habitat.get_timely_object_presence(scene_idx=scene_idx, object_id=obj_id),
            min_range_len=3, min_total_frames=5, drop_boundary_ranges=False)
        cam_list = list(presence_data.keys())
        for ref_idx in range(len(cam_list) - 1):
            for target_idx in range(ref_idx + 1, len(cam_list)):
                rows.append(reid_gen.generate_single_object_reidentification_qa(
                    scene_path=habitat._get_scene_path(scene_idx), clean_obj_name=object_name,
                    presence_data=presence_data, ref_cam=cam_list[ref_idx], target_cam=cam_list[target_idx],
                    fps=5, include_reasoning=include_reasoning, frame_counts=frame_counts))
    return rows


def route_scene_items(habitat, scene_idx, include_reasoning):
    scene_path = os.path.join(habitat.data_path, scene_idx)
    rows = []
    for route_idx in sorted(os.listdir(scene_path)):
        if os.path.isdir(os.path.join(scene_path, route_idx)):
            rows += route_gen.generate_single_route_plan_qa(habitat, scene_idx, route_idx, mode="medium",
                                                            include_reasoning=include_reasoning)
    return rows


def run_shard(args):
    shard_dir, _ = paths(args.task, args.split)
    os.makedirs(shard_dir, exist_ok=True)
    out = os.path.join(shard_dir, f"shard_{args.shard:03d}.jsonl")
    if os.path.exists(out + ".done"):
        print(f"{out} already done")
        return
    habitat = reid_gen.Habitat_Dataset(root_path=ROOT, section=TASKS[args.task]["section"], split=args.split)
    scenes = sorted(s for s in os.listdir(habitat.data_path) if os.path.isdir(os.path.join(habitat.data_path, s)))
    mine = [s for i, s in enumerate(scenes) if i % args.num_shards == args.shard]
    random.seed(args.seed * 1000 + args.shard)
    rng = random.Random(f"{args.seed}:{args.task}:cap:{args.shard}")
    include_reasoning = args.split == "train"
    print(f"{args.task} {args.split} shard {args.shard}/{args.num_shards}: {len(mine)} of {len(scenes)} scenes, "
          f"cap {args.per_scene_cap} per scene", flush=True)
    n_raw = n_kept = 0
    start = time.time()
    with open(out, "w") as f, open(out.replace(".jsonl", ".raw.jsonl"), "w") as f_raw:
        for k, scene in enumerate(mine):
            t = time.time()
            gen = reid_scene_items if args.task == "reid" else route_scene_items
            rows = gen(habitat, scene, include_reasoning)
            kept = cap_per_scene(rows, args.per_scene_cap, lambda r: answer_bucket(args.task, r), rng)
            for row in rows:
                f_raw.write(json.dumps(row) + "\n")
            for row in kept:
                f.write(json.dumps(row) + "\n")
            f.flush()
            f_raw.flush()
            n_raw += len(rows)
            n_kept += len(kept)
            print(f"[{k + 1}/{len(mine)}] {scene}: {len(rows)} generated, {len(kept)} kept in {time.time() - t:.1f}s "
                  f"(total {n_kept}/{n_raw}, {time.time() - start:.0f}s)", flush=True)
    open(out + ".done", "w").write(f"{len(mine)} scenes, {n_raw} generated, {n_kept} kept\n")


def run_merge(args):
    shard_dir, output = paths(args.task, args.split)
    if os.path.exists(output):
        raise SystemExit(f"FATAL: {output} exists, not overwriting")
    # The merged pool keeps every generated item (shard_NNN.raw.jsonl): build_train's
    # balanced_select fills each answer bucket round-robin over scenes, which spreads the
    # selected items evenly over scenes; a fixed per-scene cap here could only drop items.
    rows = []
    for shard in range(args.num_shards):
        path = os.path.join(shard_dir, f"shard_{shard:03d}.jsonl")
        if not os.path.exists(path + ".done"):
            raise SystemExit(f"FATAL: {path} not finished")
        source = path if args.use_capped else path.replace(".jsonl", ".raw.jsonl")
        rows += [json.loads(line) for line in open(source) if line.strip()]
    rng = random.Random(f"{args.seed}:{args.task}_pool")
    if args.task == "reid":
        tenths = [row.pop("_video_2_tenths") for row in rows]
        rebuilt = reoption_reid([r["question"] for r in rows], [r["answer"][3:] for r in rows], tenths, rng)
        for row, (options, answer) in zip(rows, rebuilt):
            row["options"], row["answer"] = options, answer
        stats = {}
    else:
        pools = [row.pop("_route_pool") for row in rows]
        rebuilt, stats = reoption_route_v3([r["answer"][3:] for r in rows], pools, rng)
        kept = []
        for row, (options, answer) in zip(rows, rebuilt):
            if options is not None:
                row["options"], row["answer"] = options, answer
                kept.append(row)
        print(f"route options rebuilt: kept {len(kept)}/{len(rows)}")
        rows = kept
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    scene_pos = -3 if args.task == "reid" else -4
    scenes = collections.Counter(r["video_list"][0]["video_path"].split("/")[scene_pos] for r in rows)
    buckets = collections.Counter(answer_bucket(args.task, r) for r in rows)
    print(f"{output}: {len(rows)} items from {len(scenes)} scenes (max per scene {max(scenes.values()) if scenes else 0}); "
          f"buckets {dict(buckets)}; with reasoning_cot {sum('reasoning_cot' in r for r in rows)}; "
          f"option stats {json.dumps(stats)[:500]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--per-scene-cap", type=int, default=30)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--use-capped", action="store_true", help="Merge the per-scene-capped shard files instead of all items.")
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    (run_merge if args.merge else run_shard)(args)
