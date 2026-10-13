"""
Balanced training files, `_optv3c`.

Up to 2000 items per task, answer buckets as in the eval files (Counting
{1, 2, 3+}, Route {2, 3+ hops}, Numerical {tie, diff 1, 2, 3+} with ties at 1/5
and the winner video interleaved) and answer-letter balanced. When a bucket is
short the others are subsampled in proportion -- nothing is duplicated -- and
the resulting size is reported. Options come from the same builders as the
eval files. Every pool is deduplicated by (question, videos) first.

Sources:
- Ordering, Numerical: CLEVRER train split (all 10,000 videos), current
  generators (v3-cot traces); Ordering distractors re-drawn with
  reoption_ordering_v3.
- Ordering (gapped variant): the same 2000 training videos as the `_optv3b`
  Ordering file, 4 equal-length segments with random 0.3-0.6 s gaps.
- Kinematic: CLEVRER train split, random non-consecutive pairs of
  clear-margin videos, margin 0.25, each video used at most twice.
- ReID, Counting, Route: union of the task's existing Habitat training files
  (for Counting including the uncapped pool from generate_counting_pool.SBATCH),
  items carrying a v3-cot trace (reasoning_cot) only. A (question, videos)
  group with different answers is kept only if the benchmark-setting file
  (Habitat_Object_Re_identification.jsonl etc.) gives it exactly one answer and
  a v3-cot row agrees; otherwise the whole group is dropped. Route room pools
  are pooled over items of the same route.
- Sync: Kubric scenes used by neither Kubric eval file ("released"), several
  independent crops per scene; reasoning '-'.
- Spatial: released scenes, all 6 camera orderings, every valid target per
  event exit, scenes with duplicate object names kept (only uniquely named
  objects used), at most --spatial-per-scene items per scene; reasoning '-'.

Run from the repo root, after build_eval.py:
    python -m generation.build.build_train --tasks reid counting route --dry-run
    python -m generation.build.build_train
"""
from utils import DATA_ROOT
import argparse
import collections
import gc
import glob
import json
import os
import random

from generation.option_balance import reoption_counting, reoption_numerical, reoption_ordering_v3, reoption_reid
from generation.build import reoption as v2
from generation.build.common import body, check_invariants, load_jsonl, refuse_existing, scene_of, suffixed, write_outputs
from generation.build import tasks
from generation.build.build_eval import SUFFIX, TASKS as EVAL_TASKS
from generation.build.nested import balanced_select, bucket_counts

SEED = 20260913
TR = "json_files/train/"
OUT_NAMES = {
    "ordering": "Clvr_Sequential_Ordering", "ordering_gap": "Clvr_Sequential_Ordering_gap",
    "numerical": "Clvr_Numerical_Comparison", "kinematic": "Clvr_Kinematic_Comparison",
    "reid": "Habitat_Object_Re_identification", "counting": "Habitat_Object_Counting",
    "route": "Habitat_Route_Planning", "sync": "Kubric_Multi_Angle_Sync",
    "spatial": "Kubric_Spatial_Measurement"}
OUT = {task: TR + name + SUFFIX + ".jsonl" for task, name in OUT_NAMES.items()}
# --pool-files task=substr[,substr]: that task's Habitat pool is only the training files whose
# name contains one of the substrings (e.g. the full-scene pools), instead of every file.
POOL_FILES = {}
ORDERING_GAP_VIDEOS_FROM = TR + "Clvr_Sequential_Ordering_optv3b.jsonl"
HABITAT_POOLS = {"reid": "Habitat_Object_Re_identification", "counting": "Habitat_Object_Counting",
                 "route": "Habitat_Route_Planning"}
KEEP_KEYS = ["scene_idx", "route_idx", "question_type", "question", "options", "answer", "video_list",
             "reasoning", "reasoning_cof", "reasoning_cot"]
KUBRIC_ORDERINGS = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
_clevrer_cache = {}


def select_indices(rows, rng, n, task):
    bucket_of, sub_of = tasks.BUCKETS.get(task, (None, None))
    return balanced_select(len(rows), (lambda i: bucket_of(rows[i])) if bucket_of else (lambda i: "all"),
                           lambda i: scene_of(rows[i]), rng, n,
                           sub_of=(lambda i: sub_of(rows[i])) if sub_of else None,
                           weights=tasks.BUCKET_WEIGHTS.get(task))


def dedupe_then_select(pool, rng, n, task):
    kept, dedupe = tasks.dedupe_rows(pool)
    rows = [pool[i] for i in kept]
    idx, report = select_indices(rows, rng, n, task)
    report["duplicates_removed"] = dedupe
    return [rows[i] for i in idx], report


def clevrer(args, cls_name):
    if cls_name not in _clevrer_cache:
        _clevrer_cache.clear()
        gc.collect()
        if cls_name == "temporal":
            from generation.clevrer.clvr_temporal_alignment import CLEVRER_Temporal_Alignment as cls
        else:
            from generation.clevrer.clvr_comparative_reasoning import CLEVRER_Comparative_Reasoning as cls
        _clevrer_cache[cls_name] = cls(root_path=args.clevrer_root, split="train", sample_num=args.clevrer_videos)
    return _clevrer_cache[cls_name]


def habitat_pool(task):
    """One row per (question, videos) from the task's existing training files (not _opt*), v3-cot rows only."""
    base = HABITAT_POOLS[task] + ".jsonl"
    files = sorted(f for f in glob.glob(TR + HABITAT_POOLS[task] + "*.jsonl") if "_opt" not in f)
    # json_files/train now holds both the old few-scene files and the full-scene pools; pooling all of
    # them silently brings the old scene imbalance back, so the pool must be named explicitly.
    assert task in POOL_FILES, f"{task}: pass --pool-files {task}=<substring> (e.g. scenepool_v3c)"
    if task in POOL_FILES:
        files = [f for f in files if any(s in os.path.basename(f) for s in POOL_FILES[task])]
        missing = [s for s in POOL_FILES[task] if not any(s in os.path.basename(f) for f in files)]
        assert not missing, f"{task}: no pool file matches {missing}"
    groups = collections.defaultdict(list)
    for f in files:
        for r in load_jsonl(f):
            groups[tasks.item_key(r)].append((os.path.basename(f), r))
    rows, report = [], collections.Counter()
    for members in groups.values():
        cot = [r for _, r in members if "reasoning_cot" in r]
        if not cot:
            report["groups_without_v3_cot_row"] += 1
            continue
        answers = {body(r["answer"]) for _, r in members}
        if len(answers) == 1:
            rows.append({k: cot[0][k] for k in KEEP_KEYS if k in cot[0]})
            report["copies_merged"] += len(members) - 1
            continue
        base_answers = {body(r["answer"]) for fn, r in members if fn == base}
        match = [r for r in cot if len(base_answers) == 1 and body(r["answer"]) in base_answers]
        if match:
            rows.append({k: match[0][k] for k in KEEP_KEYS if k in match[0]})
            report["conflicting_groups_kept_benchmark_file_answer"] += 1
        else:
            report["conflicting_groups_dropped"] += 1
    return rows, files, dict(report)


def released_scenes(args):
    reserved = set()
    for task in ("sync", "spatial"):
        reserved |= {scene_of(r) for r in load_jsonl(suffixed(EVAL_TASKS[task][0], SUFFIX))}
    scenes = sorted((d for d in os.listdir(args.kubric_root) if d.startswith("scene_")), key=lambda s: int(s.split("_")[1]))
    return [s for s in scenes if s not in reserved], reserved


# ---------------------------------------------------------------- per task: (rows with options, report)

def build_ordering(args, rng):
    ds = clevrer(args, "temporal")
    picks = sorted(rng.sample(range(len(ds.data_list)), min(args.n, len(ds.data_list))))
    rows = [ds.get_seq_order_QA(i, gap_time=0.0, total_frames=128, k=4) for i in picks]
    built = reoption_ordering_v3([body(r["answer"]) for r in rows], rng)
    rows = [dict(r, options=o, answer=a) for r, (o, a) in zip(rows, built)]
    kept, dedupe = tasks.dedupe_rows(rows)
    return [rows[i] for i in kept], {"videos_scanned": len(ds.data_list), "size": len(kept), "duplicates_removed": dedupe,
                                     "distinct_answer_orders": len({body(r["answer"]) for r in rows})}


def build_ordering_gap(args, rng):
    ds = clevrer(args, "temporal")
    if args.ordering_gap_exclude:
        # Videos not used by the given (contiguous Ordering) training file, so a mix of both
        # variants does not show the same video twice.
        used = {int(tasks.clevrer_id(r)) for r in load_jsonl(args.ordering_gap_exclude)}
        free = [i for i, d in enumerate(ds.data_list) if int(next(iter(d))) not in used]
        picks = sorted(rng.sample(free, min(args.n, len(free))))
        rows = [ds.get_seq_order_QA(i, total_frames=128, k=4, gap_range=tasks.GAP_RANGE,
                                    max_start_offset=tasks.MAX_START_OFFSET) for i in picks]
        built = reoption_ordering_v3([body(r["answer"]) for r in rows], rng)
        rows = [dict(r, options=o, answer=a) for r, (o, a) in zip(rows, built)]
        assert not {int(tasks.clevrer_id(r)) for r in rows} & used
    else:
        source = load_jsonl(ORDERING_GAP_VIDEOS_FROM)
        rows = tasks.gapped_ordering_items(ds, source, rng)
    gaps = [round(b[0] - a[1], 2) for r in rows
            for spans in [sorted((v["start_sec"], v["end_sec"]) for v in r["video_list"])]
            for a, b in zip(spans, spans[1:])]
    return rows, {"videos_from": ORDERING_GAP_VIDEOS_FROM, "size": len(rows), "gap_range": tasks.GAP_RANGE,
                  "gap_sec": [min(gaps), max(gaps)],
                  "first_segment_start_sec": [min(min(v["start_sec"] for v in r["video_list"]) for r in rows),
                                              max(min(v["start_sec"] for v in r["video_list"]) for r in rows)]}


def build_numerical(args, rng):
    ds = clevrer(args, "comparative")
    pool = [ds.get_collision_count_QA(i, nframes=32, k=2, time_mode="timestamp") for i in range(len(ds.data_list))]
    chosen, report = dedupe_then_select(pool, rng, args.n, "numerical")
    built, stats = reoption_numerical([r["collision_count"] for r in chosen], rng)
    report.update(pool=len(pool), option_stats=stats)
    return [dict(r, options=o, answer=a) for r, (o, a) in zip(chosen, built)], report


def build_kinematic(args, rng):
    from generation.clevrer.clvr_comparative_reasoning import nonconsecutive_kinematic_items
    ds = clevrer(args, "comparative")
    rows = nonconsecutive_kinematic_items(ds, args.n, threshold=args.kinematic_threshold, max_uses=2, rng=rng)
    kept, dedupe = tasks.dedupe_rows(rows)
    rows = [rows[i] for i in kept]
    videos = collections.Counter(v["video_path"] for r in rows for v in r["video_list"])
    return rows, {"videos_scanned": len(ds.data_list), "size": len(rows), "threshold": args.kinematic_threshold,
                  "duplicates_removed": dedupe, "max_uses_per_video": max(videos.values()) if videos else 0,
                  "answer_video": dict(collections.Counter(body(r["answer"]).split(" -- ")[0] for r in rows))}


def counting_general_first(rows, rng, n):
    """Counting with equal {1, 2, 3+} buckets, each filled from general-section items first.

    route_plan items (the same houses along other camera paths) only fill what a
    bucket's general items cannot. Within each source the selection is
    scene round-robin (balanced_select), so it spreads over as many scenes as possible.
    """
    is_general = lambda r: "/general/" in r["video_list"][0]["video_path"]
    by = collections.defaultdict(lambda: {"general": [], "route_plan": []})
    for r in rows:
        by[tasks.counting_bucket(r)]["general" if is_general(r) else "route_plan"].append(r)
    per_bucket = min(n // len(by), min(len(v["general"]) + len(v["route_plan"]) for v in by.values()))
    chosen, available, taken = [], {}, {}
    for b in sorted(by):
        need, taken[b] = per_bucket, {}
        for src in ("general", "route_plan"):
            pool = by[b][src]
            idx = []
            if need > 0 and pool:
                idx, _ = balanced_select(len(pool), lambda i: "all", lambda i: scene_of(pool[i]), rng, need)
            chosen += [pool[i] for i in idx]
            taken[b][src] = len(idx)
            need -= len(idx)
        available[b] = {s: len(v) for s, v in by[b].items()}
    rng.shuffle(chosen)
    report = {"size": len(chosen), "per_bucket": per_bucket, "available": available, "taken": taken,
              "scenes": {src: len({scene_of(r) for r in chosen if is_general(r) == (src == "general")})
                         for src in ("general", "route_plan")}}
    return chosen, report


def build_habitat(task, args, rng):
    pool, files, dedupe = habitat_pool(task)
    pools = tasks.grouped_route_pools(pool) if task == "route" else None
    usable = [i for i, r in enumerate(pool) if task != "route" or tasks.route_usable(r, pools[i])]
    rows_usable = [pool[i] for i in usable]
    if task == "counting" and args.counting_general_first:
        chosen, report = counting_general_first(rows_usable, rng, args.n)
    else:
        idx, report = select_indices(rows_usable, rng, args.n, task)
        chosen = [rows_usable[j] for j in idx]
    if task == "reid":
        tenths = int(round(args.reid_video_seconds * 10))
        built = reoption_reid([r["question"] for r in chosen], [body(r["answer"]) for r in chosen],
                              [tenths] * len(chosen), rng)
        stats = {"video_2_seconds": args.reid_video_seconds}
    elif task == "counting":
        built, stats = reoption_counting([int(body(r["answer"])) for r in chosen], rng)
    else:
        built, stats = tasks.build_route_v3(chosen, rng, pools=[pools[usable[j]] for j in idx])
        assert all(o is not None for o, _ in built), "route builder dropped a pre-filtered item"
    rows = [dict(r, options=o, answer=a) for r, (o, a) in zip(chosen, built)]
    check_invariants(chosen, rows)
    report.update(pool_files=files, pool=len(pool), duplicates=dedupe, usable=len(usable), option_stats=stats)
    return rows, report


def build_sync(args, rng):
    from generation.kubric.base_kubric import Kubric_Dataset
    scenes, reserved = released_scenes(args)
    kubric = Kubric_Dataset(path=args.kubric_root)
    pool = []
    for scene in scenes:
        for _ in range(args.sync_crops_per_scene):
            sample = kubric.get_time_sync_qa(scene_idx=scene)
            sample["reasoning"] = "-"
            pool.append(sample)
    chosen, report = dedupe_then_select(pool, rng, args.n, "sync")
    report.update(released_scenes=len(scenes), reserved_scenes=len(reserved), pool=len(pool),
                  crops_per_scene=args.sync_crops_per_scene)
    return chosen, report


def build_spatial(args, rng):
    from generation.kubric.base_kubric import Kubric_Dataset, finalize_spatial_options
    scenes, reserved = released_scenes(args)
    kubric = Kubric_Dataset(path=args.kubric_root)
    pool, per_scene, failed = [], collections.Counter(), 0
    for scene in scenes:
        kubric.instance_data.clear()
        items = []
        for cams in KUBRIC_ORDERINGS:
            try:
                items.extend(kubric.get_cross_video_exit_qa(scene_idx=scene, cams=list(cams), all_targets=True,
                                                            sample_by_exit_time=False, allow_duplicate_names=True))
            except Exception as e:  # unreadable metadata: skip the ordering, count it
                failed += 1
                print(f"  spatial {scene} cams {cams}: {type(e).__name__}: {e}")
        if len(items) > args.spatial_per_scene:
            items = rng.sample(items, args.spatial_per_scene)
        per_scene[len(items)] += 1
        pool.extend(items)
    chosen, report = dedupe_then_select(pool, rng, args.n, "spatial")
    stats = finalize_spatial_options(chosen, rng)
    for r in chosen:
        r["video_list"] = [dict(v) for v in r["video_list"]]
        r["reasoning"] = "-"
    report.update(released_scenes=len(scenes), reserved_scenes=len(reserved), pool=len(pool),
                  items_per_scene=dict(sorted(per_scene.items())), failed_orderings=failed,
                  cap_per_scene=args.spatial_per_scene, option_stats=stats)
    return chosen, report


BUILDERS = {
    "ordering": build_ordering, "ordering_gap": build_ordering_gap, "numerical": build_numerical,
    "kinematic": build_kinematic,
    "reid": lambda args, rng: build_habitat("reid", args, rng),
    "counting": lambda args, rng: build_habitat("counting", args, rng),
    "route": lambda args, rng: build_habitat("route", args, rng),
    "sync": build_sync, "spatial": build_spatial,
}
ORDER = list(BUILDERS)


def finish(task, rows, rng):
    n_options = 5 if task == "numerical" else 4
    for r in rows:
        bodies = [body(o) for o in r["options"]]
        assert len(bodies) == n_options and len(set(bodies)) == n_options and r["answer"] in r["options"], \
            (task, r["options"], r["answer"])
    assert not tasks.dedupe_rows(rows)[1], f"{task}: duplicate items after selection"
    return v2.balance_answer_letters(rows, rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", nargs="+", default=ORDER, choices=ORDER)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true", help="Build and verify in memory; write nothing.")
    parser.add_argument("--clevrer-root", default=f"{DATA_ROOT}/clvr/")
    parser.add_argument("--clevrer-videos", type=int, default=10000)
    parser.add_argument("--kinematic-threshold", type=float, default=0.25)
    parser.add_argument("--kubric-root", default=f"{DATA_ROOT}/kubric_data")
    parser.add_argument("--sync-crops-per-scene", type=int, default=2)
    parser.add_argument("--spatial-per-scene", type=int, default=8)
    parser.add_argument("--reid-video-seconds", type=float, default=20.0)
    parser.add_argument("--suffix", default=SUFFIX, help="Training-file suffix (eval files keep build_eval's).")
    parser.add_argument("--pool-files", nargs="*", default=[], metavar="TASK=SUBSTR[,SUBSTR]")
    parser.add_argument("--ordering-gap-exclude", default=None,
                        help="Training file whose CLEVRER videos the gapped Ordering items must not use.")
    parser.add_argument("--counting-general-first", action="store_true",
                        help="Fill each Counting bucket from general-section items before route_plan items.")
    args = parser.parse_args()
    OUT.update({task: TR + name + args.suffix + ".jsonl" for task, name in OUT_NAMES.items()})
    for spec in args.pool_files:
        task, substrings = spec.split("=", 1)
        POOL_FILES[task] = substrings.split(",")

    run = [t for t in ORDER if t in args.tasks]
    if not args.dry_run:
        refuse_existing([OUT[t] for t in run] + [OUT[t] + ".meta.json" for t in run])
    summary = {}
    for task in run:
        random.seed(f"{args.seed}:{task}")  # the CLEVRER/Kubric generators draw from the module RNG
        rng = random.Random(f"{args.seed}:{task}:select")
        rows, report = BUILDERS[task](args, rng)
        rows = finish(task, rows, rng)
        bucket_of, sub_of = tasks.BUCKETS.get(task, (lambda r: "all", None))
        final = {"size": len(rows), "scenes": len({scene_of(r) for r in rows}),
                 "buckets": bucket_counts(range(len(rows)), lambda i: bucket_of(rows[i]),
                                          (lambda i: sub_of(rows[i])) if sub_of else None),
                 "letters": dict(sorted(collections.Counter(r["answer"][0] for r in rows).items()))}
        summary[task] = final
        print(f"[{task}] -> {OUT[task]}: {json.dumps(final)}")
        print(f"    report: {json.dumps(report, default=str)[:2000]}")
        if not args.dry_run:
            write_outputs(OUT[task], rows, dict(task=task, script="generation/build/build_train.py",
                                                seed=args.seed, args=vars(args), final=final, report=report))
    print("SUMMARY " + json.dumps(summary))
    if args.dry_run:
        print("dry run: nothing written")


if __name__ == "__main__":
    main()
