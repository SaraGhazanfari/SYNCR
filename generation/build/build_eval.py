"""
Nested 500-item evaluation files, `_optv3c`.

For each of the 8 SYNCR tasks, 500 items are taken from the task's original
benchmark file -- same videos, questions and answers, so the new benchmark is
a subset of the published one -- and ordered so that lines 1-200 and lines
1-500 are each answer-bucket balanced, scene-stratified and answer-letter
balanced (generation/build/nested.py). Buckets: Counting {1, 2, 3+}; Route
{2, 3+ hops}; Numerical {tie, diff 1, diff 2, diff 3+} with ties at 1/5 and the
winner video interleaved inside each; the other tasks have one bucket.

Items whose (question, videos) occurs more than once in the original pool with
different answers are dropped before selection; identical repeats keep one.

Options are rebuilt for the 500 selected items: v2 builders for Sync, ReID,
Counting and Numerical (refitted to the 500); v3 builders for Ordering
(distractors follow the answers' order distribution), Route (hop mix fitted,
room pools pooled over items of the same route, start/goal rooms only at the
answers' rate) and Spatial (attributes and names matched to the answers');
Kinematic (control) keeps its options and only gets balanced letters.

`ordering_gap` is an added variant, not a benchmark subset: the same 500
CLEVRER validation videos as the `_optv3b` Ordering eval, re-cut into 4
equal-length segments with random 0.3-0.6 s gaps and a random start.

The Kubric eval pools shrink to their 500 items. Spatial is selected first;
Sync then prefers scenes already used by Spatial, so more scenes are released
for training (build_train.py reads the scenes of both files).

Run from the repo root (spatial reads Kubric metadata):
    python -m generation.build.build_eval --dry-run
    python -m generation.build.build_eval
"""
from utils import DATA_ROOT
import argparse
import collections
import random

from generation.build import reoption as v2
from generation.build.common import (check_invariants, file_rng, load_jsonl, refuse_existing, scene_of, sha256,
                                       suffixed, write_outputs)
from generation.build import tasks
from generation.build.nested import nested_order

SEED = 20260913
SUFFIX = "_optv3c"
EV = "json_files/eval/"
ORDERING_GAP_OUT = EV + "Temporal_Alignment/Clvr_Sequential_Ordering_gap" + SUFFIX + ".jsonl"
ORDERING_GAP_VIDEOS_FROM = EV + "Temporal_Alignment/Clvr_Sequential_Ordering_optv3b.jsonl"


def keep_options(rows, rng, args):
    return [(r["options"], r["answer"]) for r in rows], {}


TASKS = {  # task: (original eval file, option builder)
    "spatial": (EV + "Spatial_Tracking/Kubric_Cross_Video_Spatial_Measurement.jsonl", tasks.build_spatial_v3),
    "sync": (EV + "Temporal_Alignment/Kubric_Multi_Angle_Synchronization.jsonl", v2.per_row(v2.sync_options)),
    "ordering": (EV + "Temporal_Alignment/Clvr_Sequential_Ordering.jsonl", tasks.build_ordering_v3),
    "reid": (EV + "Spatial_Tracking/Habitat_Object_Re_identification.jsonl", v2.build_reid),
    "counting": (EV + "Holistic_Synthesis/Habitat_Object_Counting.jsonl", v2.build_counting),
    "route": (EV + "Holistic_Synthesis/Habitat_Route_Planning.jsonl", None),
    "numerical": (EV + "Comparative_Reasoning/Clvr_Numerical_Comparison.jsonl", v2.build_numerical),
    "kinematic": (EV + "Comparative_Reasoning/Clvr_Kinematic_Comparison.jsonl", keep_options),
}
ORDER = list(TASKS) + ["ordering_gap"]  # spatial before sync


OUT_SUFFIX = SUFFIX  # --suffix
SOURCES = {}  # --source task=path: select from this pool instead of the original benchmark file


def output_path(task):
    return ORDERING_GAP_OUT if task == "ordering_gap" else suffixed(TASKS[task][0], OUT_SUFFIX)


def build_task(task, args, spatial_scenes):
    src, builder = TASKS[task]
    src = SOURCES.get(task, src)
    rng = file_rng(args.seed, src)
    pool = load_jsonl(src)
    kept, dedupe = tasks.dedupe_rows(pool)
    route_pools = tasks.grouped_route_pools(pool) if task == "route" else None
    eligible = [i for i in kept if task != "route" or tasks.route_usable(pool[i], route_pools[i])]
    n_eligible = len(eligible)
    bucket_of, sub_of = tasks.BUCKETS.get(task, (lambda r: "all", None))
    if task == "sync" and spatial_scenes:
        preferred = [i for i in eligible if scene_of(pool[i]) in spatial_scenes]
        others = [i for i in eligible if scene_of(pool[i]) not in spatial_scenes]
        rng.shuffle(preferred)
        rng.shuffle(others)
        eligible = (preferred + others)[:args.n]
    order, report = nested_order(len(eligible), lambda j: bucket_of(pool[eligible[j]]),
                                 lambda j: scene_of(pool[eligible[j]]), rng, sizes=(200, args.n),
                                 sub_of=(lambda j: sub_of(pool[eligible[j]])) if sub_of else None,
                                 weights=tasks.BUCKET_WEIGHTS.get(task))
    selected = [eligible[j] for j in order]
    old = [pool[i] for i in selected]
    if task == "route":
        built, stats = tasks.build_route_v3(old, rng, pools=[route_pools[i] for i in selected])
    else:
        built, stats = builder(old, rng, args)
    dropped = [i for i, (opts, _) in zip(selected, built) if opts is None]
    assert not dropped, f"{task}: builder dropped pre-filtered items {dropped[:5]}"
    new = [dict(r, options=list(o), answer=a) for r, (o, a) in zip(old, built)]
    check_invariants(old, new)
    new = v2.balance_answer_letters(new, rng)
    check_invariants(old, new)
    scenes = sorted({scene_of(r) for r in new})
    letters = {f"first{k}": dict(sorted(collections.Counter(r["answer"][0] for r in new[:k]).items()))
               for k in (200, args.n)}
    kubric_note = f", scenes shared with spatial={len(set(scenes) & spatial_scenes)}" if task == "sync" and spatial_scenes else ""
    print(f"[{task}] {src} -> {output_path(task)}: {len(new)} of {len(pool)} (usable {n_eligible}), "
          f"scenes={len(scenes)}{kubric_note}")
    print(f"    duplicates {dedupe} | buckets {report} | letters {letters}")
    if stats:
        print(f"    option stats {stats}")
    meta = dict(task=task, script="generation/build/build_eval.py", seed=args.seed,
                rng=f"random.Random('{args.seed}:{src.split('/')[-1]}')", input=src, input_sha256=sha256(src),
                pool_size=len(pool), duplicates_removed=dedupe, usable=n_eligible, source_lines=selected,
                bucket_counts=report, bucket_weights=tasks.BUCKET_WEIGHTS.get(task), letter_counts=letters,
                scenes=scenes, option_stats=stats,
                ordering="nested: lines 1-200 and 1-500 bucket- and letter-balanced, scene-stratified")
    return output_path(task), new, meta, set(scenes)


def build_ordering_gap(args):
    """Same 500 validation videos and line order as the _optv3b Ordering eval, re-cut with gaps."""
    from generation.clevrer.clvr_temporal_alignment import CLEVRER_Temporal_Alignment
    source = load_jsonl(ORDERING_GAP_VIDEOS_FROM)
    dataset = CLEVRER_Temporal_Alignment(root_path=args.clevrer_root, split="validation",
                                         sample_num=args.clevrer_val_videos)
    random.seed(f"{args.seed}:ordering_gap")  # get_seq_order_QA draws from the module RNG
    rng = random.Random(f"{args.seed}:ordering_gap:options")
    rows = v2.balance_answer_letters(tasks.gapped_ordering_items(dataset, source, rng), rng)
    starts = [min(v["start_sec"] for v in r["video_list"]) for r in rows]
    gaps = []
    for r in rows:
        spans = sorted((v["start_sec"], v["end_sec"]) for v in r["video_list"])
        gaps += [round(b[0] - a[1], 2) for a, b in zip(spans, spans[1:])]
    letters = {f"first{k}": dict(sorted(collections.Counter(r["answer"][0] for r in rows[:k]).items())) for k in (200, 500)}
    stats = {"first_segment_start_sec": [min(starts), max(starts)], "gap_sec": [min(gaps), max(gaps)],
             "segment_len_sec": [min(round(v["end_sec"] - v["start_sec"], 2) for r in rows for v in r["video_list"]),
                                 max(round(v["end_sec"] - v["start_sec"], 2) for r in rows for v in r["video_list"])]}
    print(f"[ordering_gap] videos of {ORDERING_GAP_VIDEOS_FROM} -> {ORDERING_GAP_OUT}: {len(rows)} items, "
          f"letters {letters}, {stats}")
    meta = dict(task="ordering_gap", script="generation/build/build_eval.py", seed=args.seed,
                videos_from=ORDERING_GAP_VIDEOS_FROM, videos_from_sha256=sha256(ORDERING_GAP_VIDEOS_FROM),
                gap_range=tasks.GAP_RANGE, max_start_offset=tasks.MAX_START_OFFSET, letter_counts=letters,
                segment_stats=stats, note="new items (same videos, new segments); not a subset of the benchmark")
    return ORDERING_GAP_OUT, rows, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", nargs="+", default=ORDER, choices=ORDER)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true", help="Build and verify in memory; write nothing.")
    parser.add_argument("--reid-video-seconds", type=float, default=20.0)
    parser.add_argument("--kubric-root", default=f"{DATA_ROOT}/kubric_data")
    parser.add_argument("--clevrer-root", default=f"{DATA_ROOT}/clvr/")
    parser.add_argument("--clevrer-val-videos", type=int, default=5000)
    parser.add_argument("--suffix", default=SUFFIX, help="Output suffix (ordering_gap keeps its own path).")
    parser.add_argument("--source", nargs="*", default=[], metavar="TASK=PATH",
                        help="Select the task's items from PATH instead of the original benchmark file.")
    args = parser.parse_args()
    global OUT_SUFFIX
    OUT_SUFFIX = args.suffix
    for spec in args.source:
        task, path = spec.split("=", 1)
        SOURCES[task] = path

    run = [t for t in ORDER if t in args.tasks]
    outputs = [output_path(t) for t in run]
    if not args.dry_run:
        refuse_existing(outputs + [o + ".meta.json" for o in outputs])

    spatial_scenes = None
    if "sync" in run and "spatial" not in run:
        try:
            spatial_scenes = {scene_of(r) for r in load_jsonl(suffixed(TASKS["spatial"][0], SUFFIX))}
        except FileNotFoundError:
            print("note: no spatial eval file with this suffix; sync selection ignores spatial scenes")
    pending = []
    for task in run:
        if task == "ordering_gap":
            pending.append(build_ordering_gap(args))
            continue
        path, rows, meta, scenes = build_task(task, args, spatial_scenes)
        if task == "spatial":
            spatial_scenes = scenes
        pending.append((path, rows, meta))
    if args.dry_run:
        print("dry run: nothing written")
        return
    for path, rows, meta in pending:
        write_outputs(path, rows, meta)
    print(f"wrote {len(pending)} eval files (+ .meta.json), suffix {SUFFIX}")


if __name__ == "__main__":
    main()
