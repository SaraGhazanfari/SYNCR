"""
Rebuild the wrong options of existing SYNCR items into `_optv2` files.

For one task this re-options the eval file and every training file that
carries the task (SFT variants and GRPO source files), then rewrites the
options inside the already-built GRPO prompt files (frames untouched). Every
field other than `options`/`answer` is kept, the answer text is kept, and
option construction is the same code the fixed generators use
(generation/option_balance.py, generation/clevrer/utils.py). The eval file is shuffled
once, stratified by scene, so its first 200 lines (what eval/evaluate.py
--sample-num 200 scores) span many scenes; training files keep their line
order. Route Planning drops broken items (see reoption_route); the sidecar
records which input line each output line came from, and why lines were
dropped.

Run from the repo root on the cluster (spatial reads Kubric metadata and
needs numpy):
    python -m generation.build.reoption --task ordering --dry-run
    python -m generation.build.reoption --task ordering
Tasks: ordering sync reid counting route spatial numerical.
"""
from utils import DATA_ROOT
import argparse
import collections
import json
import os
import re

from generation.clevrer.utils import sample_order_distractors
from generation.option_balance import (letter_options, numerical_answer_text, parse_route_option, reoption_counting,
                                 reoption_numerical, reoption_reid, reoption_route,
                                 shuffled_letter_options, spatial_distractors, sync_distractors,
                                 sync_option_text)
from generation.build.common import (SUFFIX, body, check_invariants, file_rng, load_jsonl,
                                       refuse_existing, scene_of, sha256, stratified_shuffle,
                                       suffixed, video_path, write_outputs)

SEED = 20260912
EV = "json_files/eval/"
TR = "json_files/train/"


# ---------------------------------------------------------------------------
# Per-task option builders: rows -> ([(options, answer) or (None, reason)], stats)
# ---------------------------------------------------------------------------

def per_row(fn):
    def build(rows, rng, args):
        return [fn(r, rng) for r in rows], {}
    return build


def balance_answer_letters(rows, rng):
    """Move each answer to a position drawn from shuffled blocks of all positions.

    Answer letters are then exactly balanced in every prefix of whole blocks
    (e.g. 50/50/50/50 in the first 200 eval lines) instead of only on average;
    the other options keep their random order around it.
    """
    out, block = [], []
    for r in rows:
        n = len(r["options"])
        if not block:
            block = list(range(n))
            rng.shuffle(block)
        target = block.pop()
        bodies = [body(o) for o in r["options"]]
        answer = body(r["answer"])
        current = bodies.index(answer)
        bodies[current], bodies[target] = bodies[target], bodies[current]
        options, lettered = letter_options(bodies, answer)
        out.append(dict(r, options=options, answer=lettered))
    return out


def ordering_options(row, rng):
    answer = body(row["answer"])
    correct = tuple(int(tok.strip().split()[-1]) for tok in answer.split("->"))
    assert " -> ".join(f"Video {i}" for i in correct) == answer, answer
    if "video_list" in row:
        assert sorted(correct) == list(range(1, len(row["video_list"]) + 1)), answer
    distractors = [" -> ".join(f"Video {i}" for i in p)
                   for p in sample_order_distractors(correct, len(row["options"]) - 1, rng)]
    return shuffled_letter_options(answer, distractors, rng)


def sync_options(row, rng):
    answer = body(row["answer"])
    if "video_list" in row:  # GRPO prompt rows carry no video_list
        v = row["video_list"]
        assert all(x["end_frame"] - x["start_frame"] == 36 and 0 <= x["start_frame"] <= 24 for x in v), v
        assert answer == sync_option_text((v[1]["start_frame"] - v[0]["start_frame"]) / 12.0,
                                          (v[2]["start_frame"] - v[0]["start_frame"]) / 12.0), answer
    return shuffled_letter_options(answer, sync_distractors(answer, rng), rng)


def build_reid(rows, rng, args):
    tenths = int(round(args.reid_video_seconds * 10))
    out = reoption_reid([r["question"] for r in rows], [body(r["answer"]) for r in rows], [tenths] * len(rows), rng)
    return out, {"video_2_seconds": args.reid_video_seconds}


def build_counting(rows, rng, args):
    return reoption_counting([int(body(r["answer"])) for r in rows], rng)


def build_route(rows, rng, args):
    pools = []
    for r in rows:
        names = []
        for o in r["options"]:
            parsed = parse_route_option(body(o))
            if parsed:
                names.extend(parsed[1])
        pools.append(names)
    return reoption_route([body(r["answer"]) for r in rows], pools, rng)


def build_numerical(rows, rng, args):
    for r in rows:
        assert body(r["answer"]) == numerical_answer_text(r["collision_count"]), r["answer"]
    return reoption_numerical([r["collision_count"] for r in rows], rng)


def build_spatial(rows, rng, args):
    from generation.kubric.base_kubric import Kubric_Dataset
    kubric = Kubric_Dataset(path=args.kubric_root)
    out, stats = [], collections.Counter()
    for r in rows:
        scene = r["scene_idx"]
        if scene not in kubric.instance_data and len(kubric.instance_data) >= 8:
            kubric.instance_data.clear()  # metadata is large; keep a few scenes
        kubric._get_raw_item(scene)
        names = [kubric.get_object_name(scene, i) for i in range(len(kubric.instance_data[scene]["cam0"]))]
        assert len(set(names)) == len(names), scene
        cams = [int(re.search(r"cam(\d+)\.mp4$", video_path(v)).group(1)) for v in r["video_list"]]
        visibility = kubric.get_visibility_for_all_objects(scene, len(cams), cams=cams)
        ranked = kubric.rank_objects_by_distance(scene, names.index(r["target_object"]), names.index(r["event_object"]),
                                                 int(r["exit_frame"]), visibility, len(cams))
        ranked_names = [names[o] for o, _ in ranked]
        answer = body(r["answer"])
        stats["answer_is_recomputed_closest"] += bool(ranked_names) and ranked_names[0] == answer
        distractors = spatial_distractors([x for x in ranked_names if x != answer], answer,
                                          [r["target_object"], r["event_object"]], names, rng)
        old_bodies = [body(o) for o in r["options"] if o != r["answer"]]
        stats["items_with_fake_old_distractor"] += any(x not in names for x in old_bodies)
        stats["items_keeping_all_real_old_distractors"] += all(x in distractors for x in old_bodies if x in names)
        stats["synthetic_new_distractors"] += sum(x not in names for x in distractors)
        out.append(shuffled_letter_options(answer, distractors, rng))
    stats["items"] = len(rows)
    return out, dict(stats)


TASKS = {
    "ordering": dict(
        build=per_row(ordering_options),
        eval=EV + "Temporal_Alignment/Clvr_Sequential_Ordering.jsonl",
        train=["Clvr_Sequential_Ordering.jsonl", "Clvr_Sequential_Ordering_v3.jsonl",
               "Clvr_Sequential_Ordering_cotime.jsonl"],
        grpo=[("grpo_sequential_ordering.jsonl", "Clvr_Sequential_Ordering.jsonl"),
              ("grpo_sequential_ordering_4f.jsonl", "Clvr_Sequential_Ordering.jsonl")]),
    "sync": dict(
        build=per_row(sync_options),
        eval=EV + "Temporal_Alignment/Kubric_Multi_Angle_Synchronization.jsonl",
        train=["Kubric_Multi_Angle_Sync_newscenes.jsonl", "Kubric_Multi_Angle_Sync_grpo.jsonl"],
        # grpo_kubric_multi_angle_sync(_5f) match no raw file still on disk; their
        # options are rebuilt from the prompt text alone (sync needs only the answer).
        grpo=[("grpo_kubric_multi_angle_sync.jsonl", None),
              ("grpo_kubric_multi_angle_sync_5f.jsonl", None),
              ("grpo_kubric_multi_angle_sync_v2.jsonl", "Kubric_Multi_Angle_Sync_grpo.jsonl")]),
    "reid": dict(
        build=build_reid,
        eval=EV + "Spatial_Tracking/Habitat_Object_Re_identification.jsonl",
        train=["Habitat_Object_Re_identification.jsonl", "Habitat_Object_Re_identification_grpo.jsonl",
               "Habitat_Object_Re_identification_v3.jsonl", "Habitat_Object_Re_identification_cotime.jsonl"],
        grpo=[("grpo_habitat_reid.jsonl", "Habitat_Object_Re_identification.jsonl"),
              ("grpo_habitat_reid_v2.jsonl", "Habitat_Object_Re_identification_grpo.jsonl")]),
    "counting": dict(
        build=build_counting,
        eval=EV + "Holistic_Synthesis/Habitat_Object_Counting.jsonl",
        train=["Habitat_Object_Counting.jsonl", "Habitat_Object_Counting_v2.jsonl", "Habitat_Object_Counting_v3.jsonl",
               "Habitat_Object_Counting_cotime.jsonl", "Habitat_Object_Counting_grpo.jsonl"],
        grpo=[("grpo_habitat_counting.jsonl", "Habitat_Object_Counting.jsonl"),
              ("grpo_habitat_counting_5f.jsonl", "Habitat_Object_Counting.jsonl")]),
    "route": dict(
        build=build_route,
        eval=EV + "Holistic_Synthesis/Habitat_Route_Planning.jsonl",
        train=["Habitat_Route_Planning.jsonl", "Habitat_Route_Planning_v3.jsonl", "Habitat_Route_Planning_cotime.jsonl"],
        grpo=[("grpo_habitat_route.jsonl", "Habitat_Route_Planning.jsonl"),
              ("grpo_habitat_route_5f.jsonl", "Habitat_Route_Planning.jsonl")]),
    "spatial": dict(
        build=build_spatial,
        eval=EV + "Spatial_Tracking/Kubric_Cross_Video_Spatial_Measurement.jsonl",
        train=["Kubric_Spatial_Measurement_newscenes.jsonl"],
        grpo=[("grpo_kubric_spatial_measurement.jsonl", "Kubric_Spatial_Measurement_newscenes.jsonl")]),
    "numerical": dict(
        build=build_numerical,
        eval=EV + "Comparative_Reasoning/Clvr_Numerical_Comparison.jsonl",
        train=["Clvr_Numerical_Comparison.jsonl", "Clvr_Numerical_Comparison_v3.jsonl",
               "Clvr_Numerical_Comparison_cotime.jsonl", "Clvr_Numerical_Comparison_grpo.jsonl"],
        grpo=[("grpo_numerical_comparison.jsonl", "Clvr_Numerical_Comparison_grpo.jsonl")]),
}


# ---------------------------------------------------------------------------
# Raw files
# ---------------------------------------------------------------------------

def rebuild_raw(task, src, is_eval, args):
    rng = file_rng(args.seed, src)
    old = load_jsonl(src)
    built, stats = TASKS[task]["build"](old, rng, args)
    assert len(built) == len(old)
    kept = [i for i, (opts, _) in enumerate(built) if opts is not None]
    dropped = collections.defaultdict(list)
    for i, (opts, reason) in enumerate(built):
        if opts is None:
            dropped[reason].append(i)
    new = [dict(old[i], options=built[i][0], answer=built[i][1]) for i in kept]
    check_invariants([old[i] for i in kept], new)
    source_lines = kept
    if is_eval:
        perm = stratified_shuffle(len(new), lambda j: scene_of(new[j]), rng)
        new = [new[j] for j in perm]
        source_lines = [kept[j] for j in perm]
    new = balance_answer_letters(new, rng)
    check_invariants([old[i] for i in source_lines], new)
    drop_note = f", dropped {sum(map(len, dropped.values()))} {dict((k, len(v)) for k, v in dropped.items())}" if dropped else ""
    print(f"[{task}] {src} -> {suffixed(src)}: {len(new)}/{len(old)} items"
          f"{' (shuffled, stratified by scene)' if is_eval else ''}{drop_note}")
    if stats:
        print(f"    stats: {json.dumps(stats)}")
    meta = dict(task=task, script="generation/build/reoption.py", seed=args.seed,
                rng=f"random.Random('{args.seed}:{os.path.basename(src)}')",
                input=src, input_sha256=sha256(src), shuffled_by_scene=is_eval,
                answer_letters="balanced in shuffled blocks of len(options)",
                source_lines=source_lines, dropped=dict(dropped), stats=stats)
    return new, source_lines, meta


# ---------------------------------------------------------------------------
# GRPO prompt files
# ---------------------------------------------------------------------------

def _options_block(options):
    return "\nOptions:\n" + "\n".join(options) + "\n"


def rewrite_prompt(row, old_options, new_options, new_answer):
    """Swap the options block of a GRPO prompt row and its answer; nothing else changes."""
    old_block, new_block = _options_block(old_options), _options_block(new_options)
    row = json.loads(json.dumps(row))
    hits = [c for m in row["prompt"] for c in m["content"] if c.get("type") == "text" and old_block in c["text"]]
    assert len(hits) == 1 and hits[0]["text"].count(old_block) == 1, "options block not found exactly once"
    hits[0]["text"] = hits[0]["text"].replace(old_block, new_block)
    row["answer"] = new_answer
    return row


def prompt_options(row):
    text = [c["text"] for m in row["prompt"] for c in m["content"]
            if c.get("type") == "text" and "\nOptions:\n" in c["text"]]
    assert len(text) == 1
    block = text[0].split("\nOptions:\n", 1)[1].split("\nFormat Instructions", 1)[0]
    return block.split("\n")


def rebuild_grpo(task, grpo_name, raw_name, raw_results, args):
    src = TR + grpo_name
    rows = load_jsonl(src)
    new_rows, source_lines = [], []
    if raw_name is None:
        rng = file_rng(args.seed, src)
        builder = {"sync": sync_options}[task]
        built = []
        for g in rows:
            old_options = prompt_options(g)
            assert g["answer"] in old_options
            options, answer = builder({"options": old_options, "answer": g["answer"]}, rng)
            built.append({"options": options, "answer": answer})
        for i, (g, b) in enumerate(zip(rows, balance_answer_letters(built, rng))):
            new_rows.append(rewrite_prompt(g, prompt_options(g), b["options"], b["answer"]))
            source_lines.append(i)
        origin = "options rebuilt from the prompt text"
    else:
        raw_src = TR + raw_name
        old_raw = load_jsonl(raw_src)
        assert len(old_raw) == len(rows), (src, raw_src)
        new_raw, raw_lines = raw_results[raw_src]
        by_line = dict(zip(raw_lines, new_raw))
        for i, (g, r) in enumerate(zip(rows, old_raw)):
            assert g["answer"] == r["answer"], (src, i)
            if i not in by_line:
                continue  # item dropped from the raw file
            new_rows.append(rewrite_prompt(g, r["options"], by_line[i]["options"], by_line[i]["answer"]))
            source_lines.append(i)
        origin = f"options from {suffixed(raw_src)} (line-aligned with {raw_src})"
    for i, g in zip(source_lines, new_rows):  # the rewrite must be exactly reversible
        assert rewrite_prompt(g, prompt_options(g), prompt_options(rows[i]), rows[i]["answer"]) == rows[i]
    print(f"[{task}] {src} -> {suffixed(src)}: {len(new_rows)}/{len(rows)} prompts ({origin})")
    meta = dict(task=task, script="generation/build/reoption.py", seed=args.seed, input=src,
                input_sha256=sha256(src), options_source=origin, source_lines=source_lines,
                note="frames are the input file's own; only the options text and answer changed")
    return new_rows, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true", help="Build and verify everything in memory; write nothing.")
    parser.add_argument("--no-grpo", action="store_true", help="Skip the GRPO prompt files.")
    parser.add_argument("--reid-video-seconds", type=float, default=20.0,
                        help="Length of every ReID Video 2 (all are 100 frames at 5 fps).")
    parser.add_argument("--kubric-root", default=f"{DATA_ROOT}/kubric_data")
    args = parser.parse_args()

    spec = TASKS[args.task]
    raw_jobs = [(spec["eval"], True)] + [(TR + f, False) for f in spec["train"] if os.path.exists(TR + f)]
    for f in spec["train"]:
        if not os.path.exists(TR + f):
            print(f"note: {TR + f} not found, skipped")
    grpo_jobs = [] if args.no_grpo else [(g, r) for g, r in spec["grpo"] if os.path.exists(TR + g)]
    outputs = [suffixed(p) for p, _ in raw_jobs] + [suffixed(TR + g) for g, _ in grpo_jobs]
    if not args.dry_run:
        refuse_existing(outputs + [o + ".meta.json" for o in outputs])

    raw_results, pending = {}, []
    for src, is_eval in raw_jobs:
        new, lines, meta = rebuild_raw(args.task, src, is_eval, args)
        raw_results[src] = (new, lines)
        pending.append((suffixed(src), new, meta))
    for grpo_name, raw_name in grpo_jobs:
        if raw_name is not None and TR + raw_name not in raw_results:
            print(f"note: {grpo_name} skipped, its raw source {raw_name} was not rebuilt")
            continue
        new, meta = rebuild_grpo(args.task, grpo_name, raw_name, raw_results, args)
        pending.append((suffixed(TR + grpo_name), new, meta))

    if args.dry_run:
        print("dry run: nothing written")
        return
    for path, rows, meta in pending:
        write_outputs(path, rows, meta)
    print(f"wrote {len(pending)} files (+ .meta.json sidecars), suffix {SUFFIX}")


if __name__ == "__main__":
    main()
