"""Per-task pieces shared by generation/build/build_eval.py and build_train.py."""
import collections
import json
import re

from generation.option_balance import (NUMERICAL_TIE_OPTION, numerical_answer_text, parse_route_option,
                                 reoption_ordering_v3, reoption_route_v3, reoption_spatial_v3, route_item_status)
from generation.build.common import body, video_path


# ---------------------------------------------------------------- answer buckets

def counting_bucket(row):
    value = int(body(row["answer"]))
    return "3+" if value >= 3 else str(value)


def route_bucket(row):
    return "2" if body(row["answer"]).count("->") <= 2 else "3+"


def numerical_bucket(row):
    ranked = sorted(row["collision_count"], reverse=True)
    diff = ranked[0] - ranked[1]
    return "tie" if diff == 0 else ("3+" if diff >= 3 else str(diff))


def numerical_winner(row):
    text = numerical_answer_text(row["collision_count"])
    return "tie" if text == NUMERICAL_TIE_OPTION else text.split(" -- ")[0]


# task -> (bucket_of(row), sub_bucket_of(row) or None); tasks not listed have a single bucket
BUCKETS = {
    "counting": (counting_bucket, None),
    "route": (route_bucket, None),
    "numerical": (numerical_bucket, numerical_winner),
}
# Relative bucket sizes; unlisted buckets weigh 1. Numerical: ties are 1/5 of items
#, so "always tie" scores chance (0.20) with the tie option always shown.
BUCKET_WEIGHTS = {
    "numerical": {"tie": 1.0, "1": 4 / 3, "2": 4 / 3, "3+": 4 / 3},
}


# ---------------------------------------------------------------- duplicate items

def item_key(row):
    """Same question and same videos (including crop bounds) = the same item."""
    return row["question"], json.dumps(row["video_list"], sort_keys=True)


def dedupe_rows(rows):
    """Indices to keep: items repeated with different answers are all dropped, identical repeats keep the first.

    The option audit found the same ReID question and videos with different "first
    appearance" windows (e.g. "10.8s to 16.8s" vs "11.2s to 16.8s").
    Returns (sorted indices, counts).
    """
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[item_key(r)].append(i)
    keep, report = [], collections.Counter()
    for idx in groups.values():
        answers = {body(rows[i]["answer"]) for i in idx}
        if len(idx) == 1:
            keep.append(idx[0])
        elif len(answers) > 1:
            report["conflicting_groups"] += 1
            report["conflicting_items_removed"] += len(idx)
        else:
            keep.append(idx[0])
            report["identical_groups"] += 1
            report["identical_extra_removed"] += len(idx) - 1
    return sorted(keep), dict(report)


# ---------------------------------------------------------------- Ordering

def build_ordering_v3(rows, rng, args=None):
    return [tuple(x) for x in reoption_ordering_v3([body(r["answer"]) for r in rows], rng)], {}


# ---------------------------------------------------------------- Route

def route_pool(row):
    """Middle-room names shown in one item's options (the stored names cannot be regenerated)."""
    names = []
    for o in row["options"]:
        parsed = parse_route_option(body(o))
        if parsed:
            names.extend(parsed[1])
    return names


def grouped_route_pools(rows, min_rooms=8):
    """Per row: middle-room names shown by every item of the same route.

    Items with the same scene_idx and route_idx share the same three videos,
    so their rooms are rooms those videos show. One item's own options name too
    few rooms to give distractors the answers' hop mix. The item's own start
    and goal rooms are removed later (route_item_status); when fewer than
    min_rooms other rooms remain, rooms from the scene's other routes (the same
    house) are added, so short distractors do not run out.
    """
    by_route, by_scene = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        names = route_pool(r)
        by_route[(r.get("scene_idx"), r.get("route_idx"))].extend(names)
        by_scene[r.get("scene_idx")].extend(names)
    pools = []
    for r in rows:
        pool = list(dict.fromkeys(route_pool(r) + by_route[(r.get("scene_idx"), r.get("route_idx"))]))
        parsed = parse_route_option(body(r["answer"]))
        ends = (parsed[0], parsed[2]) if parsed else ()
        if sum(x not in ends for x in pool) < min_rooms:
            pool = list(dict.fromkeys(pool + by_scene[r.get("scene_idx")]))
        pools.append(pool)
    return pools


def route_usable(row, pool=None):
    return route_item_status(body(row["answer"]), route_pool(row) if pool is None else pool)[1] is None


def build_route_v3(rows, rng, args=None, pools=None):
    pools = [route_pool(r) for r in rows] if pools is None else pools
    return reoption_route_v3([body(r["answer"]) for r in rows], pools, rng)


# ---------------------------------------------------------------- Spatial Measurement

def spatial_candidates(kubric, row):
    """(visible objects ranked by distance to the target, all scene object names) for an existing item."""
    scene = row["scene_idx"]
    if scene not in kubric.instance_data and len(kubric.instance_data) >= 8:
        kubric.instance_data.clear()  # metadata is large; keep a few scenes
    kubric._get_raw_item(scene)
    names = [kubric.get_object_name(scene, i) for i in range(len(kubric.instance_data[scene]["cam0"]))]
    cams = [int(re.search(r"cam(\d+)\.mp4$", video_path(v)).group(1)) for v in row["video_list"]]
    visibility = kubric.get_visibility_for_all_objects(scene, len(cams), cams=cams)
    ranked = kubric.rank_objects_by_distance(scene, names.index(row["target_object"]), names.index(row["event_object"]),
                                             int(row["exit_frame"]), visibility, len(cams))
    return [names[o] for o, _ in ranked], names


def build_spatial_v3(rows, rng, args):
    from generation.kubric.base_kubric import Kubric_Dataset
    kubric = Kubric_Dataset(path=args.kubric_root)
    items, recomputed = [], 0
    for r in rows:
        ranked, names = spatial_candidates(kubric, r)
        answer = body(r["answer"])
        recomputed += bool(ranked) and ranked[0] == answer
        items.append({"answer": answer, "target": r["target_object"], "event": r["event_object"],
                      "ranked": [x for x in ranked if x != answer], "scene_names": names})
    out, stats = reoption_spatial_v3(items, rng)
    stats["answer_is_recomputed_closest"] = f"{recomputed}/{len(rows)}"
    return out, stats


# ---------------------------------------------------------------- gapped Ordering

GAP_RANGE = (0.3, 0.6)
MAX_START_OFFSET = 0.5


def clevrer_index(dataset):
    """{annotation id string (e.g. '10000'): data_list index}."""
    return {next(iter(d)): i for i, d in enumerate(dataset.data_list)}


def clevrer_id(row):
    return re.search(r"video_(\d+)\.mp4$", video_path(row["video_list"][0])).group(1)


def gapped_ordering_items(dataset, source_rows, rng):
    """Gapped, equal-length 4-segment items for the same CLEVRER videos as source_rows, in their order.

    Options from reoption_ordering_v3 over the new items. The module RNG must be
    seeded by the caller (get_seq_order_QA draws from it).
    """
    index = clevrer_index(dataset)
    rows = [dataset.get_seq_order_QA(index[clevrer_id(r)], total_frames=128, k=4,
                                     gap_range=GAP_RANGE, max_start_offset=MAX_START_OFFSET)
            for r in source_rows]
    built = reoption_ordering_v3([body(r["answer"]) for r in rows], rng)
    return [dict(r, options=o, answer=a) for r, (o, a) in zip(rows, built)]
