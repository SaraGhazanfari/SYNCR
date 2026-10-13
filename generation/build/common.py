"""
Shared helpers for the option rebuild.

A rebuild keeps every field of an existing item except `options` and `answer`
(whose text is kept; only its letter may change), so train/eval video
disjointness, reasoning traces and process_gt carry over untouched. Outputs
go next to their inputs with SUFFIX, never over an existing file, with a
`<output>.meta.json` sidecar recording the seed, file hashes and which input
line every output line came from.

SUFFIX is `_optv2`, not `_v2`: `_v2` already names unrelated files
(Habitat_Object_Counting_v2.jsonl, grpo_habitat_reid_v2.jsonl,
grpo_kubric_multi_angle_sync_v2.jsonl, train_syncr_v2*.json), and
Habitat_Object_Counting.jsonl + `_v2` would collide with one of them.
"""
import collections
import datetime
import hashlib
import json
import os
import random
import re
import subprocess
import sys

SUFFIX = "_optv2"
LETTER_RE = re.compile(r"^[A-Z]\)\s*")


def body(option):
    return LETTER_RE.sub("", option, count=1)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def suffixed(path, suffix=SUFFIX):
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def refuse_existing(paths):
    existing = [p for p in paths if os.path.exists(p)]
    if existing:
        sys.exit("FATAL: refusing to overwrite existing file(s):\n  " + "\n  ".join(existing))


def file_rng(seed, path):
    """Independent, reproducible stream per input file (str seeds are hashed deterministically)."""
    return random.Random(f"{seed}:{os.path.basename(path)}")


def video_path(entry):
    return entry if isinstance(entry, str) else entry["video_path"]


def scene_of(row):
    """Scene id used to stratify the eval shuffle."""
    if "scene_idx" in row:
        return row["scene_idx"]
    parts = video_path(row["video_list"][0]).split("/")
    if "habitat_data" in parts:  # habitat_data/<section>/<split>/<scene>/...
        return parts[parts.index("habitat_data") + 3]
    if "kubric_data" in parts:
        return parts[parts.index("kubric_data") + 1]
    return parts[-1]  # CLEVRER: one simulation per video file


def stratified_shuffle(n, group_of, rng):
    """Permutation of range(n) that interleaves groups (e.g. scenes).

    Groups are visited round-robin in a random order, items shuffled within
    each group, so any prefix -- in particular the first 200 lines that
    eval/evaluate.py --sample-num scores -- covers as many groups as possible.
    """
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[group_of(i)].append(i)
    order = list(groups)
    rng.shuffle(order)
    for g in order:
        rng.shuffle(groups[g])
    perm, depth = [], 0
    while len(perm) < n:
        for g in order:
            if depth < len(groups[g]):
                perm.append(groups[g][depth])
        depth += 1
    return perm


def check_invariants(old_rows, new_rows):
    """New rows differ from old only in option wording/order and the answer letter."""
    assert len(old_rows) == len(new_rows)
    for i, (o, n) in enumerate(zip(old_rows, new_rows)):
        assert list(o) == list(n), (i, list(o), list(n))
        for k in o:
            if k not in ("options", "answer"):
                assert o[k] == n[k], (i, k)
        assert body(o["answer"]) == body(n["answer"]), i
        assert len(n["options"]) == len(o["options"]), i
        bodies = [body(x) for x in n["options"]]
        assert len(set(bodies)) == len(bodies), (i, bodies)
        assert n["options"].count(n["answer"]) == 1, i
        assert all(x.startswith(f"{chr(65 + j)}) ") for j, x in enumerate(n["options"])), i


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def write_outputs(path, rows, meta):
    with open(path, "x") as f:  # "x": fail rather than overwrite
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = dict(meta, output=path, output_sha256=sha256(path), n=len(rows),
                git_commit=git_commit(), created=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(path + ".meta.json", "x") as f:
        json.dump(meta, f, indent=1)
