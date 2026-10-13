"""Single-video / no-video controls and the Num per-video perception control (pre-registered
before the runs).

For one model's saved predictions, per task: lenient accuracy with all videos, with one seeded random video
(<tag>_single) and with no video (<tag>_novideo), paired on the same first N items; chance (mean 1/#options over those
items); the retained fraction R = (condition - chance) / (full - chance); and the pre-registered flags.
With --pervideo it also scores the Num per-video control: P_both = share of items whose two per-video collision counts
are both correct, next to the multi-video Num accuracy on the same items.

  python -m eval.controls_report --tag Qwen__Qwen3-VL-8B-Instruct --n 200 --suite syncr --pervideo
  python -m eval.controls_report --tag Qwen__Qwen3-VL-8B-Instruct --n 200 --suite real
  python -m eval.controls_report --tag gemini-3.1-pro-preview --n 25 --suite syncr --pervideo
"""
import argparse
import glob
import json
import os
import re

from eval.answer_parse import score_answer_lenient
from utils import read_jsonl_file

SUITES = {
    "syncr": [
        ("Sync", "Kubric_Multi_Angle_Synchronization_optv3c"),
        ("Order", "Clvr_Sequential_Ordering_gap_optv3d"),
        ("ReID", "Habitat_Object_Re_identification_optv3e"),
        ("Meas", "Kubric_Cross_Video_Spatial_Measurement_optv3c"),
        ("Num", "Clvr_Numerical_Comparison_optv3e"),
        ("Kin", "Clvr_Kinematic_Comparison_optv3c"),
        ("Count", "Habitat_Object_Counting_optv3e"),
        ("Route", "Habitat_Route_Planning_optv3e"),
    ],
    "real": [
        ("CrossVid BU", "BU_data"),
        ("CrossVid CC", "CC_data"),
        ("CrossVid NC", "NC_data"),
        ("CrossVid PI (clips)", "PI_clip_data"),
        ("CrossVid PEA (clips)", "PEA_clip_data"),
        ("MVU SU", "SU_data"),
        ("MVU Counting", "Counting_data"),
        ("MVU Comparison", "Comparison_data"),
        ("MVU OR", "OR_data"),
        ("MVU ICL", "ICL_data"),
        ("MVU KIR", "KIR_data"),
        ("MVU TR", "TR_data"),
        ("MVU RAG", "RAG_data"),
    ],
}
PERVIDEO_STEM = "Clvr_Numerical_PerVideo_optv3e"
NUM_STEM = "Clvr_Numerical_Comparison_optv3e"


def find_predictions(root, tag, stem, n):
    """Predictions file covering the first n items: the exact __n<n> run (its _retried version first, written by
    eval/retry_failed_items.py and used for the main table), else the smallest larger run."""
    for name in (f"{stem}__n{n}_retried.jsonl", f"{stem}__n{n}.jsonl"):
        path = os.path.join(root, tag, name)
        if os.path.exists(path):
            return path
    best = None
    for path in glob.glob(os.path.join(root, tag, f"{stem}__n*.jsonl")):
        match = re.search(r"__n(\d+|all)\.jsonl$", path)
        if match is None:
            continue
        size = float("inf") if match.group(1) == "all" else int(match.group(1))
        if size >= n and (best is None or size < best[0]):
            best = (size, path)
    return best[1] if best else None


def lenient_correct(path, n):
    """({index: correct} for indices below n, the eval file's items); lenient parser, options from the eval file.
    Indices missing from the file (never scored) count as wrong, as in eval/evaluate.py."""
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    samples = read_jsonl_file(rows[0]["filepath"])
    correct = {}
    for r in rows:
        if r["index"] < n:
            correct[r["index"]] = score_answer_lenient(r.get("prediction"), r["answer"], samples[r["index"]]["options"])[0]
    return correct, samples


def pct(values):
    return 100 * sum(values) / len(values)


def report_suite(root, full_root, tag, n, suite):
    print(f"\n### {tag}, {suite}, first {n} items (lenient)\n")
    print("| Task | n | chance | all videos | single video | R single | no video | R no video | flags |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, stem in SUITES[suite]:
        full_path = find_predictions(full_root, tag, stem, n)
        if full_path is None:
            print(f"| {name} | | | pending | | | | | |")
            continue
        full, samples = lenient_correct(full_path, n)
        items = range(min(n, len(samples)))
        chance = 100 * sum(1 / len(samples[i]["options"]) for i in items) / len(items)
        acc_full = pct([full.get(i, False) for i in items])
        above = acc_full - chance
        cells, flags = [], []
        for cond in ("single", "novideo"):
            path = find_predictions(root, f"{tag}_{cond}", stem, n)
            if path is None:
                cells += ["pending", ""]
                continue
            got, _ = lenient_correct(path, n)
            acc = pct([got.get(i, False) for i in items])
            retained = (acc - chance) / above if above > 0 else float("nan")
            cells += [f"{acc:.1f}", f"{retained:.2f}"]
            if suite == "syncr" and cond == "single" and above >= 10:
                flags.append("H_A1 holds (R ≤ 0.3)" if retained <= 0.3 else "H_A1 fails (R > 0.3)")
            if suite == "real" and cond == "single" and retained >= 0.7:
                flags.append("does not need all videos (R ≥ 0.7)")
            if suite == "real" and cond == "novideo" and acc >= chance + 10:
                flags.append("text prior (≥ chance + 10)")
        if suite == "syncr" and above < 10:
            flags.append("full within 10 of chance: not used for H_A1")
        print(f"| {name} | {len(items)} | {chance:.1f} | {acc_full:.1f} | {' | '.join(cells)} | {'; '.join(flags)} |")


def report_pervideo(root, full_root, tag, n):
    pv_path = find_predictions(root, tag, PERVIDEO_STEM, 2 * n)
    num_path = find_predictions(full_root, tag, NUM_STEM, n)
    if pv_path is None or num_path is None:
        print(f"\nNum per-video control ({tag}): pending")
        return
    pv, pv_samples = lenient_correct(pv_path, 2 * n)
    num, _ = lenient_correct(num_path, n)
    by_item = {}
    for i, sample in enumerate(pv_samples[:2 * n]):
        by_item.setdefault(sample["source_index"], []).append(pv.get(i, False))
    items = [k for k in range(n) if len(by_item.get(k, [])) == 2]
    per_video = pct([ok for k in items for ok in by_item[k]])
    both = pct([all(by_item[k]) for k in items])
    multi = pct([num.get(k, False) for k in items])
    gap = both - multi
    verdict = "integration" if gap >= 15 else ("perception" if abs(gap) <= 5 else "mixed")
    print(f"\nNum per-video control ({tag}, {len(items)} items, lenient): per-video count accuracy {per_video:.1f} "
          f"(chance 20.0); both counts correct {both:.1f}; multi-video Num accuracy {multi:.1f}; "
          f"gap {gap:+.1f} -> {verdict} (pre-registered: ≥ +15 integration, within 5 perception, else mixed)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", required=True, help="predictions directory name of the model, e.g. Qwen__Qwen3-VL-8B-Instruct")
    parser.add_argument("--n", type=int, required=True, help="first N items (200 open models, 25 API models)")
    parser.add_argument("--suite", choices=sorted(SUITES), default=None)
    parser.add_argument("--pervideo", action="store_true", help="also report the Num per-video control")
    parser.add_argument("--root", default="eval_predictions/hf", help="where the control runs' predictions are")
    parser.add_argument("--full-root", default=None,
                        help="where the all-video reference predictions are, if elsewhere (API main-table runs: "
                             "eval_predictions_gpt_fps/hf, eval_predictions_gemini_out32k/hf)")
    args = parser.parse_args()
    full_root = args.full_root or args.root
    if args.suite:
        report_suite(args.root, full_root, args.tag, args.n, args.suite)
    if args.pervideo:
        report_pervideo(args.root, full_root, args.tag, args.n)


if __name__ == "__main__":
    main()
