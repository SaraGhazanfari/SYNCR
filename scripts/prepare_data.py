#!/usr/bin/env python3
"""Convert the released SYNCR dataset into the per-task files the evaluator reads.

The Hugging Face dataset ships one unified annotation file, `test/metadata.jsonl`.
`eval.evaluate` reads one file per task, keyed by `video_list`. This script bridges
the two so the released code runs against the released data unchanged.

Usage:
    python scripts/prepare_data.py --dataset-root /path/to/SYNCR

Writes json_files/eval/<Category>/<Task>.jsonl, matching the paths in configs/.
"""
import argparse, json, os, collections

# (category, task) in metadata.jsonl  ->  path referenced by configs/*.conf
TASK_FILES = {
    ("Temporal_Alignment",    "Multi_Angle_Synchronization"):
        "Temporal_Alignment/Kubric_Multi_Angle_Synchronization_optv3c",
    ("Temporal_Alignment",    "Sequential_Ordering"):
        "Temporal_Alignment/Clvr_Sequential_Ordering_gap_optv3d",
    ("Spatial_Tracking",     "Object_Re_identification"):
        "Spatial_Tracking/Habitat_Object_Re_identification_optv3e",
    ("Spatial_Tracking",     "Spatial_Measurement"):
        "Spatial_Tracking/Kubric_Cross_Video_Spatial_Measurement_optv3c",
    ("Comparative_Reasoning", "Numerical_Comparison"):
        "Comparative_Reasoning/Clvr_Numerical_Comparison_optv3e",
    ("Comparative_Reasoning", "Kinematic_Comparison"):
        "Comparative_Reasoning/Clvr_Kinematic_Comparison_optv3c",
    ("Holistic_Synthesis",  "Object_Counting"):
        "Holistic_Synthesis/Habitat_Object_Counting_optv3e",
    ("Holistic_Synthesis",  "Route_Planning"):
        "Holistic_Synthesis/Habitat_Route_Planning_optv3e",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True,
                    help="Directory holding test/metadata.jsonl and videos/")
    ap.add_argument("--out", default="json_files/eval",
                    help="Where to write the per-task files (default: json_files/eval)")
    ap.add_argument("--absolute-video-paths", action="store_true",
                    help="Rewrite video paths to absolute paths under --dataset-root. "
                         "Use this when running the evaluator from a different directory.")
    args = ap.parse_args()

    meta = os.path.join(args.dataset_root, "test", "metadata.jsonl")
    if not os.path.exists(meta):
        raise SystemExit(f"not found: {meta}")

    buckets = collections.defaultdict(list)
    missing_videos, total = set(), 0

    with open(meta) as fh:
        for line in fh:
            row = json.loads(line)
            key = (row["category"], row["task"])
            if key not in TASK_FILES:
                raise SystemExit(f"unmapped task in metadata: {key}")

            ranges = row.get("video_ranges") or []
            videos = []
            for i, v in enumerate(row["videos"]):
                path = os.path.join(args.dataset_root, v) if args.absolute_video_paths else v
                entry = {"video_path": path}
                if i < len(ranges) and ranges[i]:
                    entry["start_sec"], entry["end_sec"] = ranges[i][0], ranges[i][1]
                videos.append(entry)
                if not os.path.exists(os.path.join(args.dataset_root, v)):
                    missing_videos.add(v.split("/")[1] if "/" in v else v)

            out = {"video_list": videos, "question": row["question"],
                   "options": row["options"], "answer": row["answer"]}
            if row.get("reasoning"):
                out["reasoning"] = row["reasoning"]
            buckets[key].append(out)
            total += 1

    for key, rows in sorted(buckets.items()):
        rel = TASK_FILES[key] + ".jsonl"
        dest = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"  {len(rows):5d}  {dest}")

    print(f"\nwrote {total} questions across {len(buckets)} tasks")
    if missing_videos:
        print("\nWARNING: video files were not found for these sources:")
        for s in sorted(missing_videos):
            print(f"  - {s}")
        print("Extract the dataset archives into <dataset-root>/videos/ before evaluating.")


if __name__ == "__main__":
    main()
