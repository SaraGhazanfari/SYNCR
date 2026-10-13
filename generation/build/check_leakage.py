"""
Verify that no video used for SFT training also appears in the SYNCR benchmark.

This underwrites the paper's contamination claim, so it should be re-run whenever
training data is regenerated.

Videos are keyed by their path suffix beneath the dataset root, not by basename.
Basename is unique for CLEVRER (video_00042.mp4) but badly wrong for Habitat and
Kubric, where every scene contains files literally named cam0.mp4 -- keying on
basename there collapses thousands of distinct videos into three and would hide
real leakage. Using the suffix also survives train and eval trees living under
different roots (data/clvr vs dataset/clvr).

Run:
    python -m generation.build.check_leakage
    python -m generation.build.check_leakage --train json_files/train/*.jsonl
"""
import argparse
import glob
import json
import os
import re
import sys

# Path components that mark the start of a dataset tree. Everything after the
# first match identifies the video within that dataset.
ROOT_MARKERS = ('clvr', 'clevrer', 'habitat_data', 'kubric_data')


def video_key(path):
    """Identity of a video: its path relative to the dataset root."""
    parts = path.split('/')
    for i, part in enumerate(parts):
        if part.lower() in ROOT_MARKERS:
            suffix = '/'.join(parts[i + 1:])
            if suffix:
                return suffix
    # No recognized marker: fall back to the last two components, which still
    # separates scene_5/cam0.mp4 from scene_6/cam0.mp4.
    return '/'.join(parts[-2:]) if len(parts) >= 2 else path


def clevrer_video_id(path):
    """CLEVRER video id, or None. Ids are globally unique across splits."""
    m = re.search(r'video_(\d+)\.mp4$', path)
    return int(m.group(1)) if m else None


def video_keys(jsonl_path):
    """Return (identity keys, CLEVRER ids) for every video a .jsonl references."""
    keys, clevrer_ids = set(), set()
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            for entry in sample.get('video_list', []):
                path = entry['video_path'] if isinstance(entry, dict) else entry
                keys.add(video_key(path))
                cid = clevrer_video_id(path)
                if cid is not None:
                    clevrer_ids.add(cid)
    return keys, clevrer_ids


def main():
    parser = argparse.ArgumentParser(
        description='Check that training and benchmark videos are disjoint.')
    parser.add_argument('--train', nargs='+', default=['json_files/train/*.jsonl'],
                        help='Training .jsonl files (globs allowed).')
    parser.add_argument('--eval', nargs='+', default=['json_files/eval/*/*.jsonl'],
                        help='Benchmark .jsonl files (globs allowed).')
    args = parser.parse_args()

    def expand(patterns):
        out = []
        for p in patterns:
            out.extend(sorted(glob.glob(p)) or ([p] if os.path.exists(p) else []))
        return out

    train_files, eval_files = expand(args.train), expand(args.eval)

    if not train_files:
        print('No training files matched.')
        return 1
    if not eval_files:
        print('No benchmark files matched.')
        return 1

    train_videos, train_ids = set(), set()
    print('Training files:')
    for f in train_files:
        v, ids = video_keys(f)
        train_videos |= v
        train_ids |= ids
        print(f'  {len(v):6d} videos  {f}')

    eval_videos, eval_ids = set(), set()
    print('\nBenchmark files:')
    for f in eval_files:
        v, ids = video_keys(f)
        eval_videos |= v
        eval_ids |= ids
        print(f'  {len(v):6d} videos  {f}')

    overlap = train_videos & eval_videos
    id_overlap = train_ids & eval_ids

    print(f'\nUnique training videos : {len(train_videos)}')
    print(f'Unique benchmark videos: {len(eval_videos)}')
    print(f'Path overlap           : {len(overlap)}')

    if train_ids and eval_ids:
        print(f'\nCLEVRER id ranges      : train {min(train_ids)}-{max(train_ids)}, '
              f'eval {min(eval_ids)}-{max(eval_ids)}')
        print(f'CLEVRER id overlap     : {len(id_overlap)}')

    failed = False

    if overlap:
        failed = True
        print('\nLEAKAGE DETECTED (same video path in both sets):')
        for name in sorted(overlap)[:20]:
            print(f'  {name}')
        if len(overlap) > 20:
            print(f'  ... and {len(overlap) - 20} more')

    # A CLEVRER id collision is leakage even across differing roots, since the id
    # identifies the underlying simulation regardless of where it is stored.
    if id_overlap:
        failed = True
        print('\nLEAKAGE DETECTED (same CLEVRER video id in both sets):')
        for vid in sorted(id_overlap)[:20]:
            print(f'  video_{vid:05d}.mp4')
        if len(id_overlap) > 20:
            print(f'  ... and {len(id_overlap) - 20} more')

    if failed:
        return 1

    print('\nPASS: no video appears in both training and benchmark data.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
