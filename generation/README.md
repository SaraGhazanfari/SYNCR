# SYNCR data generation

This directory holds the code that produces SYNCR's questions from simulator state, and the build scripts that
turn the generated question pools into the released benchmark (500 questions per task, 4,000 total) and
training set (15,960 questions).

**You do not need any of this to evaluate a model.** The released annotations on
[Hugging Face](https://huggingface.co/datasets/CrossVideoReasoning/SYNCR) are the output of this pipeline; see
the main [README](../README.md). Use this code to understand how an item was built, to generate more data, or
to make new task variants.

```
generation/
├── clevrer/                     Sequential Ordering, Kinematic and Numerical Comparison (CLEVRER)
│   ├── clvr_temporal_alignment.py
│   ├── clvr_comparative_reasoning.py
│   ├── base_clevrer.py, utils.py
├── kubric/base_kubric.py        Multi-Angle Synchronization, Spatial Measurement (Kubric)
├── habitat/                     Object Re-identification, Object Counting, Route Planning (Habitat / HM3D)
│   ├── habitat_simulator.py       renders the walkthrough videos and per-frame instance masks
│   ├── habitat_entity_tracking.py Object Re-identification
│   ├── habitat_holistic_aggregation.py  Object Counting and Route Planning
│   ├── habitat_traces.py, base_habitat.py
├── option_balance.py            distractor construction shared by all tasks
├── trace_style.py               reasoning-trace templates (training files carry step-by-step traces)
└── build/                       assembly of the released evaluation and training files
```

All commands run from the repository root as modules (`python -m generation...`).

## 1. Raw data

Set `SYNCR_DATA` to the directory holding the raw simulator data (default: `./data`):

```
$SYNCR_DATA/
├── clvr/
│   ├── video_train/video_00000.mp4 ...           CLEVRER ids 0-9999
│   ├── video_validation/video_10000.mp4 ...      CLEVRER ids 10000-14999
│   ├── annotation_train/annotation_00000.json ...
│   └── annotation_validation/annotation_10000.json ...
├── kubric_data/
│   └── scene_<N>/cam{0,1,2}.mp4, scene_<N>/cam{0,1,2}/metadata.json        scenes 0-1999
└── habitat_data/
    ├── general/{train,val}/<scene>/cam<i>/{cam<i>.mp4, semantic_video_ids.npz}, scene_semantic_objects.json
    └── route_plan/{train,val}/<scene>/route<k>/cam<i>/..., video_routes.json
```

- **CLEVRER**: download videos and annotations from <http://clevrer.csail.mit.edu>. The official archives nest
  videos in subfolders of 1,000 (`video_10000-11000/`); flatten them so every file sits directly in
  `video_train/` or `video_validation/`:
  `find video_validation -mindepth 2 -name '*.mp4' -exec mv {} video_validation/ \;`
- **Habitat**: the released `habitat_data.tar` contains the rendered videos and masks. To re-render (or render
  more scenes) from the semantically annotated HM3D scenes, install `habitat-sim` and run
  ```bash
  python -m generation.habitat.habitat_simulator --task both --split val \
      --hm3d-root /path/to/hm3d --output-root $SYNCR_DATA/habitat_data
  ```
  Each trajectory is 100 frames at 5 FPS and ends with a panoramic sweep.
- **Kubric**: the released `kubric_data.tar` contains the rendered scenes (10-15 objects, 5 s at 12 FPS, three
  synchronized cameras). They were rendered with a modified version of Kubric's MOVi worker, which is not
  included here.

## 2. Question generators

Each generator reads the raw data and writes a question pool (`json_files/eval/...` for validation data,
`json_files/train/...` with `--split train`). These pools are the "original benchmark files" the build step
selects from.

```bash
# Temporal Alignment
python -m generation.kubric.base_kubric --task sync --total-num 1000
python -m generation.clevrer.clvr_temporal_alignment --total-num 1000            # Sequential Ordering
# Spatial Tracking
python -m generation.habitat.habitat_entity_tracking --split val                 # Object Re-identification
python -m generation.kubric.base_kubric --task spatial --total-num 1000
# Comparative Reasoning
python -m generation.clevrer.clvr_comparative_reasoning --task-type collision --total-num 1000   # Numerical
python -m generation.clevrer.clvr_comparative_reasoning --task-type kinematic --total-num 1000   # Kinematic
# Holistic Synthesis
python -m generation.habitat.habitat_holistic_aggregation --task object_counting --split val
python -m generation.habitat.habitat_holistic_aggregation --task route_plan --split val
```

Answers come only from simulator state: crop start times (Sync), shared instance IDs and per-frame masks
(ReID, Counting), 3D positions at the event frame (Spatial Measurement), annotated velocities and collisions
(Kinematic, Numerical), and navigation graphs (Route Planning). Models only ever see the RGB clips.

**Distractors** (`option_balance.py`) are drawn so that the answer cannot be picked from the options alone:
Sync distractors are independent draws of the crop-offset process, ReID distractors are visibility windows
with similar timing, Route distractors are paths with a matched hop-count mix, Ordering distractors follow the
answer permutation frequencies, and Numerical distractors use wrong video identities and count differences.

## 3. Building the released files

`build/` turns the pools into the released files. Every output gets a `.meta.json` sidecar (seed, input
hashes, git commit), and no script overwrites an existing file. The default seed is `20260913` throughout.

| Script | Role |
|---|---|
| `build_eval.py` | selects 500 items per task; lines 1-200 and 1-500 are each answer-bucket, scene and letter balanced (`nested.py`); rebuilds options |
| `build_train.py` | 2,000 balanced training items per task from videos disjoint from evaluation |
| `habitat_pool_shards.py`, `counting_routepool.py`, `numerical_eval_pool.py` | full-scene ReID, Route, Counting and Numerical pools (sharded for a cluster) |
| `counting_total.py` | Object Counting as "total instances of k named categories" |
| `balance_ordering_perms.py` | equalizes the 24 answer orders in a Sequential Ordering file |
| `tasks.py`, `reoption.py`, `common.py`, `nested.py` | shared builders and balancing |
| `check_leakage.py` | verifies no video is shared between training and evaluation files |

The released set was built in stages; each stage's file suffix records it:

```bash
# A. _optv3c: evaluation subsets and Kubric scene split
python -m generation.build.build_eval
python -m generation.build.build_train

# B. full-scene pools (each command is one shard; run all shards, then --merge)
python -m generation.build.habitat_pool_shards --task reid  --split train --shard $i --num-shards 33
python -m generation.build.habitat_pool_shards --task route --split train --shard $i --num-shards 40
python -m generation.build.counting_routepool --split train --section route_plan --shard $i --num-shards 40
python -m generation.build.habitat_pool_shards --task reid  --split val ...     # likewise for route, counting
python -m generation.build.numerical_eval_pool --out json_files/eval/_pools/Clvr_Numerical_Comparison_fullval3.jsonl

# C. _optv3d: training files from the full-scene pools, balanced Ordering permutations
python -m generation.build.build_train --suffix _optv3d \
    --tasks numerical kinematic reid counting route sync spatial \
    --pool-files reid=scenepool_v3c route=scenepool_v3c counting=trainpool4_v3c,routepool_v3c \
    --counting-general-first
python -m generation.build.balance_ordering_perms IN.jsonl OUT.jsonl

# D. _optv3e: final ReID / Route / Numerical evaluation, Counting, gapped Ordering training
P=json_files/eval/_pools
python -m generation.build.build_eval --suffix _optv3e --tasks reid route numerical \
    --source reid=$P/Habitat_Object_Re_identification_fullval.jsonl \
             route=$P/Habitat_Route_Planning_fullval.jsonl \
             numerical=$P/Clvr_Numerical_Comparison_fullval3.jsonl
python -m generation.build.counting_total --split val   --draws 100 --total-weight-power 2
python -m generation.build.counting_total --split train --draws 100 --total-weight-power 2
python -m generation.build.build_train --suffix _optv3e_unbalanced --tasks ordering_gap \
    --ordering-gap-exclude json_files/train/Clvr_Sequential_Ordering_optv3d.jsonl
python -m generation.build.balance_ordering_perms \
    json_files/train/Clvr_Sequential_Ordering_gap_optv3e_unbalanced.jsonl \
    json_files/train/Clvr_Sequential_Ordering_gap_optv3e.jsonl

python -m generation.build.check_leakage
```

Released files (the suffixes match the configs in `../configs/`):

| Task | Evaluation (500) | Training |
|---|---|---|
| Multi-Angle Synchronization | `Kubric_Multi_Angle_Synchronization_optv3c` | `Kubric_Multi_Angle_Sync_optv3d` (2,000) |
| Sequential Ordering (gapped) | `Clvr_Sequential_Ordering_gap_optv3d` | `Clvr_Sequential_Ordering_gap_optv3e` (2,000) |
| Object Re-identification | `Habitat_Object_Re_identification_optv3e` | `..._optv3d` (2,000) |
| Spatial Measurement | `Kubric_Cross_Video_Spatial_Measurement_optv3c` | `Kubric_Spatial_Measurement_optv3d` (2,000) |
| Numerical Comparison | `Clvr_Numerical_Comparison_optv3e` | `..._optv3d` (2,000) |
| Kinematic Comparison | `Clvr_Kinematic_Comparison_optv3c` | `..._optv3d` (2,000) |
| Object Counting | `Habitat_Object_Counting_optv3e` | `..._optv3e` (1,960) |
| Route Planning | `Habitat_Route_Planning_optv3e` | `..._optv3d` (2,000) |

Object Counting has 1,960 training items rather than 2,000 because the balanced selection over
(k, count) buckets runs out of items with large counts.

### Reproducibility notes

The build is deterministic given its inputs, but regenerating the released files byte for byte is not
guaranteed:

- `base_clevrer.py` lists annotation files in filesystem order, so CLEVRER item selection can differ between
  machines.
- Shard seeds depend on the shard count (`seed * 1000 + shard`). The validation pools were built with 10-12
  shards; the exact counts were not recorded.
- `Habitat_Object_Counting_trainpool4_v3c.jsonl`, an input to stage C, was assembled from earlier Counting
  pools by hand rather than by a script here.
- The gapped Ordering evaluation reuses the 500 CLEVRER validation videos of an earlier Ordering file
  (`Clvr_Sequential_Ordering_optv3b.jsonl`, read by `build_eval.py`); point `ORDERING_GAP_VIDEOS_FROM` at any
  500-item Ordering file to build a new one.

For new data, run the generators and stages A-D on your own pools; the balancing, disjointness and audit
logic does not depend on these specifics.
