# Data Generation Guide

This guide explains how to use the unified `generate_data.py` script to generate data for all SYNCR datasets.

## Overview

The `generate_data.py` script consolidates all data generation arguments into a single entry point, supporting all 8 SYNCR benchmark tasks organized across 4 cognitive pillars:

- **Temporal Alignment**: Sequential ordering and multi-angle synchronization
- **Spatial Tracking**: Spatial measurement and object re-identification
- **Comparative Reasoning**: Kinematic and collision comparison
- **Holistic Synthesis**: Object counting and route planning

## Basic Usage

```bash
python generate_data.py --dataset <DATASET_NAME> [OPTIONS]
```

## Datasets

### 1. CLEVRER Temporal Alignment (`clevrer_temporal`)

Generates sequential ordering QA pairs - given shuffled video segments from a continuous event, determine the correct chronological order.

```bash
python generate_data.py --dataset clevrer_temporal \
    --root-path /path/to/clevrer/data \
    --split validation \
    --total-num 1000 \
    --k 4 \
    --total-frames 128 \
    --gap-time 0.0 \
    --save
```

**Arguments:**
- `--root-path` (required): Root path for the CLEVRER dataset
- `--split`: Dataset split to use (default: `validation`)
- `--total-num`: Total number of samples to retrieve (default: 1000)
- `--k`: Number of video segments to sequence (default: 4)
- `--total-frames`: Total frames for sequence generation (default: 128)
- `--gap-time`: Time gap in seconds between chronological segments (default: 0.0)
- `--save`: Include to save output to JSONL file

---

### 2. Kubric Multi-Angle Synchronization (`kubric_sync`)

Generates temporal synchronization QA pairs from multi-angle video.

```bash
python generate_data.py --dataset kubric_sync \
    --path /path/to/kubric/data \
    --total-num 1000 \
    --save \
    --output-path json_files/Temporal_Alignment/Kubric_Multi_Angle_Synchronization.jsonl
```

**Arguments:**
- `--path` (required): Path to the base directory containing Kubric scene folders
- `--total-num`: Total number of QA samples to generate (default: 1000)
- `--save`: Include to save output to JSONL file
- `--output-path`: Custom output file path

---

### 3. Kubric Spatial Measurement (`kubric_spatial`)

Generates spatial measurement QA pairs.

```bash
python generate_data.py --dataset kubric_spatial \
    --path /path/to/kubric/data \
    --total-num 1000 \
    --save \
    --output-path json_files/Spatial_Tracking/Kubric_Spatial_Measurement.jsonl
```

**Arguments:**
- `--path` (required): Path to the base directory containing Kubric scene folders
- `--total-num`: Total number of QA samples to generate (default: 1000)
- `--save`: Include to save output to JSONL file
- `--output-path`: Custom output file path

---

### 4. Habitat Object Re-identification (`habitat_tracking`)

Generates object tracking/re-identification QA pairs.

```bash
python generate_data.py --dataset habitat_tracking \
    --root-path /path/to/habitat/data \
    --fps 5 \
    --min-range-len 3 \
    --min-total-frames 5 \
    --save \
    --verbose
```

**Arguments:**
- `--root-path` (required): Root path for the Habitat dataset
- `--fps`: Frames per second for time conversions (default: 5)
- `--min-range-len`: Minimum frames a continuous range must have (default: 3)
- `--min-total-frames`: Minimum total frames object must be visible (default: 5)
- `--save`: Include to save output to JSONL file
- `--verbose`: Enable verbose output

---

### 5,6. CLEVRER Comparative Reasoning (`clevrer_comparative`)

Generates comparative reasoning QA pairs for collision or kinematic tasks.

```bash
# Kinematic comparison
python generate_data.py --dataset clevrer_comparative \
    --root-path /path/to/clevrer/data \
    --task-type kinematic \
    --split validation \
    --total-num 1000 \
    --num-samples 100 \
    --threshold 0.25

# Collision comparison
python generate_data.py --dataset clevrer_comparative \
    --root-path /path/to/clevrer/data \
    --task-type collision \
    --split validation \
    --total-num 1000 \
    --num-samples 100 \
    --nframes 32 \
    --time-mode timestamp
```

**Arguments:**
- `--root-path` (required): Root path for the CLEVRER dataset
- `--task-type` (required): Task type - `collision` or `kinematic`
- `--split`: Dataset split to use (default: `validation`)
- `--total-num`: Total number of samples for indexing (default: 1000)
- `--num-samples`: Number of comparative QA samples to generate (default: 100)
- `--k`: Number of videos/objects to compare (default: 2)
- `--nframes`: Number of frames for collision data (default: 32)
- `--time-mode`: Time mode - `timestamp` or `frame` (default: `timestamp`)
- `--threshold`: Velocity comparison threshold (default: 0.25)

---

### 7,8. Habitat Holistic Aggregation (`habitat_holistic`)

Generates object counting or route planning QA pairs.

```bash
# Object counting
python generate_data.py --dataset habitat_holistic \
    --root-path /path/to/habitat/data \
    --task object_counting \
    --section general \
    --total-num 10000 \
    --save \
    --verbose

# Route planning
python generate_data.py --dataset habitat_holistic \
    --root-path /path/to/habitat/data \
    --task route_plan \
    --route-mode medium \
    --save \
    --verbose
```

**Arguments:**
- `--root-path` (required): Root path for the Habitat dataset
- `--task` (required): Task type - `object_counting` or `route_plan`
- `--route-mode`: Path length mode - `all`, `longest`, or `medium` (default: `medium`)
- `--section`: Data section to use (default: `general`)
- `--total-num`: Maximum number of QA pairs to generate (default: 10000)
- `--save`: Include to save output to JSONL file
- `--verbose`: Enable verbose output

---



## Common Arguments

All datasets support these common arguments:

- `--save`: Save output to a JSONL file (optional)
- `--verbose`: Enable verbose terminal output (optional)

## Examples

### Generate all datasets for a validation run

```bash
# Kubric synchronization
python generate_data.py --dataset kubric_sync --path /data/kubric --total-num 100 --save

# Kubric spatial
python generate_data.py --dataset kubric_spatial --path /data/kubric --total-num 100 --save

# CLEVRER temporal alignment
python generate_data.py --dataset clevrer_temporal --root-path /data/clevrer \
    --total-num 100 --k 4 --total-frames 128 --save

# CLEVRER kinematic
python generate_data.py --dataset clevrer_comparative --root-path /data/clevrer \
    --task-type kinematic --total-num 100 --num-samples 50 --save

# CLEVRER collision
python generate_data.py --dataset clevrer_comparative --root-path /data/clevrer \
    --task-type collision --total-num 100 --num-samples 50 --save

# Habitat counting
python generate_data.py --dataset habitat_holistic --root-path /data/habitat \
    --task object_counting --total-num 1000 --save --verbose

# Habitat tracking
python generate_data.py --dataset habitat_tracking --root-path /data/habitat \
    --total-num 1000 --save --verbose
```

### Full production run

```bash
python generate_data.py --dataset kubric_sync --path /data/kubric --total-num 3000 --save
python generate_data.py --dataset kubric_spatial --path /data/kubric --total-num 3500 --save
python generate_data.py --dataset clevrer_temporal --root-path /data/clevrer --total-num 1000 --save
python generate_data.py --dataset clevrer_comparative --root-path /data/clevrer \
    --task-type kinematic --total-num 1083 --num-samples 799 --save
python generate_data.py --dataset habitat_tracking --root-path /data/habitat \
    --total-num 1630 --save --verbose
python generate_data.py --dataset habitat_holistic --root-path /data/habitat \
    --task object_counting --total-num 1389 --save --verbose
python generate_data.py --dataset habitat_holistic --root-path /data/habitat \
    --task route_plan --total-num 1266 --save --verbose
```

## Notes

- All `--root-path` and `--path` arguments can be relative or absolute paths
- Default values are set for optional arguments; override them as needed
- Use `--save` flag to persist output to JSONL files
- Use `--verbose` for debugging and to preview sample outputs
- Dataset-specific arguments are ignored if not applicable to the selected dataset
