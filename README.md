# SYNCR: A Cross-Video Reasoning Benchmark with Synthetic Grounding

<p align="center">
  <img src="https://img.shields.io/badge/Task-Cross--Video%20Reasoning-blue">
  <img src="https://img.shields.io/badge/Data-Synthetic-green">
</p>

[**🤗 Dataset**](https://huggingface.co/datasets/CrossVideoReasoning/SYNCR) | [**📄 Paper**](https://arxiv.org/abs/2605.08412) 

**SYNCR** is a fully synthetic diagnostic benchmark engineered to test the spatiotemporal boundaries of Multimodal Large Language Models (MLLMs) operating across independent, overlapping video streams. 

Unlike existing multi-video benchmarks that rely on human-annotated real-world footage, SYNCR leverages three advanced simulation engines—**CLEVRER**, **Kubric**, and **Habitat**—to eliminate annotation noise and provide deterministic, mathematically verifiable ground truth for complex, multi-perspective scenarios.

## 📖 Abstract

Multimodal Large Language Models (MLLMs) have demonstrated strong capabilities in single-video understanding, yet cross-video reasoning remains largely underexplored. Existing multi-video benchmarks primarily rely on human-annotated, real-world footage, which introduces annotator ambiguity and lacks absolute physical and spatial ground truth, complicating the diagnosis of model failures. To address this, we introduce **SYNCR**, a synthetic diagnostic benchmark designed to systematically evaluate MLLMs across independent, overlapping video streams. Leveraging three simulation engines (CLEVRER, Kubric, and Habitat), SYNCR decomposes multi-video understanding into four programmatically verified cognitive pillars: **Temporal Alignment**, **Spatial Tracking**, **Comparative Reasoning**, and **Holistic Synthesis**. 

## 🏛 The Four Cognitive Pillars

SYNCR decomposes cross-video understanding into four foundational diagnostic pillars:
1. **Temporal Alignment:** Synchronizing and sequencing disjointed events.
2. **Spatial Tracking:** Maintaining object permanence and resolving cross-view geometries.
3. **Comparative Reasoning:** Isolating kinematic or structural deviations.
4. **Holistic Synthesis:** Aggregating fragmented inputs into a global context.

## ⚙️ Repository Structure

This repository contains the codebase to procedurally generate all 8 SYNCR benchmark tasks using different 3D engines:

```
SYNCR/
├── generate_data.py              # Unified data generation script for all 8 tasks
├── DATA_GENERATION_GUIDE.md      # Comprehensive guide with all task specifications
├── README.md                     # This file
├── utils.py                      # Shared utilities
├── clevrer/                      # CLEVRER engine tasks
│   ├── base_clevrer.py
│   ├── clvr_temporal_alignment.py       # Task: Sequential ordering (Temporal Alignment)
│   ├── clvr_comparative_reasoning.py    # Task: Kinematic & Numerical comparison (Comparative Reasoning)
│   └── utils.py
├── kubric/                       # Kubric engine tasks
│   ├── base_kubric.py
│   ├── kubric_sync.py                  # Task: Multi-angle synchronization (Temporal Alignment)
│   ├── kubirc_spatial_meas.py          # Task: Spatial measurement (Spatial Tracking)
│   └── base_kubric.py
└── habitat/                      # Habitat engine tasks
    ├── base_habitat.py
    ├── habitat_spatial_tracking.py     # Task: Object re-identification (Spatial Tracking)
    └── habitat_holistic_aggregation.py # Tasks: Object counting & route planning (Holistic Synthesis)
```

## 🚀 Data Generation

### Quick Start

To generate SYNCR benchmark data, use the unified `generate_data.py` script:

```bash
python generate_data.py --dataset <DATASET_NAME> [OPTIONS]
```

### Available Datasets (8 Total Tasks)

**Temporal Alignment:**
- **`kubric_sync`** — Multi-angle temporal synchronization QA
- **`clevrer_temporal`** — Sequential ordering of video segments

**Spatial Tracking:**
- **`kubric_spatial`** — Spatial measurement 
- **`habitat_tracking`** — Object re-identification 

**Comparative Reasoning:**
- **`clevrer_comparative`** — Kinematic and collision comparison QA

**Holistic Synthesis:**
- **`habitat_holistic`** — Object counting and route planning QA

### Example Commands

```bash
# Temporal Alignment: Multi-angle synchronization
python generate_data.py --dataset kubric_sync --path /path/to/kubric --total-num 1000 --save

# Temporal Alignment: Sequential ordering
python generate_data.py --dataset clevrer_temporal --root-path /path/to/clevrer --total-num 1000 --save

# Spatial Tracking: Spatial measurement
python generate_data.py --dataset kubric_spatial --path /path/to/kubric --total-num 1000 --save

# Comparative Reasoning: Kinematic comparison
python generate_data.py --dataset clevrer_comparative --root-path /path/to/clevrer \
    --task-type kinematic --num-samples 100 --save

# Holistic Synthesis: Object counting
python generate_data.py --dataset habitat_holistic --root-path /path/to/habitat \
    --task object_counting --total-num 10000 --save --verbose
```

For complete documentation, argument specifications, and advanced usage, see [**DATA_GENERATION_GUIDE.md**](DATA_GENERATION_GUIDE.md).

## 📝 Citation

If you use SYNCR in your research, please consider citing our work:

```bibtex

```
