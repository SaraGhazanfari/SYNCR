# SYNCR: Diagnosing and Learning Cross-Video Reasoning from Simulation

<p align="center">
  <img src="https://img.shields.io/badge/Task-Cross--Video%20Reasoning-blue">
  <img src="https://img.shields.io/badge/Data-Simulator--Grounded-green">
  <img src="https://img.shields.io/badge/Eval-4%2C000%20questions-orange">
  <img src="https://img.shields.io/badge/Train-15%2C960%20questions-purple">
</p>

[**📄 Paper**](https://arxiv.org/abs/2609.37918) | [**📄 Workshop paper (NeurIPS 2026 BabyVLM, oral)**](https://arxiv.org/abs/2605.08412) | [**🤗 Dataset**](https://huggingface.co/datasets/CrossVideoReasoning/SYNCR) | [**🤗 Checkpoints**](#checkpoints)

Reasoning across videos requires aligning events, matching identities, comparing motion, and integrating
partial observations. **SYNCR** is a simulator-grounded framework for diagnosing these abilities in multimodal
large language models and testing whether they can be learned. Built on **Habitat**, **Kubric**, and
**CLEVRER**, its task generators derive every answer from environment state, while models see only RGB
videos, the question, and the answer options. The same generators produce a benchmark of **4,000 evaluation
questions** and a training set of **15,960 questions** over disjoint videos, spanning eight tasks.

This repository contains the task generators, the scripts that build the released benchmark and training set,
and the evaluation harness used for the 22 models in the paper. We also release
[LoRA checkpoints](#checkpoints) fine-tuned on the SYNCR training set.

> The benchmark and zero-shot evaluation first appeared in the
> [workshop paper](https://arxiv.org/abs/2605.08412) (*SYNCR: A Cross-Video Reasoning Benchmark with Synthetic
> Grounding*, oral at the NeurIPS 2026 BabyVLM workshop). The [full paper](https://arxiv.org/abs/2609.37918)
> adds the training set, the fine-tuning experiments and the checkpoints.

## Tasks

Eight tasks in four families. Each question presents two to four clips; answer categories and option letters
are balanced within the first 200 and across all 500 evaluation items of every task.

| Family | Task | Engine | Videos / question | Clip length | Eval FPS | Chance |
|---|---|---|:-:|:-:|:-:|:-:|
| **Temporal Alignment** | Multi-Angle Synchronization: start offsets of three cropped camera views of one event | Kubric | 3 | 3.0 s | 12 | 25% |
| | Sequential Ordering: chronological order of four shuffled, gapped segments | CLEVRER | 4 | 0.87 s | 25 | 25% |
| **Spatial Tracking** | Object Re-identification: when an object seen in Video 1 first appears in Video 2 | Habitat | 2 | 20 s | 2 | 25% |
| | Spatial Measurement: which object is closest in 3D to a target when another leaves a view | Kubric | 2 | 5.0 s | 6 | 25% |
| **Comparative Reasoning** | Numerical Comparison: which video has more collisions, and by how many | CLEVRER | 2 | 5.12 s | 6.25 | 20% |
| | Kinematic Comparison: which of four objects reaches the highest peak speed | CLEVRER | 2 | 5.12 s | 6.25 | 25% |
| **Holistic Synthesis** | Object Counting: distinct instances of named categories across three walkthroughs | Habitat | 3 | 20 s | 2 | 25% |
| | Route Planning: shortest path between rooms shown in separate route clips | Habitat | 3 | 20 s | 2 | 25% |

## Results

Zero-shot accuracy (%) for selected models; the full table with all 22 models is in the paper. Open-weight
models use all 500 questions per task; proprietary models (†) and the four-person human majority vote use the
same first 100.

| Model | Sync | Order | ReID | Meas | Num | Kin | Count | Route | **Avg** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Human (majority vote) | 100.0 | 100.0 | 92.0 | 76.0 | 100.0 | 68.0 | 92.0 | 88.0 | **89.5** |
| GPT-6 Astra† | 88.0 | 100.0 | 88.0 | 52.0 | 60.0 | 20.0 | 48.0 | 60.0 | **64.5** |
| Gemini-3.1-Pro† | 60.0 | 96.0 | 68.0 | 48.0 | 16.0 | 8.0 | 32.0 | 40.0 | **46.0** |
| Qwen3-VL 32B | 30.0 | 64.4 | 56.2 | 35.2 | 33.6 | 19.4 | 28.6 | 32.4 | **37.5** |
| InternVL3.5 38B | 26.0 | 44.4 | 34.6 | 39.0 | 41.0 | 26.2 | 30.0 | 32.8 | **34.3** |
| Qwen3-VL 8B | 29.6 | 56.4 | 47.6 | 32.6 | 29.6 | 17.2 | 30.2 | 29.4 | **34.1** |
| LLaVA-OV 72B | 21.8 | 26.0 | 31.2 | 30.0 | 22.6 | 18.6 | 27.4 | 23.6 | **25.2** |

Physical comparison and scene integration remain hard at every scale. Supervised fine-tuning on the SYNCR
training set raises Qwen3-VL-8B's average from 32.6% to 61.6% (on matched 200-item sets), and temporal
ordering gains transfer to real footage (Assembly101, Panoptic); see [Checkpoints](#checkpoints).

## Checkpoints

LoRA adapters fine-tuned on all eight tasks (the 15,960-question training set, answer-only targets, one
epoch, rank 16, frozen vision encoder). Accuracy (%) on the first 200 evaluation items per task:

| Adapter | Base model | Base avg | SFT avg |
|---|---|:-:|:-:|
| [CrossVideoReasoning/SYNCR-Qwen3-VL-8B-SFT](https://huggingface.co/CrossVideoReasoning/SYNCR-Qwen3-VL-8B-SFT) | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 32.6 | **61.6** |
| [CrossVideoReasoning/SYNCR-Qwen3-VL-4B-SFT](https://huggingface.co/CrossVideoReasoning/SYNCR-Qwen3-VL-4B-SFT) | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | – | **62.6** |
| [CrossVideoReasoning/SYNCR-InternVL3.5-8B-SFT](https://huggingface.co/CrossVideoReasoning/SYNCR-InternVL3.5-8B-SFT) | [OpenGVLab/InternVL3_5-8B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF) | 29.3 | **62.8** |

Download an adapter and evaluate it on top of its base model:

```bash
huggingface-cli download CrossVideoReasoning/SYNCR-Qwen3-VL-8B-SFT --local-dir checkpoints/SYNCR-Qwen3-VL-8B-SFT
MODEL=Qwen/Qwen3-VL-8B-Instruct ADAPTER_PATH=checkpoints/SYNCR-Qwen3-VL-8B-SFT N=200 bash scripts/eval_syncr.sh
```

For the InternVL adapter use `MODEL=OpenGVLab/InternVL3_5-8B-HF`.

## Repository layout

```
SYNCR/
├── generation/          task generators (CLEVRER, Kubric, Habitat) and benchmark build scripts
│   └── README.md        how the data is generated and how the released files were built
├── eval/
│   ├── evaluate.py          run a model on one task file
│   ├── evaluate_vllm.py     faster vLLM path (Qwen3-VL only)
│   ├── answer_parse.py      answer parsing (strict and lenient scoring)
│   ├── rescore_predictions.py, retry_failed_items.py
│   └── stats_report.py, controls_report.py   Wilson intervals, McNemar tests, evidence controls
├── models/              wrappers for Qwen2.5-VL / Qwen3-VL, InternVL3 / 3.5, LLaVA-OneVision, GPT, Gemini
├── configs/             one config per task: eval file, frame rate, batch size, generation budget
├── scripts/
│   ├── prepare_data.py      converts the Hugging Face release into per-task eval files
│   └── eval_syncr.sh        runs all eight tasks for one model
└── utils.py
```

## Installation

```bash
git clone https://github.com/SaraGhazanfari/SYNCR.git && cd SYNCR
pip install -r requirements.txt
```

Install `openai` or `google-genai` for API models, and `flash-attn` if your GPU supports it.

## Evaluating a model

**1. Download the data.**

```bash
huggingface-cli download CrossVideoReasoning/SYNCR --repo-type dataset --local-dir hf_data
cd hf_data && mkdir -p videos && for f in *_data.tar; do tar -xf "$f" -C videos/; done && cd ..
```

The Habitat and Kubric videos are in the archives. **CLEVRER videos are not redistributed**: download
`video_validation.zip` (and `video_train.zip` for the training set) from <http://clevrer.csail.mit.edu> and
extract them so that paths like `hf_data/videos/clvr/video_validation/video_11194.mp4` exist. If an archive
extracts into subfolders of 1,000 videos, move the files up one level.

**2. Write the per-task evaluation files.**

```bash
python scripts/prepare_data.py --dataset-root hf_data --absolute-video-paths
```

This writes `json_files/eval/<Family>/<Task>.jsonl`, the files the configs point to, and reports any video
source it cannot find.

**3. Evaluate.** All eight tasks with the paper's settings:

```bash
MODEL=Qwen/Qwen3-VL-8B-Instruct bash scripts/eval_syncr.sh
```

or a single task:

```bash
python -m eval.evaluate --model_version Qwen/Qwen3-VL-8B-Instruct \
    --filepath json_files/eval/Comparative_Reasoning/Clvr_Kinematic_Comparison_optv3c.jsonl \
    --fps 6.25 --batch_size 2 --sample-num 500
```

The wrapper is chosen from the model name: `qwen` (Qwen2.5-VL, Qwen3-VL), `intern` (InternVL3 / 3.5 via remote
code; names ending in `-hf` use the native transformers classes), `llava` (LLaVA-OneVision), `gpt`, `gemini`.
API keys are read from `OPENAI_API_KEY` / `GEMINI_API_KEY`, or from `tokens/{gpt,gemini}_api_key.json`.

**4. Report.** Per-item predictions are saved under `eval_predictions/hf/<model>/`. Accuracy with 95% Wilson
intervals and a bootstrapped task average:

```bash
python -m eval.stats_report --tag Qwen__Qwen3-VL-8B-Instruct --n 500
# paired comparison of two models on the same items (exact McNemar test per task)
python -m eval.stats_report --tag <model A> --compare <model B> --n 500
```

### Protocol

- Videos are passed in annotation order and labelled `Video 1:`, `Video 2:`, ... The question refers to them
  by these labels, so order matters.
- Clips with `start_sec` / `end_sec` are cropped before they reach the model. For Qwen, per-frame timestamps
  are rebased to start at 0 in each clip, and Gemini receives pre-cut clips, so absolute timestamps cannot
  reveal the answer to Ordering or Synchronization.
- Decoding is deterministic where the backend supports it. The prompt asks for
  `Answer: <letter>) <option text>`. `eval.evaluate` prints strict accuracy, which requires that line. The
  paper reports lenient accuracy, computed from the saved predictions by `eval.stats_report` (or
  `eval.rescore_predictions`): the final `Answer:` line when one exists, otherwise an explicit option
  commitment, a single lettered final line, or an exact option text. Unparsed responses count as wrong.
- Give reasoning models a large enough generation budget. Synchronization and Ordering use
  `max_new_tokens=8192` (set in their configs); at 2,048 tokens many answers are cut off before the answer
  line and score as wrong.

### Evidence controls

| Flag | Control |
|---|---|
| `--no-video` | question and options only; tests for text shortcuts |
| `--single-video` | one randomly chosen video per item; tests dependence on multiple inputs |
| `--no-option` | open-ended answering instead of multiple choice |

Run them with `CONDITION=novideo` or `CONDITION=single` in `scripts/eval_syncr.sh`, then summarize with
`python -m eval.controls_report --tag <model> --n <N> --suite syncr`.

## Generating data

See [`generation/README.md`](generation/README.md) for the raw data layout, the question generators, and the
staged build of the released evaluation and training files.

## Citation

```bibtex
@article{ghazanfari2026syncr,
  title   = {SYNCR: Diagnosing and Learning Cross-Video Reasoning from Simulation},
  author  = {Ghazanfari, Sara and Garg, Siddharth and Krishnamurthy, Prashanth and Khorrami, Farshad},
  journal = {arXiv preprint arXiv:2609.37918},
  year    = {2026}
}

@article{ghazanfari2026syncrbenchmark,
  title   = {SYNCR: A Cross-Video Reasoning Benchmark with Synthetic Grounding},
  author  = {Ghazanfari, Sara and Garg, Siddharth and Krishnamurthy, Prashanth and Khorrami, Farshad},
  journal = {arXiv preprint arXiv:2605.08412},
  year    = {2026}
}
```

## License

The code is released under the MIT License (`LICENSE`). The annotations on Hugging Face are released under
CC BY 4.0. Rendered videos derive from CLEVRER, Kubric, and Habitat (HM3D) and remain subject to their upstream
licenses.
