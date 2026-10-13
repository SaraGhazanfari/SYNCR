"""Evaluate Qwen3-VL on a list of task files with one vLLM engine (run in vllm_env).

Same scoring protocol as eval/evaluate.py (the HF path), faster:
- the model loads once and scores every task file given;
- vLLM batches all requests of a task (continuous batching), no fixed batch size;
- frames are decoded once per (video, clip bounds, fps, max_pixels) and cached on disk,
  so every later model/adapter reuses them.

Kept identical to the HF path:
- prompt text and message layout (models/qwenvl.py prepare_batch_inputs);
- frame sampling: qwen_vl_utils.process_vision_info with max_pixels=100352 and the task fps;
- clip-relative timestamps: utils.make_timestamps_clip_relative on the sampled metadata, and the
  frames are passed pre-sampled (do_sample_frames=False) so vLLM's processor labels "<t seconds>"
  from exactly those indices;
- decoding: greedy, repetition_penalty 1.15, max_new_tokens 2048;
- scoring: eval.answer_parse.is_correct_answer; failed items count as wrong.

Outputs match the HF path: per task a line "LABEL<TAB>file<TAB>Final Accuracy for ..." appended to
--results_tsv (tasks already recorded there are skipped = resume), per-item predictions under
--predictions_dir, plus "Failed items: N/total" in the log.

LoRA adapters are merged into the base weights once (cached under --merged_dir) and the merged
model is served; --adapter_path is recorded in the outputs.

  python -m eval.evaluate_vllm --results_tsv eval_results/x.tsv \
      --task json_files/eval/.../A.jsonl:25 --task json_files/cvid/NC_data.jsonl:0.3 --task json_files/mvu/SU_data.jsonl:none
"""
import argparse
import glob
import hashlib
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

# The HF path decodes with decord (qwen_vl_utils' default in our HF environment). In the vLLM environment the
# default pick is torchcodec, which could not load libavutil on our cluster, and its torchvision fallback
# calls torchvision.io.read_video, which torchvision 0.28 removed. Force decord for identical frames.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

import numpy as np

from eval.answer_parse import is_correct_answer
from utils import make_timestamps_clip_relative, my_print, read_jsonl_file

MAX_PIXELS = 100352
FORMAT_OPTIONS = ('\nFormat Instructions:\nProvide your final answer in its own line and using exactly this format:  '
                  'Answer: <option_alphabet>) <full_option_text>')
FORMAT_OPEN = '\nFormat Instructions:\nProvide your final answer in its own line and using exactly this format:  Answer: <final answer>'


# ----------------------------------------------------------------------------- prompts (HF parity)

def build_prompt_text(sample, fps, no_option):
    """eval/evaluate.py SYNCREvaluation._build_batch_texts, one sample."""
    question = sample["question"].format(fps=fps) if "fps" in sample["question"] else sample["question"]
    if not no_option:
        return question + "\nOptions:\n" + "\n".join(sample["options"]) + FORMAT_OPTIONS, sample["answer"]
    return question + FORMAT_OPEN, sample["answer"].split(")")[-1].strip()


def video_element(video, fps):
    """models/qwenvl.py prepare_batch_inputs: one video entry of the user message."""
    if isinstance(video, dict):
        path = video["video_path"]
        start, end = (video["start_sec"], video["end_sec"]) if "start_sec" in video and "end_sec" in video else (0.0, float("inf"))
    else:
        path, start, end = video, 0.0, float("inf")
    ele = {"type": "video", "video": path, "video_start": start, "video_end": end, "max_pixels": MAX_PIXELS}
    if fps is not None:
        ele["fps"] = fps
    return ele


def build_message(text, video_list, fps, no_video):
    message = {"role": "user", "content": []}
    for i, video in enumerate([] if no_video else video_list):
        message["content"].append({"type": "text", "text": f"Video {i+1}:\n"})
        message["content"].append(video_element(video, fps))
        message["content"].append({"type": "text", "text": "\n"})
    message["content"].append({"type": "text", "text": text})
    return message


# ----------------------------------------------------------------------------- frames + cache

def frame_key(ele):
    spec = json.dumps({k: ele.get(k) for k in ("video", "video_start", "video_end", "max_pixels", "fps")}, sort_keys=True,
                      default=str)
    return hashlib.sha1(spec.encode()).hexdigest()


def load_frames(ele, cache_dir):
    """(frames uint8 np.ndarray, clip-relative metadata dict, video_kwargs) for one video element."""
    import torch
    from qwen_vl_utils import process_vision_info

    path = os.path.join(cache_dir, frame_key(ele)[:2], frame_key(ele) + ".npz") if cache_dir else None
    if path and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return z["frames"], meta["metadata"], meta["video_kwargs"]
    _, videos, video_kwargs = process_vision_info([[{"role": "user", "content": [ele]}]], return_video_kwargs=True,
                                                  return_video_metadata=True)
    frames, metadata = videos[0]
    metadata = dict(metadata)
    metadata["frames_indices"] = [int(i) for i in metadata["frames_indices"]]
    metadata = make_timestamps_clip_relative([metadata])[0]
    metadata = {k: (v.item() if hasattr(v, "item") else v) for k, v in metadata.items()}
    # qwen_vl_utils returns resized float32 frames in [0, 255]; the cache stores uint8 (rounding moves a
    # pixel by < 0.5/255, far below bf16 engine noise) to keep ~6 MB per 22-frame 252x364 clip, not 24 MB.
    frames = frames.round().clamp(0, 255).to(torch.uint8).numpy()
    video_kwargs = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in video_kwargs.items()}
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".tmp{os.getpid()}.npz"
        np.savez(tmp, frames=frames, meta=json.dumps({"metadata": metadata, "video_kwargs": video_kwargs, "element": ele},
                                                     default=str))
        os.replace(tmp, path)
    return frames, metadata, video_kwargs


def build_request(sample, args, fps, processor):
    text, gt = build_prompt_text(sample, fps, args.no_option)
    message = build_message(text, sample.get("video_list", []), fps, args.no_video)
    prompt = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    request = {"prompt": prompt}
    videos = [c for c in message["content"] if c.get("type") == "video"]
    if videos:
        items, kwargs = [], {}
        for ele in videos:
            frames, metadata, video_kwargs = load_frames(ele, args.frame_cache_dir)
            items.append((frames, {**metadata, "do_sample_frames": False}))
            kwargs = video_kwargs
        request["multi_modal_data"] = {"video": items}
        request["mm_processor_kwargs"] = {k: v for k, v in kwargs.items() if k != "fps"} | {"do_sample_frames": False}
    return request, gt


# ----------------------------------------------------------------------------- model

def resolve_model(args):
    """Local snapshot of the base model, or a merged copy with the LoRA adapter applied."""
    snaps = sorted(glob.glob(os.path.join(args.cache_dir, "hub", "models--" + args.model_version.replace("/", "--"),
                                          "snapshots", "*")) +
                   glob.glob(os.path.join(args.cache_dir, "models--" + args.model_version.replace("/", "--"),
                                          "snapshots", "*")))
    # Some models are cached twice (under hub/ and directly under cache_dir); a snapshot can hold only the config
    # (Qwen3-VL-2B's hub/ copy failed vLLM with "Cannot find any model weights"). Use one with weights.
    snaps = [s for s in snaps if glob.glob(os.path.join(s, "*.safetensors")) or glob.glob(os.path.join(s, "*.bin"))]
    base = snaps[0] if snaps else args.model_version
    if not args.adapter_path:
        return base
    tag = os.path.abspath(args.adapter_path).strip("/").replace("/", "__")
    merged = os.path.join(args.merged_dir, tag)
    if os.path.exists(os.path.join(merged, "config.json")):
        return merged
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    my_print(f"Merging LoRA adapter {args.adapter_path} into {base} -> {merged}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    tmp = merged + f".tmp{os.getpid()}"
    model.save_pretrained(tmp, safe_serialization=True)
    AutoProcessor.from_pretrained(base).save_pretrained(tmp)
    os.replace(tmp, merged)
    return merged


# ----------------------------------------------------------------------------- scoring one task

def recorded(results_tsv, key):
    if not os.path.exists(results_tsv):
        return False
    with open(results_tsv) as f:
        return any(line.startswith(key + "\t") for line in f)


def predictions_path(args, filepath, sample_num):
    """eval_predictions/<vllm|vllm_lora>/<adapter or model tag>[_novideo]/<task>__n<N>.jsonl -- the engine (and, for
    adapters, merged weights vs LoRA request) is part of the path so HF, merged and LoRA-request runs never collide."""
    tag = (args.adapter_path or args.model_version).strip("./").replace("/", "__")
    tag += ("_novideo" if args.no_video else "") + ("_nooption" if args.no_option else "")
    stem = os.path.splitext(os.path.basename(filepath))[0]
    engine = "vllm_lora" if getattr(args, "lora_adapters", None) else "vllm"
    return os.path.join(args.predictions_dir, engine, tag, f"{stem}__n{sample_num if sample_num else 'all'}.jsonl")


def run_task(llm, processor, sampling, args, filepath, fps, lora_request=None):
    label = "FINETUNED" if args.adapter_path else "BASE"
    key = f"{label}\t{filepath}"
    if args.results_tsv and recorded(args.results_tsv, key):
        my_print(f"SKIPPED {filepath}: already recorded in {args.results_tsv}")
        return
    samples = read_jsonl_file(filepath=filepath)
    if args.sample_num:
        samples = samples[:args.sample_num]
    t0 = time.time()
    records = [None] * len(samples)
    count = n_failed = 0
    # Build requests (frame decode / cache) on CPU threads, chunk by chunk so frames are not all in RAM at once.
    pool = ThreadPoolExecutor(max_workers=args.decode_workers)
    for lo in range(0, len(samples), args.chunk_size):
        chunk = list(range(lo, min(lo + args.chunk_size, len(samples))))

        def prep(i):
            try:
                return build_request(samples[i], args, fps, processor)
            except Exception as e:
                return e

        built = list(pool.map(prep, chunk))
        ok = [(i, b) for i, b in zip(chunk, built) if not isinstance(b, Exception)]
        for i, b in zip(chunk, built):
            if isinstance(b, Exception):
                records[i] = {"index": i, "answer": samples[i]["answer"], "prediction": None, "correct": False,
                              "failed": True, "error": f"prepare: {type(b).__name__}: {b}"[:500]}
                n_failed += 1
                my_print(f"[{os.path.basename(filepath)}] item {i} failed to prepare: {b}")
        if not ok:
            continue
        try:
            outputs = llm.generate([b[0] for _, b in ok], sampling_params=sampling, use_tqdm=False,
                                   lora_request=lora_request)
        except Exception as e:
            my_print(f"[{os.path.basename(filepath)}] generate failed for items {ok[0][0]}-{ok[-1][0]}: {e}\n{traceback.format_exc()}")
            outputs = [None] * len(ok)
        for (i, (_, gt)), out in zip(ok, outputs):
            text = out.outputs[0].text if out is not None and out.outputs else None
            if text is None:
                records[i] = {"index": i, "answer": samples[i]["answer"], "prediction": None, "correct": False,
                              "failed": True, "error": "generate: no output"}
                n_failed += 1
                continue
            correct = bool(is_correct_answer(text, gt))
            count += correct
            records[i] = {"index": i, "answer": samples[i]["answer"], "prediction": text, "correct": correct,
                          "failed": False, "error": None, "prompt_tokens": len(out.prompt_token_ids),
                          "completion_tokens": len(out.outputs[0].token_ids), "finish_reason": out.outputs[0].finish_reason}
            if i < 5:  # parity evidence: the "<t seconds>" markers vLLM actually put in the prompt
                decoded = processor.tokenizer.decode(out.prompt_token_ids)
                records[i]["timestamp_markers"] = re.findall(r"<[0-9.]+ seconds>", decoded)
        done = sum(r is not None for r in records)
        my_print(f"[{os.path.basename(filepath)}] {done}/{len(samples)} scored, accuracy so far {count/done:.4f}, "
                 f"failed {n_failed}, {time.time() - t0:.0f}s")
    pool.shutdown()
    total = len(samples)
    fps_text = fps if fps is not None else None
    if n_failed == total:
        line = (f"Final Accuracy for {args.model_version}, data:{filepath}, FPS: {fps_text}: NO VALID SAMPLES (0/0) "
                f"-- every item failed")
    else:
        line = f"Final Accuracy for {args.model_version}, data:{filepath}, FPS: {fps_text}: {count/total:.4f} ({count}/{total})"
    my_print(line)
    my_print(f"Failed items: {n_failed}/{total} (counted as wrong); wall time {time.time() - t0:.0f}s")
    if args.predictions_dir:
        path = predictions_path(args, filepath, args.sample_num)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps({**r, "model_version": args.model_version, "adapter_path": args.adapter_path,
                                    "filepath": filepath, "engine": "vllm"}) + "\n")
        my_print(f"Saved per-item predictions to {path}")
    if args.results_tsv and n_failed < total:
        os.makedirs(os.path.dirname(args.results_tsv) or ".", exist_ok=True)
        with open(args.results_tsv, "a") as f:
            f.write(f"{key}\t{line}\n")
    return n_failed < total


def parse_task(spec):
    """'file.jsonl:fps' (fps a number or 'none'), or a scripts/configs/*.conf path."""
    if spec.endswith(".conf"):
        conf = dict(line.strip().split("=", 1) for line in open(spec) if "=" in line and not line.startswith("#"))
        fps = conf.get("fps", "none").strip('"')
        return conf["filepath"].strip('"'), None if fps == "none" else float(fps)
    path, _, fps = spec.rpartition(":") if spec.count(":") else (spec, "", "none")
    return path, None if fps in ("", "none") else float(fps)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", action="append", required=True, help="file.jsonl:fps | file.jsonl:none | a .conf path")
    parser.add_argument("--model_version", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--cache_dir", default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    parser.add_argument("--merged_dir", default="merged_models")
    parser.add_argument("--frame_cache_dir", default=".cache/frames", help="empty string disables")
    parser.add_argument("--results_tsv", default=None)
    parser.add_argument("--predictions_dir", default="eval_predictions")
    parser.add_argument("--sample-num", dest="sample_num", type=int, default=200, help="0 = all items")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=2048)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--no-video", dest="no_video", action="store_true")
    parser.add_argument("--no-option", dest="no_option", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=65536)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=64, help="requests built and submitted per chunk")
    parser.add_argument("--decode_workers", type=int, default=6)
    parser.add_argument("--limit_videos_per_prompt", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="GPUs for one engine (e.g. 2 for Qwen3-VL-32B)")
    parser.add_argument("--lora_adapters", nargs="+", default=None,
                        help="Serve several LoRA adapters/checkpoints from ONE engine (vLLM LoRA requests, base weights "
                             "loaded once) instead of merging --adapter_path. --results_tsv may contain {adapter}, "
                             "replaced by each adapter's tag; without it each adapter gets eval_results/<tag>__vllm.tsv.")
    parser.add_argument("--merge_adapter", action="store_true",
                        help="Merge --adapter_path into the base weights instead of serving it as a LoRA request. "
                             "Default is the LoRA request: on the v3b Ordering adapter (gapped Ordering, 200 items) it "
                             "matched HF (0.980 vs HF 0.975) while the merged model scored 0.955 (6 of 7 differing "
                             "items went against the merged model).")
    args = parser.parse_args()
    if args.adapter_path and not args.merge_adapter:
        # a single adapter is served as a LoRA request; results still go to the --results_tsv given
        args.lora_adapters, args.adapter_path = [args.adapter_path], None

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    engine_kwargs = dict(dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
                         tensor_parallel_size=args.tensor_parallel_size,
                         max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs, seed=0,
                         limit_mm_per_prompt={"video": args.limit_videos_per_prompt, "image": 0})
    if args.lora_adapters:
        assert not args.adapter_path, "use either --adapter_path (merged) or --lora_adapters (LoRA requests)"
        ranks = [json.load(open(os.path.join(a, "adapter_config.json")))["r"] for a in args.lora_adapters]
        model_path = resolve_model(args)
        engine_kwargs.update(enable_lora=True, max_loras=min(len(args.lora_adapters), 4), max_lora_rank=max(ranks))
    else:
        model_path = resolve_model(args)
    processor = AutoProcessor.from_pretrained(model_path)
    llm = LLM(model=model_path, **engine_kwargs)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                              repetition_penalty=args.repetition_penalty, seed=0)

    if args.lora_adapters:
        from vllm.lora.request import LoRARequest
        failures = 0
        template = args.results_tsv
        for k, adapter in enumerate(args.lora_adapters, start=1):
            tag = adapter.strip("./").replace("/", "__")
            args.adapter_path = adapter
            if template and "{adapter}" in template:
                args.results_tsv = template.replace("{adapter}", tag)
            elif template and len(args.lora_adapters) == 1:
                args.results_tsv = template  # single adapter: the file the caller asked for
            else:
                args.results_tsv = os.path.join("eval_results", f"{tag}__vllm.tsv")
            request = LoRARequest(tag, k, os.path.abspath(adapter))
            my_print(f"vLLM engine ready: {model_path}, LoRA request {k}/{len(args.lora_adapters)}: {adapter}")
            for spec in args.task:
                filepath, fps = parse_task(spec)
                try:
                    if run_task(llm, processor, sampling, args, filepath, fps, lora_request=request) is False:
                        failures += 1
                except Exception as e:
                    failures += 1
                    my_print(f"TASK FAILED {adapter} {filepath}: {e}\n{traceback.format_exc()}")
        raise SystemExit(1 if failures else 0)
    my_print(f"vLLM engine ready: {model_path} (adapter {args.adapter_path})")
    failures = 0
    for spec in args.task:
        filepath, fps = parse_task(spec)
        try:
            if run_task(llm, processor, sampling, args, filepath, fps) is False:
                failures += 1
        except Exception as e:
            failures += 1
            my_print(f"TASK FAILED {filepath}: {e}\n{traceback.format_exc()}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
