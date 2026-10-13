from models import GeminiModel, GPTModel, InternVLModel, QwenVLModel, LlavaOneVisionModel
import os, argparse, json, random
from eval.answer_parse import is_correct_answer
from utils import write_jsonl_file, read_jsonl_file, my_print
import time
import traceback
import torch
from concurrent.futures import ThreadPoolExecutor


class SYNCREvaluation:
    def __init__(self, model_name, model_version, cache_dir, adapter_path=None):
        if model_name == 'internvl_hf':
            # Imported here: needs transformers' native InternVL classes, which an environment set up for the
            # remote-code InternVL / LLaVA wrappers may not have.
            from models.internvl_hf import InternVLHFModel
            self.model = InternVLHFModel(model_version, cache_dir=cache_dir, adapter_path=adapter_path)
            return
        self.model = {'internvl': InternVLModel,
                      'gpt': GPTModel,
                      'gemini': GeminiModel,
                      'llava': LlavaOneVisionModel,
                      'qwenvl': QwenVLModel}[model_name](model_version, cache_dir=cache_dir,
                                                         adapter_path=adapter_path)

    def _build_batch_texts(self, args, samples, fps):
        """CPU-only: prompt/GT construction, no video decode. Split out from
        evaluate_batch so the video-decode step (prepare_batch) can be
        prefetched independently -- see prepare_batch's docstring."""
        text_parts = []
        video_lists = []
        gt_list = []

        for idx, sample in enumerate(samples):
            if 'fps' in sample['question']:
                sample['question'] = sample['question'].format(fps=fps)

            if not args.no_option:
                prompt = sample['question'] + "\nOptions:\n" + "\n".join(sample['options'])
                prompt += '\nFormat Instructions:\nProvide your final answer in its own line and using exactly this format:  Answer: <option_alphabet>) <full_option_text>'
                gt_list.append(sample['answer'])
            else:
                prompt = sample['question']
                prompt += '\nFormat Instructions:\nProvide your final answer in its own line and using exactly this format:  Answer: <final answer>'
                gt_list.append(sample['answer'].split(')')[-1].strip())

            if getattr(args, "force_guess", False):
                # Blind baseline for models that refuse without videos (GPT-6): ask for a best guess instead.
                prompt += "\nThe videos are not available. Answer anyway: choose the single most likely option using only the question and the options."
            text_parts.append(prompt)
            if not args.no_video:
                video_lists.append(sample['video_list'])
            else:
                video_lists.append([])

        return text_parts, video_lists, gt_list

    def prepare_batch(self, args, samples, fps):
        """
        CPU-bound: prompt construction + video decode (via model.prepare_batch_inputs,
        which calls qwen_vl_utils' decord decode), no GPU work. Meant to be run on a
        background thread for batch N+1 while batch N's generate_batch runs on GPU --
        the eval loop's original plain synchronous for-loop had zero overlap between
        these, which is what tripped the cluster's GPU-utilization cancellation
        policy on eval jobs (confirmed: eval only ever requests 1 GPU, so it isn't
        the multi-GPU pipeline-sharding idle-bubble that caused the training-job
        cancellations fixed earlier -- this is a second, independent instance of the
        same underlying "no prefetching" problem, in a different code path).

        Only implemented for the qwenvl model path (prepare_batch_inputs is
        QwenVLModel-specific); other model backends fall back to no prefetch.
        """
        text_parts, video_lists, gt_list = self._build_batch_texts(args, samples, fps)
        if not hasattr(self.model, "prepare_batch_inputs"):
            return None, text_parts, video_lists, gt_list
        inputs = self.model.prepare_batch_inputs(text_parts, video_lists, verbose=True, fps=fps)
        return inputs, text_parts, video_lists, gt_list

    def generate_batch(self, args, prepared, fps):
        """GPU-bound half matching prepare_batch's output. Falls back to the
        original non-prefetched path when prepare_batch_inputs isn't available
        for this model backend (inputs is None)."""
        inputs, text_parts, video_lists, gt_list = prepared
        if inputs is None:
            return self.model.get_response_batch(text_parts=text_parts, video_lists=video_lists,
                                                 gt_list=gt_list, fps=fps, max_new_tokens=args.max_new_tokens)
        return self.model.generate_from_inputs(inputs, text_parts, gt_list, verbose=True,
                                               max_new_tokens=args.max_new_tokens)

    def evaluate_batch(self, args, samples, fps):
        """Unchanged behavior/API for any caller that doesn't need prefetching."""
        prepared = self.prepare_batch(args, samples, fps)
        return self.generate_batch(args, prepared, fps)
 
 
def get_model_name(model_version):
    model_version = model_version.lower()

    if 'gemini' in model_version:
        return 'gemini'
    if 'gpt' in model_version:
        return 'gpt'
    if 'llava' in model_version:
        return 'llava'
    if 'intern' in model_version and model_version.rstrip('/').endswith('-hf'):
        return 'internvl_hf'
    if 'intern' in model_version:
        return 'internvl'
    if 'qwen' in model_version:
        return 'qwenvl'
    else:
        return model_version   
    
def predictions_path(args):
    """eval_predictions/hf/<adapter or model tag>[_novideo]/<task file stem>__n<N>.jsonl
    (engine subdirectory: the vLLM path writes under vllm/ or vllm_lora/, so runs never overwrite each other)."""
    tag = args.adapter_path or args.model_version
    tag = tag.strip("./").replace("/", "__") + ("_novideo" if args.no_video else "") + ("_forced" if getattr(args, "force_guess", False) else "") + ("_nooption" if args.no_option else "") + ("_single" if getattr(args, "single_video", False) else "")
    stem = os.path.splitext(os.path.basename(args.filepath))[0]
    n = args.sample_num if args.sample_num is not None else "all"
    return os.path.join(args.predictions_dir, "hf", tag, f"{stem}__n{n}.jsonl")


def save_predictions(args, records):
    """One line per scored item (index into the file, answer, raw prediction, correct, failed, error).
    Items never reached (e.g. a crash before the end) are left out and show as missing indices."""
    if not args.predictions_dir:
        return
    path = predictions_path(args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Keep the previous run rather than clobbering it. A base checkpoint's predictions
    # live at one path per model and task, whichever sweep produced them, so re-running
    # a base model for a new analysis silently destroys the predictions an earlier
    # result was derived from. That happened on 2026-09-21: a construct-validity sweep
    # re-ran Qwen3-VL-8B base on Panoptic ordering and overwrote the file behind a
    # number already in the paper, which then no longer reproduced (58/200 vs 57/200 --
    # ordinary decoding variation, but invisible once the original was gone).
    # Superseded files are kept alongside with a timestamp so both runs stay auditable.
    if os.path.exists(path):
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(os.path.getmtime(path)))
        backup = f"{path[:-6]}.superseded_{stamp}.jsonl" if path.endswith(".jsonl") else path + f".superseded_{stamp}"
        try:
            os.replace(path, backup)
            my_print(f"Kept previous predictions as {os.path.basename(backup)}")
        except OSError as e:
            my_print(f"WARNING: could not preserve previous predictions ({e}); overwriting")
    with open(path, "w") as f:
        for r in records:
            if r is not None:
                f.write(json.dumps({**r, "model_version": args.model_version, "adapter_path": args.adapter_path,
                                    "filepath": args.filepath, "engine": "hf"}) + "\n")
    my_print(f"Saved per-item predictions to {path}")


# Helper function to chunk lists into batches
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def evaluate_models(args):   
    model_name = get_model_name(args.model_version)
    cache_dir = None  # Hugging Face's default cache ($HF_HOME)
    
    benchmark_samples = read_jsonl_file(filepath=args.filepath)
    if args.sample_num is not None:
        benchmark_samples = benchmark_samples[:args.sample_num]
    if getattr(args, "single_video", False):
        # Single-video control (MVU-Eval Table 4 style): keep one randomly chosen video per item, question and
        # options unchanged. Seeded per file, so every model sees the same kept video for the same item.
        rng = random.Random(f"single-video|{os.path.basename(args.filepath)}")
        for sample in benchmark_samples:
            kept = rng.randrange(len(sample['video_list']))
            sample['video_list'] = [sample['video_list'][kept]]
            sample['_kept_video'] = kept
    my_print(f"Evaluating model: {args.model_version} in batches of {args.batch_size} ...")
    count, total = 0, 0
    # Every item gets a record, including items of batches that failed to prepare or generate:
    # those count as wrong (they used to be skipped, which silently shrank the denominator).
    n_failed = 0
    records = [None] * len(benchmark_samples)
    syncr_eval = SYNCREvaluation(model_name=model_name, model_version=args.model_version,
                                cache_dir=cache_dir, adapter_path=args.adapter_path)
    wrong_indices = list()

    batches = list(chunker(benchmark_samples, args.batch_size))

    # Prefetch pipeline: while the GPU generates for batch N, up to
    # PREFETCH_DEPTH background threads decode videos + build tensors for
    # batches N+1..N+DEPTH (prepare_batch), so CPU-bound decode overlaps with
    # GPU-busy time instead of happening serially after it. Depth > 1 (not
    # just one batch ahead) because some tasks' videos are expensive enough
    # to decode that a single batch's prep time exceeds one batch's generate
    # time -- observed directly: CrossVid's "cook" category (1651 videos,
    # some triggering repeated ffmpeg "mmco: unref short failure" warnings,
    # i.e. genuinely slower/quirkier to decode) left the GPU at a sustained
    # 0% for over 20 seconds even with 1-deep prefetching, since that only
    # hides up to one generate-call's worth of decode time. A deeper queue
    # gives multiple videos' worth of decode a head start, and multiple
    # threads let the CPUs allocated to this job (see --cpus-per-task)
    # actually decode more than one batch concurrently rather than one at a
    # time. The plain synchronous loop this replaced entirely (zero overlap)
    # is what tripped the cluster's GPU-utilization cancellation policy on
    # eval jobs in the first place (eval only ever requests 1 GPU, so this
    # isn't the multi-GPU pipeline-sharding idle-bubble that caused the
    # training-job cancellations fixed earlier).
    #
    # Bumped from 3 to 5 for "cook"'s GPU-utilization problem (see above --
    # depth=3 alone left the GPU at 0% for 20+s on that category, and a real
    # cluster utilization-policy WARNING landed at 50.67%
    # average). But depth=5 turned out to trade that risk for a worse one on
    # CrossVid's NC ("movie") category specifically: sacct
    # showed MaxRSS hit 268,401,176 KB -- essentially the full 256GB ceiling
    # -- and the process was SIGKILLed by the OOM killer (exit 137, no
    # Python-level exception at all, unlike the catchable CUDA OOM handled
    # below). Root cause: decord's vr.get_batch(idx) only *returns* the
    # requested sparse frame indices, but for H.264-style encodes with
    # widely-spaced keyframes, decoding any target frame requires decoding
    # through the intervening P/B-frames since the last keyframe -- so a
    # handful of sparse indices from a very long, low-keyframe-density movie
    # clip can still force decord to internally decode (and hold) a large
    # fraction of the video. With 5 such decodes potentially in flight at
    # once, more than one long "movie" clip landing in the same prefetch
    # window is enough to exhaust 256GB. This has hit essentially every
    # adapter's NC task at least once this session. Backed down to 3
    # (depth=3 was never the thing that caused an OOM -- only depth=5 was
    # observed doing so -- so this keeps the "cook" utilization fix's original
    # value rather than reverting to the pre-fix depth=1 that caused it).
    PREFETCH_DEPTH = 3
    executor = ThreadPoolExecutor(max_workers=PREFETCH_DEPTH)

    def _prepare(idx):
        return syncr_eval.prepare_batch(args, batches[idx], args.fps)

    futures = {}
    for i in range(min(PREFETCH_DEPTH, len(batches))):
        futures[i] = executor.submit(_prepare, i)

    for batch_idx in range(len(batches)):
        next_idx = batch_idx + PREFETCH_DEPTH
        # Keep the pipeline full: submit the next not-yet-queued batch before
        # blocking on this one's result, so preparation stays DEPTH batches
        # ahead of whichever one is currently generating.
        if next_idx < len(batches):
            futures[next_idx] = executor.submit(_prepare, next_idx)

        first = batch_idx * args.batch_size

        def mark_failed(stage, err):
            nonlocal total, n_failed
            for j, sample in enumerate(batches[batch_idx]):
                records[first + j] = {"index": first + j, "answer": sample["answer"], "prediction": None,
                                      "correct": False, "failed": True, "error": f"{stage}: {type(err).__name__}: {err}"[:500]}
            total += len(batches[batch_idx])
            n_failed += len(batches[batch_idx])

        try:
            prepared = futures.pop(batch_idx).result()
        except Exception as e:
            tb_str = traceback.format_exc()
            my_print(f"Error occurred while preparing batch {batch_idx+1}: {e}\n{tb_str}")
            mark_failed("prepare", e)
            continue

        try:
            preds = syncr_eval.generate_batch(args, prepared, args.fps)
            batch_samples = batches[batch_idx]
            assert len(preds) == len(batch_samples), f"{len(preds)} predictions for {len(batch_samples)} samples"

            # Unpack the predictions back to the individual samples
            for j, (sample, pred) in enumerate(zip(batch_samples, preds)):
                if args.no_option:
                    sample['answer'] = sample['answer'].split(')')[-1].strip()

                # eval.answer_parse.is_correct_answer:
                # anchors on the last recognizable answer-marker line (no
                # whole-body fallback scan when a response never commits to
                # one -- same anti-reward-hacking reasoning the old lenient
                # fallback here was missing), matches on the option LETTER
                # as the primary key when both sides have one, and falls
                # back to normalized (casefold, dash/underscore-unified)
                # text containment otherwise. Replaces the old
                # `ground_truth in line` verbatim substring check, which
                # scored a genuinely correct letter-only answer, a single-
                # dash variant, a capitalization difference, or a missing
                # space after the colon all as wrong. sample['answer'] is a
                # plain string for every task except CrossVid's BU, which
                # stores it as a list (e.g. ['D']) -- is_correct_answer
                # handles both.
                correct = bool(is_correct_answer(pred, sample['answer']))
                count += correct
                total += 1
                records[first + j] = {"index": first + j, "answer": sample["answer"], "prediction": pred,
                                      "correct": correct, "failed": False, "error": None,
                                      **({"kept_video": sample["_kept_video"]} if "_kept_video" in sample else {})}
            my_print(f"Model: {args.model_version}, Batch {batch_idx+1} Complete, Accuracy So Far: {count/total:.4f} ({count}/{total})")
        except Exception as e:
            tb_str = traceback.format_exc()
            my_print(f"Error occurred while evaluating batch {batch_idx+1}: {e}\n{tb_str}")
            mark_failed("generate", e)
            if isinstance(e, torch.cuda.OutOfMemoryError):
                # A caught CUDA OOM leaves a lot of "reserved but unallocated"
                # memory sitting in PyTorch's caching allocator (confirmed via
                # the error's own message: e.g. "10.58 GiB is reserved by
                # PyTorch but unallocated") --
                # without releasing it, that fragmentation compounds across
                # subsequent batches until even small allocations fail,
                # cascading one recoverable OOM into total task failure
                # instead of just this one skipped batch (observed: CC/NC
                # have repeatedly needed full-task backfills, most recently
                # traced to exactly this pattern rather than the video-decode
                # segfault fixed earlier). empty_cache() returns the reserved
                # blocks to the driver so the next batch starts from a clean
                # slate.
                torch.cuda.empty_cache()

    executor.shutdown(wait=True)

    if total == n_failed:
        # (Failed items count as wrong below; a task where EVERY item failed is still reported as
        # unscored and exits non-zero, so eval_task.SBATCH does not record a 0.0 for a broken task.)
        # Every batch failed at prepare or generate time (e.g. the video
        # files a task's dataset file points to aren't present on disk).
        # count/total would raise ZeroDivisionError here; that's not a
        # transient error a resume retry can fix, so report it plainly
        # instead of crashing the whole eval job over one broken task.
        my_print(f"Final Accuracy for {args.model_version}, data:{args.filepath}, FPS: {args.fps}: NO VALID SAMPLES (0/0) -- every batch failed, see errors above")
    else:
        my_print(f"Final Accuracy for {args.model_version}, data:{args.filepath}, FPS: {args.fps}: {count/total:.4f} ({count}/{total})")
    # Separate line so the "Final Accuracy" line keeps the format the eval scripts parse.
    my_print(f"Failed items: {n_failed}/{total} (counted as wrong)")
    save_predictions(args, records)
    if total == n_failed:
        raise SystemExit(1)

    if args.save:        
        write_jsonl_file(filepath=f"json_files/data_with_{model_name}_response.jsonl", 
                         data_list=benchmark_samples) 

    with open("wrong_indices.txt", "w") as f:
        for item in wrong_indices:
            f.write(str(item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multimodal video models.")
    
    # Add command-line arguments
    parser.add_argument("--model_version", type=str, default="Qwen/Qwen3-VL-8B-Instruct", 
                        help="The model version/path to evaluate.")
    
    parser.add_argument("--adapter_path", type=str, default=None, 
                        help="Path to the LoRA adapter weights.")
    
    parser.add_argument("--filepath", type=str, default="json_files/eval/Temporal_Alignment/Kubric_Multi_Angle_Synchronization_optv3c.jsonl",
                        help="Path to the JSONL dataset.")
    parser.add_argument("--fps", type=float, default=None, 
                        help="frame per second rate")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for model inference.")
    parser.add_argument("--save", action="store_true", 
                        help="Include this flag to save the results to a new JSONL file.")
    parser.add_argument("--no-video", action="store_true", 
                        help="Text only inference without video context")
    parser.add_argument("--force-guess", action="store_true", default=bool(os.environ.get("FORCE_GUESS")),
                        help="With --no-video: tell the model the videos are unavailable and ask for a best-guess option "
                             "(default on when FORCE_GUESS is set)")
    parser.add_argument("--no-option", action="store_true",
                        help="Open form inference without options")
    parser.add_argument("--single-video", action="store_true",
                        help="Keep one randomly chosen video per item (seeded per file); question and options unchanged")
    parser.add_argument("--max-new-tokens", type=int, default=2048, 
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--sample-num", type=int, default=None,
                        help="Score only the first N items of the file.")
    parser.add_argument("--predictions-dir", default="eval_predictions",
                        help="Per-item predictions go to <dir>/<model tag>/<task>__n<N>.jsonl; empty string disables.")
    args = parser.parse_args()

    # Execute with parsed arguments
    evaluate_models(args)
    
    
