"""Re-query only the items that failed to run in a saved predictions file.

API evals can lose single items to transient server errors (e.g. Gemini 503 UNAVAILABLE); those items are saved with
"failed": true and counted wrong. This script re-asks just those items with the same prompt construction as
eval/evaluate.py (SYNCREvaluation._build_batch_texts) and the same model wrapper, and writes the merged predictions to a
new file -- the original predictions file is never overwritten. Items that succeeded are kept exactly as they were.

  python -m eval.retry_failed_items --predictions <file.jsonl> --model_version gemini-3-flash-preview --fps 12 \
      --max-new-tokens 32768 --out <new_file.jsonl>
"""
import argparse
import json
import os
from types import SimpleNamespace

from eval.evaluate import SYNCREvaluation, get_model_name
from eval.answer_parse import is_correct_answer
from utils import read_jsonl_file


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=2048)
    parser.add_argument("--cache_dir", default=None, help="model cache (default: Hugging Face's)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"FATAL: {args.out} exists, not overwriting")

    rows = [json.loads(l) for l in open(args.predictions)]
    failed = [r for r in rows if r.get("failed")]
    print(f"{args.predictions}: {len(rows)} items, {len(failed)} failed -> re-querying those only", flush=True)
    samples = read_jsonl_file(rows[0]["filepath"])
    evaluator = SYNCREvaluation(get_model_name(args.model_version), args.model_version, args.cache_dir)
    eval_args = SimpleNamespace(no_option=False, no_video=False, max_new_tokens=args.max_new_tokens)

    for r in failed:
        sample = dict(samples[r["index"]])
        text_parts, video_lists, gt_list = evaluator._build_batch_texts(eval_args, [sample], args.fps)
        try:
            text = evaluator.model.get_response_batch(text_parts=text_parts, video_lists=video_lists, gt_list=gt_list,
                                                      fps=args.fps, max_new_tokens=args.max_new_tokens)[0]
        except Exception as e:
            print(f"item {r['index']} failed again: {str(e)[:200]}", flush=True)
            continue
        r.update(prediction=text, correct=bool(text is not None and is_correct_answer(text, r["answer"])),
                 failed=False, error=None, retried=True)
        print(f"item {r['index']} retried: correct={r['correct']}", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_ok = sum(bool(r["correct"]) for r in rows)
    still_failed = sum(bool(r.get("failed")) for r in rows)
    print(f"Merged accuracy {n_ok / len(rows):.4f} ({n_ok}/{len(rows)}), still failed {still_failed}; wrote {args.out}")


if __name__ == "__main__":
    main()
