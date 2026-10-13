"""Re-score saved per-item predictions with the eval-only lenient parser.

Reads eval_predictions/<engine>/<model tag>/<task file stem>__n<N>.jsonl (written by eval/evaluate.py and
eval/evaluate_vllm.py), looks up each item's options in its eval file, and reports per model and task:
- strict accuracy (eval/answer_parse.is_correct_answer, the score in eval_results/; re-computed and checked against
  the stored "correct" field),
- lenient accuracy (eval/answer_parse.score_answer_lenient: strict when an "Answer:" marker exists, otherwise one
  clear commitment to an option),
- the share of items without a marker, and of items still unparsed after the lenient rules.

  python -m eval.rescore_predictions --glob 'eval_predictions/hf/*/*__n500.jsonl' --out eval_results/rescore_v3e_<stamp>.json
  python -m eval.rescore_predictions --glob '...' --examples 5     (print rescued and unparsed examples per file)
"""
import argparse
import collections
import glob
import json
import os

from eval.answer_parse import is_correct_answer, score_answer_lenient
from utils import read_jsonl_file


def rescore_file(path, examples=0):
    rows = [json.loads(l) for l in open(path)]
    samples = read_jsonl_file(rows[0]["filepath"])
    rules = collections.Counter()
    strict = lenient = mismatched = 0
    shown = collections.Counter()
    for r in rows:
        sample = samples[r["index"]]
        text = r.get("prediction")
        s = bool(text is not None and is_correct_answer(text, r["answer"]))
        mismatched += s != bool(r["correct"])
        ok, rule = score_answer_lenient(text, r["answer"], sample["options"])
        strict += s
        lenient += ok
        rules[rule] += 1
        kind = "rescued" if ok and not s else ("unparsed" if rule == "unparsed" else None)
        if examples and kind and shown[kind] < examples:
            shown[kind] += 1
            print(f"  [{kind}:{rule}] gt={r['answer']!r} :: {str(text)[-300:]!r}")
    n = len(rows)
    return dict(file=path, model=rows[0].get("model_version"), adapter=rows[0].get("adapter_path"),
                eval_file=rows[0]["filepath"], n=n, strict=strict / n, lenient=lenient / n,
                no_marker=1 - rules["marker"] / n, unparsed=rules["unparsed"] / n, rules=dict(rules),
                stored_mismatch=mismatched)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--glob", required=True)
    parser.add_argument("--out", default=None, help="JSON summary (refuses to overwrite)")
    parser.add_argument("--examples", type=int, default=0)
    args = parser.parse_args()
    if args.out and os.path.exists(args.out):
        raise SystemExit(f"FATAL: {args.out} exists, not overwriting")
    results = []
    for path in sorted(glob.glob(args.glob)):
        if args.examples:
            print(f"== {path}")
        res = rescore_file(path, args.examples)
        results.append(res)
        print(f"{res['model']}\t{os.path.basename(res['eval_file'])}\tstrict {res['strict']:.3f}\tlenient "
              f"{res['lenient']:.3f}\tno marker {res['no_marker']:.1%}\tunparsed {res['unparsed']:.1%}\t"
              f"rules {res['rules']}\tstored mismatch {res['stored_mismatch']}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
