"""Answer parsing and scoring shared by the SYNCR evaluation scripts (and our GRPO reward).

Strict scoring (`is_correct_answer`, used for headline accuracy):
  - anchor on the LAST line carrying an answer marker (`Answer:`, `Final
    Answer:`, markdown-bold variants, missing space after the colon). A
    response that never commits to an answer scores 0, rather than a lucky
    substring match somewhere in its reasoning.
  - compare the option LETTER when both sides have one (every ground truth
    is "X) full option text"). A wrong letter is wrong however close the
    trailing text is; a right letter is right even if the text paraphrases.
  - fall back to normalized text containment (casefold, collapse whitespace,
    unify dash/underscore variants) only when no letter can be parsed, e.g.
    in --no-option mode, where the ground truth has no "X) " prefix.

Lenient scoring (`score_answer_lenient`, used for the statistics in the paper)
additionally accepts answers given without the marker; see below.
"""
import re

_ANSWER_MARKER_RE = re.compile(r'(?i)\**\s*(?:final\s+)?answer\s*\**\s*:\s*')
_OPTION_LETTER_RE = re.compile(r'^\s*\**\s*([A-Za-z])\s*(?:[).:]|$)')
_DASH_OR_UNDERSCORE_RE = re.compile(r'[‐-―_-]+')
_WHITESPACE_RE = re.compile(r'\s+')


def find_last_answer_line_index(generated_text):
    """Index (into generated_text.split("\n")) of the LAST line carrying a
    recognizable answer marker, or None if no line has one. Everything before
    that line is the reasoning body."""
    last_idx = None
    for i, line in enumerate(generated_text.split("\n")):
        if _ANSWER_MARKER_RE.search(line):
            last_idx = i
    return last_idx


def find_last_answer_text(generated_text):
    """Text following the marker on the LAST line that has one, e.g.
    'Answer: B) foo' -> 'B) foo', '**Final Answer:**B' -> 'B'. None if no
    line in generated_text carries a recognizable answer marker at all."""
    lines = generated_text.split("\n")
    idx = find_last_answer_line_index(generated_text)
    if idx is None:
        return None
    m = _ANSWER_MARKER_RE.search(lines[idx])
    return lines[idx][m.end():].strip()


def parse_option_letter(text):
    """Leading option letter from text, e.g. 'B) video_1 -- gray metal
    cube' -> 'B', '**B) ...**' -> 'B', a bare 'B' -> 'B'. None if text
    doesn't start with a letter immediately followed by ')', '.', ':', or
    end-of-string -- e.g. a plain descriptive answer with no lettered
    option, or ordinary prose that happens to start with 'A'/'I' (those
    require a following word character, which the ':'|'.'|')'|$
    alternation excludes).

    Known gap: 'B - video_1...' (dash) or 'B video_1...' (no punctuation)
    don't match, so such answers fall through to the normalized-text
    fallback. The evaluation prompt asks for the ')' format this expects.
    """
    m = _OPTION_LETTER_RE.match(text)
    return m.group(1).upper() if m else None


def normalize_answer_text(text):
    """casefold + unify dash/underscore variants to a space + collapse
    whitespace, so 'Video_1 -- Gray Metal Cube' and 'video 1  gray metal
    cube' compare equal."""
    text = _DASH_OR_UNDERSCORE_RE.sub(" ", str(text).casefold())
    return _WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Eval-only lenient scoring. Not used by the GRPO reward, which keeps
# requiring an answer marker (see the module docstring).
#
# Many baseline models ignore the "Answer: X) ..." format line and write e.g.
# "So the correct option is C) 6." or just "D) 8". is_correct_answer scores
# those as wrong, which measures format-following rather than the task. The
# lenient scorer keeps the strict result whenever an answer marker exists (so
# lenient >= strict, item by item) and otherwise looks for ONE clear
# commitment to an option, in this order:
#   1. the last explicit commitment phrase: "answer is C", "correct option is
#      (C)", "choice: C", "option C is correct";
#   2. a response whose only option-lettered line is its last non-empty line
#      ("D) 8", "(B)") -- a list of several lettered lines is an enumeration,
#      not a pick;
#   3. a response that is exactly one option's text ("8").
# The letter must be one of the item's option letters. Anything else is
# "unparsed" and scored wrong.
# ---------------------------------------------------------------------------
# the letter must follow on the same line: "the correct option is:\nA) ...\nB) ..." is an enumeration, not a pick
_L = r'[ \t\*"\'(\[]*([A-Z])(?=[ \t]*[)\].,:;\*"\']|[ \t]*$)'
_COMMIT_RES = [
    re.compile(r'(?i:\b(?:answer|choice|option)[ \t]+(?:is|would[ \t]+be|should[ \t]+be|will[ \t]+be)\b[ \t]*:?'
               r'|\b(?:answer|choice)[ \t]*[:=])[ \t]*(?i:option[ \t]+|choice[ \t]+)?' + _L, re.M),
    re.compile(r'(?i:\b(?:option|choice))[ \t]*[(\[]?([A-Z])[)\]]?[^\n.]{0,60}?'
               r'(?i:\bis\s+(?:the\s+)?(?:correct|right|best|answer))', re.M),
    # LaTeX-style final answers from thinking models: "\boxed{A}", "$\boxed{A)}$"
    re.compile(r'\\boxed\{[ \t]*\(?([A-Z])(?=[ \t]*[)}.:])'),
]
_LETTER_LINE_RE = re.compile(r'^\s*(?i:option[ \t]+)?[\*"\'(\[]*([A-Z])\s*(?:[)\].:]|$)')


def lenient_option_letter(generated_text, options):
    """(letter, rule) for a response without an answer marker, or (None, "unparsed").
    options: the item's option strings, e.g. ["A) 5", "B) 6", ...]."""
    letters = {parse_option_letter(o) for o in options} - {None}
    text = generated_text or ""
    best = None
    for rx in _COMMIT_RES:
        for m in rx.finditer(text):
            if m.group(1) in letters and (best is None or m.end() > best[1]):
                best = (m.group(1), m.end())
    if best:
        return best[0], "commitment"
    lines = [l for l in text.split("\n") if l.strip()]
    lettered = [i for i, l in enumerate(lines) if _LETTER_LINE_RE.match(l)]
    if len(lettered) == 1 and lettered[0] in (0, len(lines) - 1):
        letter = _LETTER_LINE_RE.match(lines[lettered[0]]).group(1)
        if letter in letters:
            return letter, "last_line"
    norm = normalize_answer_text(text).rstrip(".")
    exact = [parse_option_letter(o) for o in options
             if normalize_answer_text(re.sub(r'^\s*[A-Z]\)\s*', '', o)) == norm]
    if norm and len(exact) == 1:
        return exact[0], "option_text"
    return None, "unparsed"


def score_answer_lenient(generated_text, ground_truth, options):
    """Eval-only: (correct, rule). rule is "marker" when the strict parser applies
    (result identical to is_correct_answer), else the lenient rule used or "unparsed"."""
    if generated_text is None:
        return False, "unparsed"
    if find_last_answer_line_index(generated_text) is not None:
        return bool(is_correct_answer(generated_text, ground_truth)), "marker"
    letter, rule = lenient_option_letter(generated_text, options)
    return letter is not None and letter == parse_option_letter(str(ground_truth)), rule


def is_correct_answer(generated_text, ground_truth):
    """True if generated_text's last stated answer matches ground_truth.

    ground_truth is normally a string (e.g. "B) video_1 -- gray metal
    cube"). CrossVid's BU task stores it as a list of parts that must all
    appear in the answer text (kept from the original per-call-site
    handling in both the GRPO training code and eval/evaluate.py).
    """
    answer_text = find_last_answer_text(generated_text)
    if answer_text is None:
        return False

    if isinstance(ground_truth, list):
        norm_answer = normalize_answer_text(answer_text)
        return all(normalize_answer_text(part) in norm_answer for part in ground_truth)

    gt_letter = parse_option_letter(str(ground_truth))
    answer_letter = parse_option_letter(answer_text)
    if gt_letter is not None and answer_letter is not None:
        return answer_letter == gt_letter

    norm_gt = normalize_answer_text(ground_truth)
    norm_answer = normalize_answer_text(answer_text)
    return norm_gt in norm_answer or norm_answer in norm_gt
