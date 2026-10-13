"""
Distractor construction shared by the SYNCR generators (generation/) and the
re-optioning script (generation/build/reoption.py).

The option audit found that several tasks built
their wrong options from the correct answer, so the answer could be picked
from the options alone. Every builder here avoids that in one of two ways:

- Distractors are drawn from the same distribution as real answers, so the
  answer is exchangeable with them: Multi-Angle Sync (independent draws of the
  crop-offset process), Object ReID (windows borrowed from other items'
  answers with a similar Video-1 window and duration), Spatial Measurement
  (real scene objects, synthetic names in the same format as a last resort).
- When the answer distribution is skewed (Counting, Route, Numerical), the
  option set shown for each answer value is sampled from a distribution fitted
  so that, given the options alone, every option is as close to equally
  likely to be correct as the answer distribution allows
  (fit_balanced_option_sets).

Every function takes an explicit random.Random (or the `random` module).
"""
import collections
import itertools
import math
import re


# ---------------------------------------------------------------------------
# Lettering
# ---------------------------------------------------------------------------

def letter_options(bodies, answer_body):
    """Letter options A), B), ... in the given order; return (options, answer)."""
    assert bodies.count(answer_body) == 1, (answer_body, bodies)
    options = [f"{chr(65 + i)}) {b}" for i, b in enumerate(bodies)]
    return options, options[bodies.index(answer_body)]


def shuffled_letter_options(answer_body, distractors, rng):
    bodies = [answer_body] + list(distractors)
    assert len(set(bodies)) == len(bodies), bodies
    rng.shuffle(bodies)
    return letter_options(bodies, answer_body)


# ---------------------------------------------------------------------------
# Balanced option sets
# ---------------------------------------------------------------------------

def fit_balanced_option_sets(answer_counts, candidate_sets, features, iters=3000, lr=0.5):
    """Per answer value v, a distribution q[v] over candidate option sets.

    answer_counts   {v: number of items whose answer is v}
    candidate_sets  v -> iterable of hashable option sets S that contain v
    features        S -> Counter{option feature: how many options carry it}

    A model that sees only S and knows the answer distribution P picks the
    feature with the highest P(f) q[f](S) / mult(f, S); its accuracy is
    sum_S max_f P(f) q[f](S) / mult(f, S) ("blind_bayes_acc"). We minimise
    the smooth convex surrogate sum_S sum_f J(f,S)^2 / (mult(f,S) T(S)),
    J = P(f) q[f](S), T = sum_f J(f,S) (the expected collision probability of
    the option posterior) by exponentiated gradient, which pushes every
    option set towards equal posteriors. When one value holds more than
    1/len(S) of the answers, the floor is that value's share: re-optioning
    cannot remove a skewed answer prior.

    Returns (q, stats).
    """
    total = sum(answer_counts.values())
    P = {v: n / total for v, n in answer_counts.items() if n}
    q, feats, by_set = {}, {}, collections.defaultdict(list)
    for v in sorted(P, key=repr):
        sets = list(dict.fromkeys(candidate_sets(v)))
        assert sets, f"no candidate option set for answer {v!r}"
        q[v] = {S: 1.0 / len(sets) for S in sets}
        for S in sets:
            if S not in feats:
                feats[S] = features(S)
            assert feats[S].get(v, 0) >= 1, (v, S)
            by_set[S].append(v)

    def bayes():
        acc = 0.0
        for S, vs in by_set.items():
            acc += max(P[v] * q[v][S] / feats[S][v] for v in vs)
        return acc

    start_acc = bayes()
    for _ in range(iters):
        grad = {v: {} for v in q}
        for S, vs in by_set.items():
            J = {v: P[v] * q[v][S] for v in vs}
            T = sum(J.values())
            if T < 1e-300:
                continue
            m = feats[S]
            post = {v: J[v] / T for v in vs}
            collision = sum(post[v] ** 2 / m[v] for v in vs)
            for v in vs:  # d/dJ of the surrogate, rescaled per answer value (EG is scale-free)
                grad[v][S] = 2 * post[v] / m[v] - collision
        for v, dist in q.items():
            g = grad[v]
            new = {S: max(w * math.exp(-lr * g.get(S, 0.0)), 1e-300) for S, w in dist.items()}
            z = sum(new.values())
            q[v] = {S: w / z for S, w in new.items()}
    stats = {"blind_bayes_acc_before_fit": round(start_acc, 4),
             "blind_bayes_acc": round(bayes(), 4),
             "majority_answer_share": round(max(P.values()), 4),
             "answer_distribution": {repr(v): answer_counts[v] for v in sorted(P, key=repr)}}
    return q, stats


def sample_option_set(dist, rng, feasible=None):
    """Draw S from {S: weight}, restricted to feasible(S); None if nothing is feasible."""
    items = [(S, w) for S, w in dist.items() if feasible is None or feasible(S)]
    if not items:
        return None
    x = rng.random() * sum(w for _, w in items)
    for S, w in items:
        x -= w
        if x <= 0:
            return S
    return items[-1][0]


# ---------------------------------------------------------------------------
# Multi-Angle Sync
# ---------------------------------------------------------------------------

def sync_option_text(video2_offset_sec, video3_offset_sec):
    return (f"Video 2 starts at {video2_offset_sec:+.1f} seconds; "
            f"Video 3 starts at {video3_offset_sec:+.1f} seconds.")


def sync_distractors(answer_text, rng, n=3, fps=12, original_duration=5, crop_duration=3):
    """Offsets from fresh, independent runs of get_time_sync_qa's crop sampling.

    The old distractors were the swapped, sign-flipped and fuzzed answer, so
    the answer was the one option with both twins (97.5% of eval items).
    """
    total_frames, crop_frames = int(original_duration * fps), int(crop_duration * fps)
    out, seen = [], {answer_text}
    while len(out) < n:
        t1, t2, t3 = rng.sample(range(total_frames - crop_frames + 1), 3)
        text = sync_option_text((t2 - t1) / float(fps), (t3 - t1) / float(fps))
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


# ---------------------------------------------------------------------------
# Object ReID
# ---------------------------------------------------------------------------

REID_ABSENT_OPTION = "The object does not appear in this video."
_REID_V1_RE = re.compile(r"visible from (\d+(?:\.\d+)?)s to (\d+(?:\.\d+)?)s")
_REID_WINDOW_RE = re.compile(r"^(\d+(?:\.\d+)?)s to (\d+(?:\.\d+)?)s$")


def _tenths(x):
    return int(round(float(x) * 10))


def reid_window_text(start_tenths, end_tenths):
    return f"{start_tenths / 10:.1f}s to {end_tenths / 10:.1f}s"


def reoption_reid(questions, answer_bodies, video_tenths, rng, n_neighbours=30):
    """Options for a whole ReID file (distractors are borrowed across items).

    The old distractors were the answer shifted by +1 s, -1 s, +2 s, ..., so
    the answer was the second-earliest window on 90% of items. Here each
    distractor is the answer window (start and length) of another item whose
    Video-1 window has a similar start and length, so answer and distractor
    windows share one distribution given everything the question shows --
    including the many windows that start at 0.0 s and the tendency to start
    after the Video-1 window. (Keeping the answer's own length for every
    option, as before, is not exchangeable: no distractor could start at
    0.0 s when the answer does, so such answers would always be earliest.)

    questions / answer_bodies / video_tenths: per item; video_tenths is the
    length of Video 2 in 0.1 s. When some answers are REID_ABSENT_OPTION,
    present items also offer that option, at the rate that makes it correct
    about 1 time in 4.
    Returns [(options, answer)] aligned with the input.
    """
    n = len(questions)
    v1, windows = [], []
    for q, a in zip(questions, answer_bodies):
        m = _REID_V1_RE.search(q)
        assert m, q
        v1.append((_tenths(m.group(1)), _tenths(m.group(2)) - _tenths(m.group(1))))
        w = _REID_WINDOW_RE.match(a)
        if w:
            start, end = _tenths(w.group(1)), _tenths(w.group(2))
            assert end > start, a
            windows.append((start, end))
        else:
            assert a == REID_ABSENT_OPTION, a
            windows.append(None)
    present = [i for i in range(n) if windows[i] is not None]
    assert present, "ReID re-optioning needs items whose object appears"
    p_absent = 1 - len(present) / n
    offer_absent = min(1.0, 3 * p_absent / (1 - p_absent))

    out = []
    for i in range(n):
        T = video_tenths[i]
        if windows[i] is not None:
            assert windows[i][1] <= T, (i, windows[i], T)
        n_windows = 2 if windows[i] is not None and rng.random() < offer_absent else 3
        donors = [j for j in present if j != i]
        rng.shuffle(donors)  # random tie-break for the stable sort
        donors.sort(key=lambda j: abs(v1[j][0] - v1[i][0]) + abs(v1[j][1] - v1[i][1]))
        near, far = donors[:n_neighbours], donors[n_neighbours:]
        rng.shuffle(near)
        rng.shuffle(far)
        used, chosen = {answer_bodies[i]}, []
        for j in near + far:
            if len(chosen) == n_windows:
                break
            text = reid_window_text(*windows[j])
            if windows[j][1] <= T and text not in used:
                used.add(text)
                chosen.append(text)
        assert len(chosen) == n_windows, f"item {i}: fewer than {n_windows} distinct donor windows"
        distractors = chosen + ([REID_ABSENT_OPTION] if n_windows == 2 else [])
        out.append(shuffled_letter_options(answer_bodies[i], distractors, rng))
    return out


# ---------------------------------------------------------------------------
# Object Counting
# ---------------------------------------------------------------------------

def reoption_counting(answers, rng, window=4):
    """Options for a whole Counting file.

    The old distractors were count + randint(-3, 3) clipped at 0, so with
    most answers 1-2 the answer was the second-smallest option (62%) and 0
    was offered often but never correct. Option sets are 4 distinct values
    within +/-window of the answer, drawn from fit_balanced_option_sets.
    Returns ([(options, answer)], stats).
    """
    def candidates(v):
        return [S for S in itertools.combinations(range(max(0, v - window), v + window + 1), 4) if v in S]

    q, stats = fit_balanced_option_sets(collections.Counter(answers), candidates, collections.Counter)
    out = []
    for v in answers:
        S = sample_option_set(q[v], rng)
        out.append(shuffled_letter_options(str(v), [str(u) for u in S if u != v], rng))
    return out, stats


# ---------------------------------------------------------------------------
# Route Planning
# ---------------------------------------------------------------------------

def parse_route_option(text):
    """'Region 1 <a> -> b -> Region 2 <c>' -> ('a', ['b'], 'c'); None if malformed."""
    parts = text.split(" -> ")
    if len(parts) < 2:
        return None
    m1 = re.fullmatch(r"Region 1 <(.*)>", parts[0])
    m2 = re.fullmatch(r"Region 2 <(.*)>", parts[-1])
    if not (m1 and m2):
        return None
    return m1.group(1), parts[1:-1], m2.group(1)


def route_option_text(start, middles, goal):
    return " -> ".join([f"Region 1 <{start}>", *middles, f"Region 2 <{goal}>"])


def _route_middles_available(pool, answer_middles):
    """{r: number of distinct r-room middles a distractor can use}."""
    avail = {}
    for r in range(1, len(pool) + 1):
        avail[r] = math.perm(len(pool), r)
        if r == len(answer_middles) and len(set(answer_middles)) == r:
            avail[r] -= 1  # the answer's own middle
    return avail


def reoption_route(answer_bodies, pools, rng):
    """Options for a whole Route Planning file.

    Old problems: the direct path was offered but never correct (sub-paths
    have >= 3 regions), None names became option strings, rooms sharing a
    description gave start==goal questions and duplicate options. Here
    distractors never use the direct path, are built only from named rooms
    in `pools` (per item; the answer's own middle rooms are always included,
    as in the generator), and their hop counts come from
    fit_balanced_option_sets over the answers' hop counts.

    Returns ([(options, answer) or (None, drop_reason)], stats).
    """
    results = [None] * len(answer_bodies)
    parsed = {}
    for i, a in enumerate(answer_bodies):
        p = parse_route_option(a)
        if p is None:
            results[i] = (None, "unparseable_answer")
            continue
        start, middles, goal = p
        if not middles:
            results[i] = (None, "direct_path_answer")
        elif start == goal:
            results[i] = (None, "start_equals_goal")
        else:
            pool = list(dict.fromkeys(list(middles) + [x for x in pools[i] if x and x != "None"]))
            parsed[i] = (start, middles, goal, pool)
    if not parsed:
        return results, {}

    hops = {i: len(p[1]) + 1 for i, p in parsed.items()}
    max_hops = max(hops.values())

    def candidates(h):
        return [S for S in itertools.combinations_with_replacement(range(2, max_hops + 1), 4) if h in S]

    q, stats = fit_balanced_option_sets(collections.Counter(hops.values()), candidates, collections.Counter)
    for i, (start, middles, goal, pool) in parsed.items():
        h = hops[i]
        avail = _route_middles_available(pool, middles)

        def feasible(S):
            rest = collections.Counter(S)
            rest[h] -= 1
            return all(c <= avail.get(x - 1, 0) for x, c in rest.items() if c > 0)

        S = sample_option_set(q[h], rng, feasible)
        if S is None:
            results[i] = (None, "too_few_rooms_for_distinct_options")
            continue
        rest = list(S)
        rest.remove(h)
        used, distractors = {answer_bodies[i]}, []
        for x in rest:
            distractors.append(_draw_route_middle(pool, x - 1, start, goal, used, rng))
        results[i] = shuffled_letter_options(answer_bodies[i], distractors, rng)
    return results, stats


def _draw_route_middle(pool, r, start, goal, used, rng):
    for _ in range(200):
        text = route_option_text(start, rng.sample(pool, r), goal)
        if text not in used:
            used.add(text)
            return text
    perms = list(itertools.permutations(pool, r))
    rng.shuffle(perms)
    for p in perms:
        text = route_option_text(start, p, goal)
        if text not in used:
            used.add(text)
            return text
    raise AssertionError(f"no distinct {r}-room middle left in {pool}")


# ---------------------------------------------------------------------------
# Numerical Comparison
# ---------------------------------------------------------------------------

NUMERICAL_TIE_OPTION = "most and second most have the same number of collisions -- 0"


def numerical_answer_text(collision_counts):
    ranked = sorted(collision_counts, reverse=True)
    diff = ranked[0] - ranked[1]
    if diff == 0:
        return NUMERICAL_TIE_OPTION
    return f"Video {collision_counts.index(ranked[0]) + 1} -- {diff}"


def reoption_numerical(collision_counts_list, rng):
    """Options for a whole Numerical Comparison file.

    The old non-tie options were {answer, tie, second video + true diff,
    winner + fake diff, second video + fake diff}: the answer alone shared
    its video with one option and its diff with another. Now every item
    shows the tie option plus a 2x2 grid {video a, video b} x {diff d1, d2},
    so every grid option shares its video with one option and its diff with
    another. The grid for each answer is drawn from fit_balanced_option_sets
    (answers skew towards diff 1-2).
    Returns ([(options, answer)], stats).
    """
    keys = []
    for counts in collision_counts_list:
        text = numerical_answer_text(counts)
        keys.append((len(counts), "tie" if text == NUMERICAL_TIE_OPTION else
                     (text.split(" -- ")[0], int(text.split(" -- ")[1]))))

    fits, stats = {}, {}
    for k in sorted({kk for kk, _ in keys}):
        group = [key for kk, key in keys if kk == k]
        max_diff = max([key[1] for key in group if key != "tie"] + [1]) + 1
        videos = [f"Video {i + 1}" for i in range(k)]
        all_sets = [(va, vb, d1, d2) for va, vb in itertools.combinations(videos, 2)
                    for d1, d2 in itertools.combinations(range(1, max_diff + 1), 2)]

        def candidates(key, all_sets=all_sets):
            if key == "tie":
                return all_sets
            return [S for S in all_sets if key[0] in S[:2] and key[1] in S[2:]]

        def features(S):
            va, vb, d1, d2 = S
            return collections.Counter(["tie", (va, d1), (va, d2), (vb, d1), (vb, d2)])

        fits[k], stats[f"{k}_videos"] = fit_balanced_option_sets(collections.Counter(group), candidates, features)

    out = []
    for counts, (k, key) in zip(collision_counts_list, keys):
        va, vb, d1, d2 = sample_option_set(fits[k][key], rng)
        grid = [f"{v} -- {d}" for v in (va, vb) for d in (d1, d2)]
        answer = numerical_answer_text(counts)
        # The grid always contains a non-tie answer; the tie option completes five options.
        distractors = grid if answer == NUMERICAL_TIE_OPTION else [NUMERICAL_TIE_OPTION] + [g for g in grid if g != answer]
        assert len(distractors) == 4, (answer, grid)
        out.append(shuffled_letter_options(answer, distractors, rng))
    return out, stats


# ---------------------------------------------------------------------------
# Spatial Measurement
# ---------------------------------------------------------------------------

KUBRIC_SIZES = ["Small", "Large"]
KUBRIC_MATERIALS = ["Rubber", "Metal"]
KUBRIC_SHAPES = ["Cube", "Sphere", "Cylinder"]
KUBRIC_COLORS = ["Blue", "Brown", "Cyan", "Gray", "Green", "Purple", "Red", "Yellow"]


def spatial_distractors(ranked_names, correct, exclude, scene_names, rng, n=3):
    """Distractor object names for Spatial Measurement.

    The old padding used fake names ("Huge Purple Cylinder") with no material
    and a size vocabulary real objects never use, so the answer was always an
    option with a material. Order of preference:
      1. ranked_names: objects visible at the exit frame, closest to the
         target first (same as the old generator);
      2. other objects of the same scene, in random order;
      3. synthetic names in the scene's own get_object_name format
         (Size [Color] Material Shape), not naming any scene object.
    """
    banned = set(exclude) | {correct}
    out = []
    for name in ranked_names:
        if len(out) == n:
            return out
        if name not in banned and name not in out:
            out.append(name)
    rest = [x for x in scene_names if x not in banned and x not in out]
    rng.shuffle(rest)
    out.extend(rest[:n - len(out)])
    with_color = 2 * sum(len(x.split()) == 4 for x in scene_names) >= len(scene_names)
    colors = sorted({x.split()[1] for x in scene_names if len(x.split()) == 4}) or KUBRIC_COLORS
    while len(out) < n:
        words = [rng.choice(KUBRIC_SIZES)] + ([rng.choice(colors)] if with_color else []) + \
                [rng.choice(KUBRIC_MATERIALS), rng.choice(KUBRIC_SHAPES)]
        name = " ".join(words)
        if name not in banned and name not in out and name not in scene_names:
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# v3 builders: residual cues found in the v2 files.
# The v2 functions above are kept unchanged so the _optv2 files stay reproducible.
# ---------------------------------------------------------------------------

import random as _random

def _weighted_choice(weights, rng):
    items = sorted(weights.items(), key=lambda kv: repr(kv[0]))
    x = rng.random() * sum(w for _, w in items)
    for key, w in items:
        x -= w
        if x <= 0:
            return key
    return items[-1][0]


def route_item_status(answer_body, pool_names):
    """(parsed, None) if a Route item can get 3 distinct distractors, else (None, drop reason).

    parsed = (start, middles, goal, pool, {room count: distinct middles available}).
    """
    p = parse_route_option(answer_body)
    if p is None:
        return None, "unparseable_answer"
    start, middles, goal = p
    if not middles:
        return None, "direct_path_answer"
    if start == goal:
        return None, "start_equals_goal"
    # The item's own start and goal rooms are left out of the pool: a path that
    # passes back through them is visibly not the shortest (72% of
    # _optv3b distractors did, 8% of answers). reoption_route_v3 puts them back
    # at the answers' own rate.
    pool = [x for x in dict.fromkeys(list(middles) + [x for x in pool_names if x and x != "None"])
            if x not in (start, goal)]
    avail = _route_middles_available(pool, middles)
    if sum(avail.values()) < 3:
        return None, "too_few_rooms_for_distinct_options"
    return (start, middles, goal, pool, avail), None


def reoption_route_v3(answer_bodies, pools, rng, smoothing=5.0, fit_iters=30, fit_reps=4):
    """Options for a whole Route Planning file, with exchangeable hop counts.

    The v2 audit still found answers more often 2-hop (55% vs 46% of
    distractors) and sharing fewer middle rooms with the other options: the
    fitted option sets of reoption_route could not be realised by items with
    small room pools. Here each distractor's hop count is drawn from weights
    per room-pool size, fitted (Monte Carlo, fit_iters x fit_reps draws per
    item, fixed seed) so that within every pool size the realised distractor
    hop mix matches the answers' hop mix -- after the pool's limits on how many
    distinct shorter middles exist, which otherwise push small-pool items
    towards longer distractors; a second, file-level factor per hop count then
    brings the whole file's distractor hop mix to the answers' where small pools
    cannot. Each distractor's middle rooms are a uniform
    random ordered sample of the pool. Drops as route_item_status.
    Returns ([(options, answer) or (None, reason)], stats).
    """
    results = [None] * len(answer_bodies)
    parsed = {}
    for i, (a, pool_names) in enumerate(zip(answer_bodies, pools)):
        status, reason = route_item_status(a, pool_names)
        if status is None:
            results[i] = (None, reason)
        else:
            parsed[i] = status
    if not parsed:
        return results, {}
    hops = {i: len(p[1]) + 1 for i, p in parsed.items()}
    size = {i: len(p[3]) for i, p in parsed.items()}
    marginal = collections.Counter(hops.values())
    total = sum(marginal.values())
    by_pool = collections.defaultdict(collections.Counter)
    for i in parsed:
        by_pool[size[i]][hops[i]] += 1
    base = {q: {h: by_pool[q][h] + smoothing * marginal[h] / total for h in marginal if h - 1 <= q}
            for q in by_pool}
    factors = {q: {h: 1.0 for h in base[q]} for q in base}
    global_factor = {h: 1.0 for h in marginal}

    def draw_hops(i, r):
        remaining = dict(parsed[i][4])
        q, out = size[i], []
        for _ in range(3):
            allowed = {h: base[q][h] * factors[q][h] * global_factor[h]
                       for h in base[q] if remaining.get(h - 1, 0) > 0}
            if not allowed:  # the answers' hop counts are exhausted for this pool
                allowed = {k + 1: 1.0 for k, c in remaining.items() if c > 0}
            h = _weighted_choice(allowed, r)
            remaining[h - 1] -= 1
            out.append(h)
        return out

    sim = _random.Random(0)
    for _ in range(fit_iters):
        drawn = collections.defaultdict(collections.Counter)
        for i in parsed:
            for _ in range(fit_reps):
                drawn[size[i]].update(draw_hops(i, sim))
        for q, counter in drawn.items():
            got_total, want_total = sum(counter.values()), sum(by_pool[q].values())
            for h in base[q]:
                want, got = by_pool[q][h] / want_total, counter[h] / got_total
                if got > 0:
                    factors[q][h] = min(1e3, max(1e-3, factors[q][h] * (max(want, 1e-6) / got) ** 0.5))
                elif want > 0:
                    factors[q][h] = min(1e3, factors[q][h] * 2)

    # Small pools cannot always follow their own answers' mix; a file-level factor
    # per hop count lets the larger pools make up the difference.
    for _ in range(fit_iters):
        counter = collections.Counter()
        for i in parsed:
            for _ in range(fit_reps):
                counter.update(draw_hops(i, sim))
        got_total = sum(counter.values())
        for h in marginal:
            want, got = marginal[h] / total, counter[h] / got_total
            if got > 0:
                global_factor[h] = min(1e3, max(1e-3, global_factor[h] * (want / got) ** 0.5))

    def revisits(start, middles, goal):
        return any(m in (start, goal) for m in middles)

    revisit_rate = sum(revisits(*p[:3]) for p in parsed.values()) / len(parsed)
    answer_hops, distractor_hops, revisit_count = collections.Counter(), collections.Counter(), collections.Counter()
    for i, (start, middles, goal, pool, avail) in parsed.items():
        used, distractors = {answer_bodies[i]}, []
        for h in draw_hops(i, rng):
            text = None
            if rng.random() < revisit_rate:
                text = _draw_route_middle_revisit(pool, h - 1, start, goal, used, rng)
            if text is None:
                text = _draw_route_middle(pool, h - 1, start, goal, used, rng)
            distractors.append(text)
            distractor_hops[h] += 1
            revisit_count["distractors"] += revisits(*parse_route_option(text))
        answer_hops[hops[i]] += 1
        revisit_count["answers"] += revisits(start, middles, goal)
        results[i] = shuffled_letter_options(answer_bodies[i], distractors, rng)
    share = lambda c: {k: round(v / sum(c.values()), 3) for k, v in sorted(c.items())}
    return results, {"items": len(parsed), "answer_hop_share": share(answer_hops),
                     "distractor_hop_share": share(distractor_hops),
                     "revisit_start_or_goal_share": {
                         "answers": round(revisit_count["answers"] / len(parsed), 3),
                         "distractors": round(revisit_count["distractors"] / (3 * len(parsed)), 3)}}


def _draw_route_middle_revisit(pool, r, start, goal, used, rng):
    """A distinct r-room middle that passes through the start or the goal room once, or None."""
    if r < 1 or r - 1 > len(pool):
        return None
    for _ in range(50):
        middle = rng.sample(pool, r - 1)
        middle.insert(rng.randrange(r), rng.choice([start, goal]))
        text = route_option_text(start, middle, goal)
        if text not in used:
            used.add(text)
            return text
    return None


def spatial_name_features(name, target):
    """Attributes of an option name relative to the question's target object."""
    w, t = name.split(), target.split()
    return {
        "name": name,
        "size_given_target": (w[0], t[0]),
        "shape": w[-1],
        "material": w[-2],
        "has_color": len(w) == 4,
        "same_shape": w[-1] == t[-1],
        "same_material": w[-2] == t[-2],
        "same_color": len(w) == 4 and len(t) == 4 and w[1] == t[1],
    }


def reoption_spatial_v3(items, rng, iters=60, floor=1e-3, visible_bonus=3.0):
    """Options for a whole Spatial Measurement file, distractor attributes matched to the answers'.

    The v2 audit found answers more often "Small" (61% vs 49% of distractors)
    and never sharing the target's shape (distractors 2%): the closest object
    has a different attribute mix from the other objects of a scene; the first
    _optv3 audit also found answers concentrated on a few object names. Each
    distractor is now drawn without replacement from all other real objects of
    the scene, objects visible at the exit frame weighted visible_bonus times
    (so they stay the usual choice), with weights fitted by iterative
    proportional fitting over spatial_name_features (including the full name)
    so that across the file the distractors' attribute and name shares match the answers'. (Restricting candidates to
    the visible objects left too few to rebalance.) Synthetic names only when a
    scene has fewer than 3 other objects (spatial_distractors).

    items: dicts with answer, target, event, ranked (visible object names,
    closest first, answer removed) and scene_names.
    Returns ([(options, answer)], stats).
    """
    cands, feats, bases = [], [], []
    for it in items:
        banned = {it["answer"], it["target"], it["event"]}
        visible = set(it["ranked"])
        pool = [x for x in dict.fromkeys(it["scene_names"]) if x not in banned]
        cands.append(pool)
        feats.append([spatial_name_features(x, it["target"]) for x in pool])
        bases.append([visible_bonus if x in visible else 1.0 for x in pool])
    answer_share = collections.defaultdict(collections.Counter)
    for it in items:
        for f, v in spatial_name_features(it["answer"], it["target"]).items():
            answer_share[f][v] += 1 / len(items)
    factors = collections.defaultdict(dict)

    def weight(fc, b):
        w = b
        for f, v in fc.items():
            w *= factors[f].get(v, 1.0)
        return w

    for _ in range(iters):
        expected = collections.defaultdict(collections.Counter)
        for fs, bs in zip(feats, bases):
            if not fs:
                continue
            ws = [weight(fc, b) for fc, b in zip(fs, bs)]
            z, k = sum(ws), min(3, len(fs))
            for fc, w in zip(fs, ws):
                p = min(1.0, k * w / z)
                for f, v in fc.items():
                    expected[f][v] += p
        for f, counter in expected.items():
            tot = sum(counter.values())
            for v, c in counter.items():
                ratio = answer_share[f].get(v, 0.0) / (c / tot) if c > 0 else 1.0
                factors[f][v] = min(1 / floor, max(floor, factors[f].get(v, 1.0) * ratio ** 0.5))

    out, drawn, visible_drawn, n_drawn = [], collections.defaultdict(collections.Counter), 0, 0
    for it, pool, fs, bs in zip(items, cands, feats, bases):
        remaining, chosen = list(range(len(pool))), []
        while remaining and len(chosen) < 3:
            j = _weighted_choice({j: weight(fs[j], bs[j]) for j in remaining}, rng)
            remaining.remove(j)
            chosen.append(pool[j])
        distractors = spatial_distractors(chosen, it["answer"], [it["target"], it["event"]], it["scene_names"], rng)
        for x in distractors:
            n_drawn += 1
            visible_drawn += x in it["ranked"]
            for f, v in spatial_name_features(x, it["target"]).items():
                drawn[f][v] += 1
        out.append(shuffled_letter_options(it["answer"], distractors, rng))

    def share(counter):
        tot = sum(counter.values()) or 1
        return {str(k): round(v / tot, 3) for k, v in sorted(counter.items(), key=lambda kv: repr(kv[0]))}

    stats = {f: {"answers": {str(k): round(v, 3) for k, v in sorted(answer_share[f].items(), key=lambda kv: repr(kv[0]))},
                 "distractors": share(drawn[f])}
             for f in ("size_given_target", "same_shape", "same_material", "same_color")}
    stats["distractors_visible_at_exit_share"] = round(visible_drawn / max(1, n_drawn), 3)
    return out, stats


def reoption_ordering_v3(answer_bodies, rng, alpha=1.0):
    """Options for a whole Sequential Ordering file; distractors follow the answers' order distribution.

    Uniform distractors (v2) leave answer frequency as a cue when a file's
    answers are unevenly spread over the orders (first _optv3 audit: picking the
    order most often correct in training scored 0.30 on the eval file). Each
    distractor is drawn without replacement from the other orders with
    probability proportional to that order's count among the file's answers
    plus alpha. Returns [(options, answer)].
    """
    counts = collections.Counter(answer_bodies)
    out = []
    for answer in answer_bodies:
        k = len(answer.split(" -> "))
        pool = [" -> ".join(f"Video {i}" for i in p) for p in itertools.permutations(range(1, k + 1))]
        assert answer in pool, answer
        weights = {x: counts[x] + alpha for x in pool if x != answer}
        distractors = []
        for _ in range(3):
            x = _weighted_choice(weights, rng)
            del weights[x]
            distractors.append(x)
        out.append(shuffled_letter_options(answer, distractors, rng))
    return out
