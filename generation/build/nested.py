"""
Balanced selection and nested ordering for the `_optv3*` files.

nested_order (eval): lines 1..200 and 1..500 each hold the target counts per
answer bucket (equal, or in proportion to per-bucket weights, e.g. Numerical's
tie bucket at 1/5), as far as the pool allows; sub-buckets such as Numerical's
winner video are interleaved inside each bucket, and scenes are spread by
stratified shuffling.

balanced_select (training): bucket sizes in the same proportions -- the
scarcest bucket relative to its weight sets the size and the others are
subsampled; nothing is duplicated.
"""
import collections

from generation.build.common import stratified_shuffle


def _bucket_queues(n, bucket_of, scene_of, rng, sub_of=None):
    """{bucket: indices}, scene-stratified, sub-buckets interleaved round-robin."""
    members = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in range(n):
        members[bucket_of(i)][sub_of(i) if sub_of else None].append(i)
    queues = {}
    for b in sorted(members, key=repr):
        lists = []
        for s in sorted(members[b], key=repr):
            idx = members[b][s]
            perm = stratified_shuffle(len(idx), lambda j: scene_of(idx[j]), rng)
            lists.append([idx[j] for j in perm])
        rng.shuffle(lists)
        merged, depth = [], 0
        while any(depth < len(lst) for lst in lists):
            for lst in lists:
                if depth < len(lst):
                    merged.append(lst[depth])
            depth += 1
        queues[b] = merged
    return queues


def weighted_quota(total, keys, weights=None):
    """Integer counts per key summing to total, proportional to weights (largest remainder)."""
    w = {k: float((weights or {}).get(k, 1.0)) for k in keys}
    z = sum(w.values())
    raw = {k: total * w[k] / z for k in keys}
    quota = {k: int(raw[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(raw[k] - quota[k]), repr(k)))[:total - sum(quota.values())]:
        quota[k] += 1
    return quota


def bucket_counts(indices, bucket_of, sub_of=None):
    out = dict(sorted(collections.Counter(str(bucket_of(i)) for i in indices).items()))
    if sub_of:
        out["by_sub"] = dict(sorted(collections.Counter(f"{bucket_of(i)}|{sub_of(i)}" for i in indices).items()))
    return out


def nested_order(n, bucket_of, scene_of, rng, sizes=(200, 500), sub_of=None, weights=None):
    """Indices of the items for lines 1..sizes[-1], in order.

    Each prefix sizes[k] holds weighted_quota(sizes[k]) items per bucket
    (cumulative); a bucket that runs short is topped up from the buckets with
    the most items left, and the shortfall shows in the report. Within each tier
    (between consecutive sizes) items are scene-stratified.
    Returns (order, {"first<size>": bucket counts}).
    """
    queues = _bucket_queues(n, bucket_of, scene_of, rng, sub_of)
    buckets = list(queues)
    taken = {b: 0 for b in buckets}
    order, report, prev = [], {}, 0
    for size in sizes:
        quota = weighted_quota(size, buckets, weights)
        tier = []
        for b in buckets:
            want = max(0, quota[b] - taken[b])
            got = queues[b][taken[b]:taken[b] + want]
            tier.extend(got)
            taken[b] += len(got)
        short = (size - prev) - len(tier)
        while short > 0 and any(taken[b] < len(queues[b]) for b in buckets):
            b = max(buckets, key=lambda k: len(queues[k]) - taken[k])
            tier.append(queues[b][taken[b]])
            taken[b] += 1
            short -= 1
        perm = stratified_shuffle(len(tier), lambda j: scene_of(tier[j]), rng)
        order.extend(tier[j] for j in perm)
        report[f"first{size}"] = bucket_counts(order, bucket_of, sub_of)
        prev = size
    return order, report


def balanced_select(n, bucket_of, scene_of, rng, total, sub_of=None, weights=None):
    """Buckets in weight proportion (equal by default), at most `total` items, scene-stratified order.

    Returns (indices, report) with the per-bucket sizes, what each bucket had
    available, and the final size.
    """
    queues = _bucket_queues(n, bucket_of, scene_of, rng, sub_of)
    w = {b: float((weights or {}).get(b, 1.0)) for b in queues}
    size = min(total, int(min(len(q) / w[b] for b, q in queues.items()) * sum(w.values())))
    while True:
        quota = weighted_quota(size, list(queues), w)
        if all(quota[b] <= len(q) for b, q in queues.items()):
            break
        size -= 1
    chosen = [i for b in queues for i in queues[b][:quota[b]]]
    perm = stratified_shuffle(len(chosen), lambda j: scene_of(chosen[j]), rng)
    chosen = [chosen[j] for j in perm]
    report = {"size": len(chosen), "per_bucket": {str(b): quota[b] for b in queues},
              "available_per_bucket": {str(b): len(q) for b, q in queues.items()},
              "buckets": bucket_counts(chosen, bucket_of, sub_of)}
    return chosen, report
