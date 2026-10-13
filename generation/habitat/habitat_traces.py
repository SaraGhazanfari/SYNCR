"""
Chain-of-thought trace builders for the Habitat SYNCR tasks.

Traces are written from the same simulator-derived ground truth the answer is
derived from -- semantic instance ids, frame-level visibility ranges, and NavMesh
route chunks -- so they are programmatically correct rather than model-generated.
This mirrors what generation/clevrer/ already does for the CLEVRER tasks.

Each builder picks one of several presentation variants at random. Only the
surrounding wording/structure varies (labeled sections vs. flowing prose,
sentence order, phrasing) -- the underlying facts (numbers, object names,
video indices, the final answer) are identical across variants and computed
exactly once, upstream, the same way regardless of which variant renders them.
This exists so a model trained on this data can't key on "this fixed phrase
means task X" -- the previous version used one hardcoded template per task,
verbatim, for every example.

Used for SFT training data only. Generators emit these when run with
--split train; the benchmark (--split val) is unchanged.
"""

import random

from generation.habitat.base_habitat import get_sampled_frame_index
from generation.trace_style import GROUNDING_HEADERS, OBSERVATION_HEADERS, DEDUCTION_HEADERS, CONCLUSION_HEADERS, header


def _fmt_sec(frame, fps):
    return f"{frame / float(fps):.1f}s"


def _fmt_range(rng, fps):
    return f"{_fmt_sec(rng[0], fps)} to {_fmt_sec(rng[1], fps)}"


def _fmt_range_with_frames(rng, fps, total_frames, nframes=32):
    """
    Chain-of-frames: the timestamp alone is already precise (it doesn't
    depend on how many frames get sampled at train/inference time), but
    naming the actual sampled-frame indices as well makes the claim
    verifiable against exactly what the model sees, not just what second it
    happened at. total_frames is this specific video's own native frame
    count (Habitat routes vary in length, unlike CLEVRER's fixed 128), and
    is None when the caller doesn't have it on hand -- falls back to the
    timestamp-only string in that case rather than guessing.
    """
    time_str = _fmt_range(rng, fps)
    if total_frames is None:
        return time_str
    start_idx = get_sampled_frame_index(rng[0], total_frames, nframes) + 1
    end_idx = get_sampled_frame_index(rng[1], total_frames, nframes) + 1
    return f"{time_str} (sampled frames {start_idx}-{end_idx})"


def _fmt_video_list(indices):
    """[0, 2] -> 'Video 1 and Video 3'; [0,1,2] -> 'Video 1, Video 2 and Video 3'."""
    names = [f"Video {i + 1}" for i in sorted(indices)]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _plural(n, singular, plural=None):
    """These traces are training targets, so keep the grammar clean."""
    return singular if n == 1 else (plural or singular + "s")


# ---------------------------------------------------------------------------
# Object Re-identification  (Spatial Tracking)
# ---------------------------------------------------------------------------

def _reid_absent_v0(object_name):
    return (
        f"{header(GROUNDING_HEADERS)} The same physical object must be matched across two "
        "independent camera trajectories. Because viewpoint and lighting differ "
        "between the videos, the match is made on semantic instance identity "
        "rather than raw appearance.\n\n"
        f"{header(OBSERVATION_HEADERS)}\n\n"
        f"- Scanning Video 2 for the same instance, it is never visible: the "
        f"'{object_name}' does not fall inside that camera's trajectory.\n\n"
        f"{header(DEDUCTION_HEADERS)} Since the instance has no visible window in Video 2, "
        "no timestamp range can be correct.\n\n"
        f"{header(CONCLUSION_HEADERS)} Therefore, the '{object_name}' does not appear in Video 2."
    )


def _reid_absent_v1(object_name):
    return (
        f"The '{object_name}' has to be tracked by semantic instance identity, not "
        f"raw appearance, since the two videos differ in viewpoint and lighting. "
        f"Checking every frame of Video 2's trajectory for this instance turns up "
        f"nothing -- it never enters that camera's path. With no visible window at "
        f"all, there is no timestamp range that could be correct, so the '{object_name}' "
        f"simply does not appear in Video 2."
    )


REID_ABSENT_VARIANTS = [_reid_absent_v0, _reid_absent_v1]


def _reid_present_v0(object_name, ref_range, ordered, answer_range, fps,
                      ref_total_frames=None, target_total_frames=None):
    lines = [
        f"{header(GROUNDING_HEADERS)} The same physical object must be matched across two "
        "independent camera trajectories. Because viewpoint and lighting differ "
        "between the videos, the match is made on semantic instance identity "
        "rather than raw appearance.",
        "",
        header(OBSERVATION_HEADERS),
        "",
    ]
    if ref_range:
        lines.append(
            f"- In Video 1, the '{object_name}' is continuously visible from "
            f"{_fmt_range_with_frames(ref_range, fps, ref_total_frames)}."
        )
    window_strs = "; ".join(_fmt_range_with_frames(r, fps, target_total_frames) for r in ordered)
    lines.append(
        f"- Scanning Video 2 for the same instance, it is visible during: {window_strs}."
    )
    lines += ["", ""]
    if len(ordered) == 1:
        lines.append(
            f"{header(DEDUCTION_HEADERS)} The instance has a single continuous visibility window "
            "in Video 2, so that window is its first appearance."
        )
    else:
        lines.append(
            f"{header(DEDUCTION_HEADERS)} The instance appears {len(ordered)} separate times in "
            f"Video 2. The question asks for the FIRST appearance, so the earliest "
            f"of these windows is the one to report, not the longest or the clearest."
        )
    lines += [
        "",
        f"{header(CONCLUSION_HEADERS)} Therefore, the first appearance of the '{object_name}' "
        f"in Video 2 is {_fmt_range_with_frames(answer_range, fps, target_total_frames)}.",
    ]
    return "\n".join(lines)


def _reid_present_v1(object_name, ref_range, ordered, answer_range, fps,
                      ref_total_frames=None, target_total_frames=None):
    ref_clause = (
        f"Video 1 shows the '{object_name}' from "
        f"{_fmt_range_with_frames(ref_range, fps, ref_total_frames)}. "
        if ref_range else ""
    )
    window_strs = "; ".join(_fmt_range_with_frames(r, fps, target_total_frames) for r in ordered)
    if len(ordered) == 1:
        deduction = (
            f"it only has one continuous window in Video 2, so that single window "
            f"is necessarily its first appearance there"
        )
    else:
        deduction = (
            f"it recurs {len(ordered)} separate times in Video 2, and since the "
            f"question asks specifically for the FIRST appearance rather than the "
            f"longest or clearest, the earliest window is the one that answers it"
        )
    return (
        f"{ref_clause}Matching this same physical object into Video 2 has to rely "
        f"on semantic instance identity rather than appearance, since the two cameras "
        f"differ in viewpoint and lighting. Tracing that instance through Video 2's "
        f"footage, it is visible during: {window_strs}. Because {deduction}, the "
        f"answer is {_fmt_range_with_frames(answer_range, fps, target_total_frames)}."
    )


def _reid_present_v2(object_name, ref_range, ordered, answer_range, fps,
                      ref_total_frames=None, target_total_frames=None):
    window_strs = ", ".join(_fmt_range_with_frames(r, fps, target_total_frames) for r in ordered)
    parts = [f"**Task:** Find the FIRST time the '{object_name}' from Video 1 reappears in Video 2."]
    if ref_range:
        parts.append(
            f"**Given:** Video 1 window = {_fmt_range_with_frames(ref_range, fps, ref_total_frames)}."
        )
    parts.append(f"**Found in Video 2:** {window_strs}.")
    if len(ordered) > 1:
        parts.append(
            f"**Note:** {len(ordered)} windows were found; only the earliest counts "
            f"as the first appearance."
        )
    parts.append(f"**Answer:** {_fmt_range_with_frames(answer_range, fps, target_total_frames)}.")
    return "\n".join(parts)


REID_PRESENT_VARIANTS = [_reid_present_v0, _reid_present_v1, _reid_present_v2]


def build_reid_trace(object_name, ref_range, target_ranges, answer_range, fps,
                      ref_total_frames=None, target_total_frames=None, variant_fn=None):
    """
    Args:
        object_name:        cleaned object name, e.g. 'kitchen counter'
        ref_range:           (start, end) frames -- the window quoted in the question
        target_ranges:       all (start, end) windows in the target video
        answer_range:        the correct window, or None if the object never appears
        fps:                 frames per second of the rendered video
        ref_total_frames:    Video 1's native frame count, for chain-of-frames
                              sampled-frame references. None omits them (falls
                              back to timestamp-only) rather than guessing --
                              this is also how a caller builds the plain-CoT
                              counterpart of a cof trace (pass None here and
                              in target_total_frames on the second call).
        target_total_frames: Video 2's native frame count, same purpose.
        variant_fn:          pins the presentation style (one of
                              REID_ABSENT_VARIANTS or REID_PRESENT_VARIANTS,
                              whichever the caller already knows applies) so a
                              cof/cot pair for the same example reads
                              identically apart from the frame references.
                              None picks randomly (unchanged default).
    """
    if not target_ranges:
        variant = variant_fn or random.choice(REID_ABSENT_VARIANTS)
        return variant(object_name)

    ordered = sorted(target_ranges, key=lambda r: r[0])
    variant = variant_fn or random.choice(REID_PRESENT_VARIANTS)
    return variant(object_name, ref_range, ordered, answer_range, fps,
                    ref_total_frames=ref_total_frames, target_total_frames=target_total_frames)


# ---------------------------------------------------------------------------
# Object Counting  (Holistic Synthesis)
# ---------------------------------------------------------------------------

def _per_video_lines(object_name, instance_labels, instance_to_videos, instance_frame_info=None):
    """
    instance_frame_info: {instance_id: {video_idx: (raw_frame, video_total_frames)}},
    the frame where that instance is most clearly visible in that video (peak pixel
    occupancy). Chain-of-frames grounding: None skips it (falls back to the
    video-level-only line) rather than guessing.
    """
    videos = sorted({v for vids in instance_to_videos.values() for v in vids})
    lines = []
    for v in videos:
        seen_ids = [i for i, vids in sorted(instance_to_videos.items()) if v in vids]
        if not seen_ids:
            continue
        seen = []
        for i in seen_ids:
            label = instance_labels[i]
            frame_data = (instance_frame_info or {}).get(i, {}).get(v)
            if frame_data is not None:
                raw_frame, video_total_frames = frame_data
                sampled_frame = get_sampled_frame_index(raw_frame, video_total_frames) + 1
                label = f"{label} (clearest at Frame {sampled_frame})"
            seen.append(label)
        lines.append(
            f"Video {v + 1} shows {object_name} "
            f"{_plural(len(seen), 'instance')}: {', '.join(seen)}."
        )
    return lines


def _counting_v0(object_name, instance_labels, instance_to_videos, total_count, instance_frame_info=None):
    lines = [
        f"{header(GROUNDING_HEADERS)} Each video explores a different part of the scene, and the "
        f"same '{object_name}' can be seen from more than one trajectory. The count "
        f"is over distinct instances, so repeated views of one instance must not be "
        f"counted twice.",
        "",
        header(OBSERVATION_HEADERS),
        "",
    ]
    lines += [f"- {l}" for l in _per_video_lines(object_name, instance_labels, instance_to_videos, instance_frame_info)]
    lines += ["", ""]

    total_str = f"{total_count} distinct {_plural(total_count, 'instance')}"
    repeated = {i: vids for i, vids in instance_to_videos.items() if len(vids) > 1}
    if repeated:
        parts = [
            f"{instance_labels[i]} appears in {_fmt_video_list(vids)}"
            for i, vids in sorted(repeated.items())
        ]
        lines.append(
            f"{header(DEDUCTION_HEADERS)} " + "; ".join(parts) + ". These are repeated observations "
            "of the same physical object rather than additional ones, so they collapse "
            f"to a single instance each. Taking the union across all videos gives {total_str}."
        )
    else:
        lines.append(
            f"{header(DEDUCTION_HEADERS)} No instance is seen in more than one video, so no "
            "deduplication is needed. The union across all videos is simply the total: "
            f"{total_str}."
        )
    lines += [
        "",
        f"{header(CONCLUSION_HEADERS)} Therefore, there {_plural(total_count, 'is', 'are')} "
        f"{total_str} of '{object_name}' in this scene.",
    ]
    return "\n".join(lines)


def _counting_v1(object_name, instance_labels, instance_to_videos, total_count, instance_frame_info=None):
    per_video = _per_video_lines(object_name, instance_labels, instance_to_videos, instance_frame_info)
    obs = " ".join(per_video)
    total_str = f"{total_count} distinct {_plural(total_count, 'instance')}"
    repeated = {i: vids for i, vids in instance_to_videos.items() if len(vids) > 1}
    if repeated:
        parts = [
            f"{instance_labels[i]} shows up in both {_fmt_video_list(vids)}"
            for i, vids in sorted(repeated.items())
        ]
        dedup = (
            "; ".join(parts) + " -- these are the same physical object seen twice, "
            "not two different objects, so each collapses to one when counting."
        )
    else:
        dedup = "no instance recurs across videos, so nothing needs to be collapsed."
    return (
        f"Counting '{object_name}' requires aggregating across every video, since "
        f"each one only explores part of the scene: {obs} The key is avoiding "
        f"double-counting a single physical object seen from multiple trajectories. "
        f"{dedup} Taking the union across all videos, there {_plural(total_count, 'is', 'are')} "
        f"{total_str}."
    )


def _counting_v2(object_name, instance_labels, instance_to_videos, total_count, instance_frame_info=None):
    lines = [f"**Per-video evidence for '{object_name}':**"]
    for l in _per_video_lines(object_name, instance_labels, instance_to_videos, instance_frame_info):
        lines.append(f"  * {l}")
    repeated = {i: vids for i, vids in instance_to_videos.items() if len(vids) > 1}
    total_str = f"{total_count} distinct {_plural(total_count, 'instance')}"
    if repeated:
        parts = [f"{instance_labels[i]} in {_fmt_video_list(vids)}" for i, vids in sorted(repeated.items())]
        lines.append(
            f"**Overlap check:** {'; '.join(parts)} -- same object, don't double-count."
        )
    else:
        lines.append("**Overlap check:** none -- every instance appears in exactly one video.")
    lines.append(f"**Deduplicated total:** {total_str}.")
    return "\n".join(lines)


COUNTING_VARIANTS = [_counting_v0, _counting_v1, _counting_v2]


def build_counting_trace(object_name, instance_labels, instance_to_videos, total_count,
                          instance_frame_info=None, variant_fn=None):
    """
    Args:
        object_name:         cleaned object name
        instance_labels:     {instance_id: human label, e.g. '#2'}
        instance_to_videos:  {instance_id: [video indices it is visible in]}
        total_count:         ground-truth number of distinct instances
        instance_frame_info: {instance_id: {video_idx: (raw_frame, video_total_frames)}}
                              -- the frame of peak visibility, for chain-of-frames
                              grounding. None omits it (falls back to video-level-only
                              lines) rather than guessing -- also how a caller builds
                              the plain-CoT counterpart of a cof trace.
        variant_fn:           pins the presentation style so a cof/cot pair for
                              the same example reads identically apart from the
                              frame references. None picks randomly (unchanged
                              default).
    """
    variant = variant_fn or random.choice(COUNTING_VARIANTS)
    return variant(object_name, instance_labels, instance_to_videos, total_count,
                    instance_frame_info=instance_frame_info)


# ---------------------------------------------------------------------------
# Route Planning  (Holistic Synthesis)
# ---------------------------------------------------------------------------

def _route_full_path_str(start_name, sub_path_names, end_name):
    return (
        f"Region 1 <{start_name}> -> "
        + (" -> ".join(sub_path_names[1:-1]) + " -> " if len(sub_path_names) > 2 else "")
        + f"Region 2 <{end_name}>"
    )


def _route_v0(segments, sub_path_names, start_name, end_name):
    lines = [
        f"{header(GROUNDING_HEADERS)} No single video covers the whole route. Each clip walks one "
        "contiguous stretch of the environment, and consecutive clips share the region "
        "where one ends and the next begins. The layout has to be assembled from those "
        "overlapping segments before a path can be read off it.",
        "",
        header(OBSERVATION_HEADERS),
        "",
    ]
    for video_idx, names in segments:
        if names:
            lines.append(f"- Video {video_idx + 1} traverses: {' -> '.join(names)}.")
    lines += ["", ""]
    lines.append(
        f"{header(DEDUCTION_HEADERS)} Chaining the segments on their shared regions reconstructs the "
        f"connectivity of the scene. Reading the route from Region 1 <{start_name}> to "
        f"Region 2 <{end_name}> off that reconstructed layout gives "
        f"{' -> '.join(sub_path_names)}."
    )
    lines += [
        "",
        f"{header(CONCLUSION_HEADERS)} Therefore, the shortest path is "
        f"{_route_full_path_str(start_name, sub_path_names, end_name)}.",
    ]
    return "\n".join(lines)


def _route_v1(segments, sub_path_names, start_name, end_name):
    seg_strs = [f"Video {i + 1} covers {' -> '.join(names)}" for i, names in segments if names]
    obs = "; ".join(seg_strs)
    return (
        f"Since the environment is only ever shown in disconnected clips, getting from "
        f"Region 1 <{start_name}> to Region 2 <{end_name}> means stitching the clips "
        f"together wherever one ends where another begins: {obs}. Following that chain "
        f"of shared regions end to end traces out "
        f"{_route_full_path_str(start_name, sub_path_names, end_name)}, which is the "
        f"shortest path between them."
    )


def _route_v2(segments, sub_path_names, start_name, end_name):
    lines = ["**Per-video coverage:**"]
    for video_idx, names in segments:
        if names:
            lines.append(f"  * Video {video_idx + 1}: {' -> '.join(names)}")
    lines.append(
        f"**Stitching:** consecutive clips overlap at the region where one ends and "
        f"the next starts, so the full layout can be reconstructed by chaining them."
    )
    lines.append(
        f"**Resulting path from Region 1 <{start_name}> to Region 2 <{end_name}>:** "
        f"{_route_full_path_str(start_name, sub_path_names, end_name)}"
    )
    return "\n".join(lines)


ROUTE_VARIANTS = [_route_v0, _route_v1, _route_v2]


def build_route_trace(segments, sub_path_names, start_name, end_name, variant_fn=None):
    """
    Args:
        segments:       [(video_index, [region names traversed]), ...] one per camera.
                        Region names may already carry a chain-of-frames "(Frame N)"
                        annotation (see habitat_holistic_aggregation.py) or not --
                        that's how a caller builds the cof/cot pair: pass the
                        frame-annotated segments for cof, plain ones for cot.
        sub_path_names: region names along the correct path, in order
        start_name:     region name of the start node
        end_name:       region name of the end node
        variant_fn:     pins the presentation style so a cof/cot pair for the
                        same example reads identically apart from the region
                        names. None picks randomly (unchanged default).
    """
    variant = variant_fn or random.choice(ROUTE_VARIANTS)
    return variant(segments, sub_path_names, start_name, end_name)
