'''
>> Cross-Video Temporal Alignment & Synchronization 

    The ability to recognize overlapping events, 
    aligning different camera angles of the exact same moment, 
    or sequencing disjointed clips into a cohesive timeline.
    
    Run: python -m generation.clevrer.clvr_temporal_alignment
'''

from utils import DATA_ROOT
import os
import json
import math
import random
import argparse

from utils import write_jsonl_file
from generation.clevrer.base_clevrer import *
from generation.clevrer.utils import *
from generation.trace_style import GROUNDING_HEADERS, OBSERVATION_HEADERS, DEDUCTION_HEADERS, CONCLUSION_HEADERS, header
import itertools

CLVR_FPS = 25.0


def _find_segment(frame, segment_frame_ranges):
    """
    Which chronological segment (index into segment_frame_ranges) a raw
    simulation frame falls into. Segments are separated by real time gaps
    (gap_time in get_seq_order_QA), so a frame can legitimately fall between
    two segments' ranges -- clamp to the nearest segment in that case rather
    than raising, since callers use this for a best-effort chronological
    bucketing of the physics event, not a hard boundary check.
    """
    for p, (start, end) in enumerate(segment_frame_ranges):
        if start <= frame <= end:
            return p
    return min(
        range(len(segment_frame_ranges)),
        key=lambda p: min(
            abs(frame - segment_frame_ranges[p][0]),
            abs(frame - segment_frame_ranges[p][1]),
        ),
    )


def is_permanence_sufficient(ann_dict, segment_frame_ranges):
    """
    segment_frame_ranges: list of (start_frame, end_frame) raw simulation
    frame boundaries, one per chronological segment, in chronological order.
    These must be the ACTUAL rendered clip boundaries (from
    get_seq_order_QA's variable-length segments), not an assumed uniform
    total_frames // k split -- segments can differ in length by up to ~70%
    of each other (min_length plus a randomly distributed remainder), so a
    uniform-split assumption can attribute an enter/exit/collision event to
    the wrong clip entirely.
    """
    entrance_dict = get_entrance_and_exit_frames(ann_dict)
    k = len(segment_frame_ranges)
    total_frames = segment_frame_ranges[-1][1] + 1

    # Dictionary to hold the state signature for each chronological part
    clip_signatures = {p: [] for p in range(k)}

    for obj, data in entrance_dict.items():
        enter_frame = data['enter']
        exit_frame = data['exit'] if data['exit'] is not None else total_frames

        for p in range(k):
            part_start, part_end = segment_frame_ranges[p]

            # Record the object's state in this specific clip
            if part_start <= enter_frame <= part_end:
                state = "enters"
            elif enter_frame < part_start and exit_frame > part_end:
                state = "exists_throughout"
            elif data['exit'] is not None and part_start <= exit_frame <= part_end:
                state = "exits"
            else:
                state = "absent"
                
            clip_signatures[p].append(f"Obj{obj}:{state}")
            
    # Convert signatures to tuples (so they can be hashed) and put them in a set
    unique_signatures = set([tuple(sig) for sig in clip_signatures.values()])
    
    # If the number of unique signatures equals the number of clips, it can be sorted perfectly!
    is_sufficient = len(unique_signatures) == k
    
    return is_sufficient


def is_collision_sufficient(ann_dict, segment_frame_ranges):
    """
    Evaluates if the discrete physics events are sufficient to uniquely
    identify and sort all k video clips. segment_frame_ranges: see
    is_permanence_sufficient's docstring -- must be the actual rendered clip
    boundaries, not a uniform total_frames // k assumption.
    """
    entrance_dict = get_entrance_and_exit_frames(ann_dict)
    collision_dict = get_collision_frame_dict(ann_dict)
    movement_dict = get_stationary_moving_status(ann_dict)

    k = len(segment_frame_ranges)

    # Initialize an empty signature for each chronological part
    signatures = {p: {
        'enters': [],
        'exits': [],
        'starts_moving': [],
        'stops_moving': [],
        'collisions': []
    } for p in range(k)}

    # 1. Map Entrances and Exits
    for obj, data in entrance_dict.items():
        if data['enter'] is not None and data['enter'] > 0:
            p = _find_segment(data['enter'], segment_frame_ranges)
            signatures[p]['enters'].append(obj)

        if data.get('exit') is not None:
            p = _find_segment(data['exit'], segment_frame_ranges)
            signatures[p]['exits'].append(obj)

    # 2. Map Movement Changes
    for obj, data in movement_dict.items():
        if data.get('moving') is not None and data['moving'] > 0:
            p = _find_segment(data['moving'], segment_frame_ranges)
            signatures[p]['starts_moving'].append(obj)

        if data.get('stationary') is not None and data['stationary'] > 0:
            p = _find_segment(data['stationary'], segment_frame_ranges)
            signatures[p]['stops_moving'].append(obj)

    # 3. Map Collisions (Handles both list and dict formats)
    for frame, event_data in collision_dict.items():
        p = _find_segment(frame, segment_frame_ranges)
        
        # Safely extract object IDs depending on how your dict is formatted
        if isinstance(event_data, dict) and 'object_ids' in event_data:
            obj_pair = tuple(sorted(event_data['object_ids']))
        elif isinstance(event_data, list):
            obj_pair = tuple(sorted(event_data))
        else:
            continue
            
        signatures[p]['collisions'].append(obj_pair)
        
    # 4. Hash the signatures to check for uniqueness
    unique_signatures = set()
    for p in range(k):
        # We sort the lists inside the signature and convert them to tuples
        # so they can be hashed and compared inside a Python set.
        sig_tuple = (
            tuple(sorted(signatures[p]['enters'])),
            tuple(sorted(signatures[p]['exits'])),
            tuple(sorted(signatures[p]['starts_moving'])),
            tuple(sorted(signatures[p]['stops_moving'])),
            tuple(sorted(signatures[p]['collisions']))
        )
        unique_signatures.add(sig_tuple)
        
    # If we have exactly k unique signatures, every clip is distinct!
    return len(unique_signatures) == k

def extract_permanence_process_gt(ann_dict, presented_parts, segment_frame_ranges):
    """GRPO process reward ground truth for the 'permanence' reasoning_type:
    the same per-object enter/exit timestamps build_permanence_trace states
    in its OBSERVATION text, as seconds local to each clip (matches the
    use_timestamps=True variant regardless of which reasoning variant
    actually lands in SFT training data -- see the GRPO reward's numeric-
    tolerance design for why timestamps, not frame indices).

    Mirrors build_permanence_trace's branch conditions exactly (same
    entrance_and_exit_dict, same three-way enter/exit case split) so the
    facts checked here are facts a correct trace would actually state --
    the "starts from initial frame" / "present throughout" branches assert
    no specific number, so those are skipped here too. Keep in sync with
    build_permanence_trace if that function's branching changes.
    """
    entrance_and_exit_dict = get_entrance_and_exit_frames(ann_dict)
    k = len(segment_frame_ranges)
    total_frames = segment_frame_ranges[-1][1] + 1
    part_to_video_idx = {p: presented_parts.index(p) + 1 for p in range(k)}

    facts = {}
    for obj, data in entrance_and_exit_dict.items():
        enter_frame_global = data['enter']
        exit_frame_global = data['exit'] if data['exit'] is not None else total_frames

        for p in range(k):
            part_start, part_end = segment_frame_ranges[p]
            video_id = part_to_video_idx[p]

            if part_start <= enter_frame_global <= part_end and enter_frame_global != 0:
                ts = round((enter_frame_global - part_start) / CLVR_FPS, 1)
                facts[f"obj{obj}_video{video_id}_enter_sec"] = ts
            elif data['exit'] is not None and part_start <= exit_frame_global <= part_end:
                ts = round((exit_frame_global - part_start) / CLVR_FPS, 1)
                facts[f"obj{obj}_video{video_id}_exit_sec"] = ts
    return facts


def extract_collision_process_gt(ann_dict, presented_parts, segment_frame_ranges):
    """GRPO process reward ground truth for the 'collision' reasoning_type:
    the timestamp of the one collision build_collision_trace actually
    describes (earliest frame in collision_dict -- that function processes
    only sorted_frames[0] before breaking, so this mirrors that same
    single-event selection, not every collision in is_collision_sufficient's
    broader signature check).
    """
    collision_dict = get_collision_frame_dict(ann_dict)
    k = len(segment_frame_ranges)
    part_to_video = {p: presented_parts.index(p) + 1 for p in range(k)}

    sorted_frames = sorted(collision_dict.keys())
    if not sorted_frames:
        return {}

    frame = sorted_frames[0]
    chrono_part = _find_segment(frame, segment_frame_ranges)
    collision_video = part_to_video[chrono_part]
    part_start, _ = segment_frame_ranges[chrono_part]
    ts = round((frame - part_start) / CLVR_FPS, 1)
    return {f"video_{collision_video}_collision_sec": ts}


def _permanence_v0(obs_text, correct_order):
    return "\n\n".join([
        f"{header(GROUNDING_HEADERS)} To determine the chronological sequence, we track the lifecycles, entrances, and exits of the objects across specific frames.",
        header(OBSERVATION_HEADERS),
        obs_text,
        f"{header(DEDUCTION_HEADERS)} Because an object cannot be continuously present in a scene before the moment it actually enters or after it exits, we align the lifecycles chronologically.",
        f"{header(CONCLUSION_HEADERS)} Therefore, the correct chronological order is **{correct_order}**.",
    ])


def _permanence_v1(obs_text, correct_order):
    return (
        f"An object's presence in a clip is bounded by when it enters and when it exits -- "
        f"it cannot appear in the scene before that entrance or persist in it after that exit. "
        f"{obs_text} Ordering the clips so each one's entrance/exit events are consistent with "
        f"that constraint gives {correct_order}."
    )


def _permanence_v2(obs_text, correct_order):
    return "\n".join([
        "**Entrance/exit evidence:**",
        f"  {obs_text}",
        "**Constraint:** a clip showing an object already present must come after the clip showing it enter, never before.",
        f"**Order:** {correct_order}",
    ])


PERMANENCE_VARIANTS = [_permanence_v0, _permanence_v1, _permanence_v2]


def build_permanence_trace(ann_dict, presented_parts, segment_frame_ranges, use_frames=True,
                            use_timestamps=False, variant_fn=None):
    """
    segment_frame_ranges: see is_permanence_sufficient's docstring -- the
    actual rendered clip boundaries, not a uniform total_frames // k
    assumption (which previously misattributed enter/exit events to the
    wrong clip and reported a "local frame" number that could exceed the
    clip's own length). local_enter/local_exit are additionally remapped
    via get_sampled_frame_index to the frame the model actually samples
    (nframes=32 per clip), not just the raw simulation frame within the
    clip -- chain-of-frames grounding, same verified formula as Kinematic
    Comparison and Object ReID.

    use_frames: when False (and use_timestamps is also False), omits the
    "at Frame N" clause entirely (falls back to qualitative-only phrasing)
    -- the plain-CoT counterpart to the chain-of-frames version.
    use_timestamps: emits "at {sec}s" instead of "at Frame N" -- a
    sampling-density-independent alternative to use_frames. get_sampled_frame_index
    ties "Frame N" to a specific nframes value (TRAIN_NFRAMES=8 at training
    time); eval-time sampling picks its own frame count per video via
    qwen_vl_utils' smart_nframes (duration/fps-driven, essentially never 8),
    so a trained "Frame N" reference doesn't correspond to anything the
    model actually sees at inference. A timestamp (raw frame / CLVR_FPS) is
    a property of the video itself, not of any particular frame-sampling
    scheme, so it stays meaningful regardless of how many frames get
    sampled at eval time. Takes priority over use_frames if both are True
    (callers should pass exactly one).
    variant_fn: pins the presentation style (one of PERMANENCE_VARIANTS) so
    a cof/cot/cotime triple for the same example uses the identical wording
    aside from the frame/timestamp reference; None picks randomly (unchanged
    default behavior for any caller that doesn't need a matched set).
    """
    entrance_and_exit_dict = get_entrance_and_exit_frames(ann_dict)
    k = len(segment_frame_ranges)
    total_frames = segment_frame_ranges[-1][1] + 1
    part_to_video_idx = {p: presented_parts.index(p) + 1 for p in range(k)}
    correct_order = " -> ".join([f"Video {part_to_video_idx[p]}" for p in range(k)])

    obs_list = []
    for obj, data in entrance_and_exit_dict.items():
        obj_name = get_object_name(ann_dict, obj)
        enter_frame_global = data['enter']
        exit_frame_global = data['exit'] if data['exit'] is not None else total_frames

        for p in range(k):
            part_start, part_end = segment_frame_ranges[p]
            clip_frame_count = part_end - part_start + 1
            video_id = part_to_video_idx[p]

            if part_start <= enter_frame_global <= part_end:
                local_enter = enter_frame_global - part_start
                if enter_frame_global == 0:
                    obs_list.append(f"In **Video {video_id}**, the {obj_name} starts in the scene from initial frame.")
                elif use_timestamps:
                    ts = round(local_enter / CLVR_FPS, 1)
                    obs_list.append(f"In **Video {video_id}**, the {obj_name} enters the scene at {ts}s.")
                elif use_frames:
                    sampled_frame = get_sampled_frame_index(local_enter, clip_frame_count) + 1
                    obs_list.append(f"In **Video {video_id}**, the {obj_name} enters the scene at Frame {sampled_frame}.")
                else:
                    obs_list.append(f"In **Video {video_id}**, the {obj_name} enters partway through the clip.")
            elif enter_frame_global < part_start and exit_frame_global > part_end:
                obs_list.append(f"In **Video {video_id}**, the {obj_name} is already present throughout the entire clip.")
            elif data['exit'] is not None and part_start <= exit_frame_global <= part_end:
                local_exit = exit_frame_global - part_start
                if use_timestamps:
                    ts = round(local_exit / CLVR_FPS, 1)
                    obs_list.append(f"In **Video {video_id}**, the {obj_name} exits the scene at {ts}s.")
                elif use_frames:
                    sampled_frame = get_sampled_frame_index(local_exit, clip_frame_count) + 1
                    obs_list.append(f"In **Video {video_id}**, the {obj_name} exits the scene at Frame {sampled_frame}.")
                else:
                    obs_list.append(f"In **Video {video_id}**, the {obj_name} exits partway through the clip.")

    obs_text = " ".join(obs_list)
    variant = variant_fn or random.choice(PERMANENCE_VARIANTS)
    return variant(obs_text, correct_order)


def _collision_v0(obs_text, correct_order):
    return "\n\n".join([
        f"{header(GROUNDING_HEADERS)} To determine the chronological sequence, we track the kinematic state changes and trajectory-altering collisions.",
        header(OBSERVATION_HEADERS),
        obs_text,
        f"{header(DEDUCTION_HEADERS)} Because causality dictates an object must follow its original path before an impact, and its altered path after an impact, the collision video acts as the chronological bridge.",
        f"{header(CONCLUSION_HEADERS)} Therefore, the correct chronological order is **{correct_order}**.",
    ])


def _collision_v1(obs_text, correct_order):
    return (
        f"A collision splits an object's trajectory into a before and an after: whichever clip "
        f"shows the pre-collision path has to come first, the collision itself is the pivot, and "
        f"the post-collision path comes after. {obs_text} That causal ordering fixes the sequence "
        f"as {correct_order}."
    )


def _collision_v2(obs_text, correct_order):
    return "\n".join([
        "**Collision evidence:**",
        f"  {obs_text}",
        "**Causal rule:** pre-impact trajectory precedes the collision, post-impact trajectory follows it.",
        f"**Order:** {correct_order}",
    ])


COLLISION_VARIANTS = [_collision_v0, _collision_v1, _collision_v2]


def build_collision_trace(ann_dict, presented_parts, segment_frame_ranges, use_frames=True,
                           use_timestamps=False, variant_fn=None):
    """segment_frame_ranges: see is_permanence_sufficient's docstring. Frame
    references are remapped via get_sampled_frame_index -- chain-of-frames
    grounding, same as build_permanence_trace. use_frames/use_timestamps/
    variant_fn: see build_permanence_trace's docstring -- same cof/cot/cotime
    pairing mechanism."""
    collision_dict = get_collision_frame_dict(ann_dict)
    k = len(segment_frame_ranges)
    part_to_video = {p: presented_parts.index(p) + 1 for p in range(k)}
    correct_order = " -> ".join([f"Video {part_to_video[p]}" for p in range(k)])

    sorted_frames = sorted(collision_dict.keys())
    obs_list = []

    for frame in sorted_frames:
        data = collision_dict[frame]
        obj_ids = data['object_ids']
        chrono_part = _find_segment(frame, segment_frame_ranges)
        collision_video = part_to_video[chrono_part]
        part_start, part_end = segment_frame_ranges[chrono_part]
        clip_frame_count = part_end - part_start + 1

        prior_videos = [part_to_video[p] for p in range(0, chrono_part)]
        subsequent_videos = [part_to_video[p] for p in range(chrono_part + 1, k)]

        obj_1_name = get_object_name(ann_dict, obj_ids[0])
        obj_2_name = get_object_name(ann_dict, obj_ids[1])

        if use_timestamps:
            ts = round((frame - part_start) / CLVR_FPS, 1)
            frame_clause = f" at {ts}s"
        elif use_frames:
            local_frame = get_sampled_frame_index(frame - part_start, clip_frame_count)
            frame_clause = f" at Frame {local_frame+1}"
        else:
            frame_clause = ""
        obs_list.append(f"In **Video {collision_video}**, the {obj_1_name} and the {obj_2_name} collide{frame_clause}, altering their trajectories.")

        if prior_videos:
            obs_list.append(f"In **Video {' and Video '.join(map(str, prior_videos))}**, the objects travel on their initial pre-collision paths.")
        if subsequent_videos:
            obs_list.append(f"In **Video {' and Video '.join(map(str, subsequent_videos))}**, the objects travel entirely on their new, post-collision paths.")

        break

    obs_text = " ".join(obs_list)
    variant = variant_fn or random.choice(COLLISION_VARIANTS)
    return variant(obs_text, correct_order)


def _boundary_v0(obs_text, correct_order):
    return "\n\n".join([
        f"{header(GROUNDING_HEADERS)} To determine the chronological sequence, we track the spatial boundary continuity of objects across the video cuts.",
        header(OBSERVATION_HEADERS),
        obs_text,
        f"{header(DEDUCTION_HEADERS)} Because continuous physics simulations require spatial coordinates to perfectly align across cuts, the videos with matching boundary coordinates follow one another immediately.",
        f"{header(CONCLUSION_HEADERS)} Therefore, the correct chronological order is **{correct_order}**.",
    ])


def _boundary_v1(obs_text, correct_order):
    return (
        f"Physics is continuous, so an object's position at the very last frame of one clip has "
        f"to match its position at the very first frame of whichever clip follows it -- there's no "
        f"room for a jump. {obs_text} Matching those boundary coordinates across the clips fixes "
        f"the order as {correct_order}."
    )


def _boundary_v2(obs_text, correct_order):
    return "\n".join([
        "**Boundary-coordinate evidence:**",
        f"  {obs_text}",
        "**Rule:** a clip's last-frame position must equal the next clip's first-frame position.",
        f"**Order:** {correct_order}",
    ])


BOUNDARY_VARIANTS = [_boundary_v0, _boundary_v1, _boundary_v2]


def build_spatial_continuity_trace(ann_dict, presented_parts, segment_frame_ranges, variant_fn=None):
    """
    segment_frame_ranges: see is_permanence_sufficient's docstring. This was
    the most severe form of the uniform-segment bug: end_frame_current /
    start_frame_next previously used a uniform total_frames // k assumption
    to index directly into ann_dict['motion_trajectory'], so the reported
    3D coordinates could be pulled from the wrong simulation frame entirely
    -- not just a mislabeled frame number, an actually wrong coordinate
    value that wouldn't match what's visible at the clip's real boundary.

    No use_frames toggle here (unlike build_permanence_trace/
    build_collision_trace): this trace is already coordinate-based, not
    frame-index-based, so there is no separate cof/cot content difference to
    make -- variant_fn still lets a caller pin the presentation style for
    consistency with how the other two traces build a cof/cot pair, even
    though both calls will return identical text.
    """
    k = len(segment_frame_ranges)
    part_to_video = {p: presented_parts.index(p) + 1 for p in range(k)}
    correct_order = " -> ".join([f"Video {part_to_video[p]}" for p in range(k)])

    obs_list = []
    for p in range(k - 1):
        end_frame_current = segment_frame_ranges[p][1]
        start_frame_next = segment_frame_ranges[p + 1][0]

        video_current = part_to_video[p]
        video_next = part_to_video[p + 1]

        anchor_obj = None
        for obj_id in range(len(ann_dict['object_property'])):
            if ann_dict['motion_trajectory'][end_frame_current]['objects'][obj_id]['inside_camera_view'] and ann_dict['motion_trajectory'][start_frame_next]['objects'][obj_id]['inside_camera_view']:
                anchor_obj = obj_id
                break
                
        if anchor_obj is not None:
            coords_end = ann_dict['motion_trajectory'][end_frame_current]['objects'][anchor_obj]['location'] 
            coords_start = ann_dict['motion_trajectory'][start_frame_next]['objects'][anchor_obj]['location']

            str_end = "[" + ", ".join([f"{c:.3f}" for c in coords_end]) + "]"
            str_start = "[" + ", ".join([f"{c:.3f}" for c in coords_start]) + "]"
            obj_name = get_object_name(ann_dict, anchor_obj)

            obs_list.append(f"At the final frame of **Video {video_current}**, the {obj_name} is located at coordinates `{str_end}`. At the first frame of **Video {video_next}**, the {obj_name} is located at extremely close coordinates `{str_start}`.")

    obs_text = " ".join(obs_list)
    variant = variant_fn or random.choice(BOUNDARY_VARIANTS)
    return variant(obs_text, correct_order)

class CLEVRER_Temporal_Alignment(CLEVRER_Dataset):
    def __init__(self, root_path, split, sample_num):
        super().__init__(root_path, split, sample_num)

# (Assuming add_alphabet_to_options is imported/defined elsewhere)

    def get_seq_order_QA(self, idx, total_frames=128, k=3, gap_time=0.2, gap_range=None,
                         max_start_offset=0.0):
        """gap_range=(lo, hi): k equal-length segments with gaps drawn from [lo, hi] s and a
        random start (up to max_start_offset s of extra slack); otherwise variable-length
        segments separated by gap_time."""
        print(f"Processing sample index: {idx}")
        (ann_name, ann_dict), = self.data_list[idx].items()

        # FIX 2: Dynamic parts based on k
        chronological_parts = list(range(k)) 

        presented_parts = chronological_parts.copy()
        random.shuffle(presented_parts)
        
        sample = dict()
        
        if gap_range is not None:
            # Equal-length segments, a random gap between neighbours and a random start:
            # no two clips share a boundary frame. With gap 0 the last frame of one clip
            # is the first frame of the next, and ordering reduces to matching boundary
            # frames (a single-task adapter scored 1.000 that way).
            gaps = [round(random.uniform(*gap_range), 2) for _ in range(k - 1)]
            slack = round(random.uniform(0.0, max_start_offset), 2)
            seg_len = math.floor((5.12 - sum(gaps) - slack) / k * 100) / 100
            if seg_len <= 0:
                raise ValueError(f"gaps {gaps} leave no room for {k} segments")
            spare = round(5.12 - sum(gaps) - k * seg_len, 2)
            current_time = round(random.uniform(0.0, spare), 2)
            original_ranges = []
            for j in range(k):
                end_time = min(5.12, round(current_time + seg_len, 2))
                original_ranges.append((current_time, end_time))
                current_time = round(end_time + (gaps[j] if j < k - 1 else 0.0), 2)
        else:
            # --- NEW VARIABLE LENGTH LOGIC ---
            min_length = 1.28
            total_gaps = k - 1
            usable_time = round(5.12 - (total_gaps * gap_time), 2)
        
            # Safety check to ensure the math is possible
            if usable_time < k * min_length:
                 raise ValueError(f"Impossible constraints: {usable_time}s is not enough for {k} segments of min {min_length}s.")
             
            remaining_time = round(usable_time - (k * min_length), 2)
        
            # Randomly distribute the remaining time budget among the k segments
            lengths = []
            for i in range(k - 1):
                # Pick a random chunk of the remaining time (rounded to 2 decimal places)
                add_time = round(random.uniform(0, remaining_time), 2)
                lengths.append(round(min_length + add_time, 2))
                remaining_time = round(remaining_time - add_time, 2)
            
            # The final segment gets whatever is left of the budget
            lengths.append(round(min_length + remaining_time, 2))
        
            # Shuffle the lengths so the last video segment isn't predictably the one getting the leftovers
            random.shuffle(lengths)
        
            # Map the randomized lengths to the original chronological timeline
            original_ranges = []
            current_time = 0.0
            for L in lengths:
                end_time = round(current_time + L, 2)
                original_ranges.append((current_time, end_time))
                current_time = round(end_time + gap_time, 2)
        # ---------------------------------

        # Real per-segment raw-frame boundaries, in chronological order --
        # segments are NOT equal length (min_length plus a randomly
        # distributed remainder, can differ by up to ~70% between segments
        # in the same example), so is_permanence_sufficient/
        # is_collision_sufficient/build_*_trace must use these instead of
        # assuming a uniform total_frames // k split (that assumption
        # previously misattributed physics events to the wrong clip and, in
        # build_spatial_continuity_trace, pulled coordinates from the wrong
        # simulation frame entirely). Same ceil/floor convention
        # qwen_vl_utils' calculate_video_frame_range uses for video_start/
        # video_end, so these boundaries match what actually gets rendered
        # into each clip.
        segment_frame_ranges = [
            (math.ceil(start * CLVR_FPS), min(math.floor(end * CLVR_FPS), total_frames - 1))
            for start, end in original_ranges
        ]

        sample['question'] = (
            f"These {k} videos are segments of a single continuous event, "
            "but they are shown out of order. What is the correct chronological order of these video segments?"
        )
        
        correct_order_indices = [presented_parts.index(p) + 1 for p in chronological_parts]
        correct_answer = " -> ".join([f"Video {i}" for i in correct_order_indices])
        correct_tuple = tuple(correct_order_indices) # Keep as tuple for math comparisons
        
        sample['video_list'] = [{'video_path': os.path.join(self.video_path, f"video_{ann_name}.mp4"), 
                                 'start_sec': start, 'end_sec': end} for start, end in [original_ranges[p] for p in presented_parts]]
        
        # Distractors are drawn uniformly from the wrong permutations. The old
        # lowest-Hamming-distance choice made the answer the "centre" of the
        # options, answerable without the video (see sample_order_distractors).
        wrong_options = [" -> ".join([f"Video {i}" for i in p])
                         for p in sample_order_distractors(correct_tuple, 3)]
        
        options = [correct_answer] + wrong_options
        random.shuffle(options)
        
        sample['options'], sample['answer'] = add_alphabet_to_options(options, correct_answer)
        
        if self.split == 'validation':
            return sample
        
        valid_reasoning_types = ['boundary'] # Boundary matching is always a valid fallback

        if is_permanence_sufficient(ann_dict, segment_frame_ranges):
            valid_reasoning_types.append('permanence')
        if is_collision_sufficient(ann_dict, segment_frame_ranges):
            valid_reasoning_types.append('collision')

        reasoning_type = random.choice(valid_reasoning_types)

        # Build a chain-of-frames (exact sampled-frame references), a
        # chain-of-timestamps (seconds, sampling-density-independent -- see
        # build_permanence_trace's use_timestamps docstring), and a plain-CoT
        # (all numeric grounding stripped) version of the same trace, using
        # the SAME randomly-picked presentation style for all three -- the
        # only difference between them is the frame/timestamp annotations,
        # isolating that variable cleanly. Which one actually lands in the
        # training set is decided later, per-example, by data_qwenvl.py's
        # --cof-ratio/--cotime-ratio, so the mixing ratio can be tuned
        # without re-running this generator.
        if reasoning_type == 'permanence':
            variant_fn = random.choice(PERMANENCE_VARIANTS)
            sample['reasoning_cof'] = build_permanence_trace(ann_dict, presented_parts, segment_frame_ranges, use_frames=True, variant_fn=variant_fn)
            sample['reasoning_cotime'] = build_permanence_trace(ann_dict, presented_parts, segment_frame_ranges, use_frames=False, use_timestamps=True, variant_fn=variant_fn)
            sample['reasoning_cot'] = build_permanence_trace(ann_dict, presented_parts, segment_frame_ranges, use_frames=False, variant_fn=variant_fn)
            sample['process_gt'] = extract_permanence_process_gt(ann_dict, presented_parts, segment_frame_ranges)
        elif reasoning_type == 'collision':
            variant_fn = random.choice(COLLISION_VARIANTS)
            sample['reasoning_cof'] = build_collision_trace(ann_dict, presented_parts, segment_frame_ranges, use_frames=True, variant_fn=variant_fn)
            sample['reasoning_cotime'] = build_collision_trace(ann_dict, presented_parts, segment_frame_ranges, use_frames=False, use_timestamps=True, variant_fn=variant_fn)
            sample['reasoning_cot'] = build_collision_trace(ann_dict, presented_parts, segment_frame_ranges, use_frames=False, variant_fn=variant_fn)
            sample['process_gt'] = extract_collision_process_gt(ann_dict, presented_parts, segment_frame_ranges)
        elif reasoning_type == 'boundary':
            variant_fn = random.choice(BOUNDARY_VARIANTS)
            trace = build_spatial_continuity_trace(ann_dict, presented_parts, segment_frame_ranges, variant_fn=variant_fn)
            sample['reasoning_cof'] = trace
            sample['reasoning_cotime'] = trace
            sample['reasoning_cot'] = trace
            # No process_gt: this trace reasons over raw 3D coordinates, not
            # timestamps or nameable facts -- doesn't fit process_reward's
            # numeric-tolerance or substring checks. the GRPO training code
            # defaults a missing key to {}, which process_reward treats as
            # "no process reward for this sample" (returns None, excluded),
            # same as ReID's object-absent case.

        sample['reasoning'] = sample['reasoning_cof']  # backward-compat default for any caller reading 'reasoning' directly
        return sample



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multimodal video models.")
    
    parser.add_argument("--total-num", type=int, default=10, 
                        help="Index of the sample to retrieve from the dataset.")
    
    parser.add_argument("--idx", type=int, default=0,
                        help="Index of the sample to retrieve from the dataset.")

    parser.add_argument("--split", type=str, default='validation',
                        choices=['train', 'validation'],
                        help="CLEVRER split to draw from. 'train' -> video_train/annotation_train "
                             "(ids 0-9999); 'validation' -> video_validation/annotation_validation "
                             "(ids 10000-14999). Reasoning traces are emitted for 'train' only "
                             "(see the split guard in get_seq_order_QA).")

    parser.add_argument("--root-path", type=str, default=f"{DATA_ROOT}/clvr/",
                        help="Root directory containing video_<split>/ and annotation_<split>/.")

    parser.add_argument("--output", type=str, default=None,
                        help="Explicit output .jsonl path. Defaults to a train/ or eval/ path "
                             "chosen from --split.")

    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for segment lengths, distractors and reasoning variants.")

    args = parser.parse_args()
    random.seed(args.seed)


    dataset = CLEVRER_Temporal_Alignment(
            root_path=args.root_path,
            split=args.split,
            sample_num=args.total_num
        )
    data_list = []
    for idx in range(args.total_num):
        data_list.append(dataset.get_seq_order_QA(idx, gap_time=0.0,
                                                  total_frames=128, k=4))

    if args.output:
        out_path = args.output
    elif args.split == 'train':
        out_path = 'json_files/train/Clvr_Sequential_Ordering.jsonl'
    else:
        out_path = 'json_files/eval/Temporal_Alignment/Clvr_Sequential_Ordering.jsonl'

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_jsonl_file(filepath=out_path, data_list=data_list)
    print(f"Saved {len(data_list)} samples to {out_path} (split={args.split})")

    print(idx,'---------------------'*5)
    for key, val in data_list[0].items():
        print(key, val)
    print('---------------------'*5)


# Eval (unchanged behaviour):
#   python -m generation.clevrer.clvr_temporal_alignment --total-num 1000
# Train:
#   python -m generation.clevrer.clvr_temporal_alignment --total-num 1000 --split train

