'''
python -m generation.habitat.habitat_entity_tracking
'''

from utils import DATA_ROOT
import argparse
import os
import random
from utils import write_jsonl_file
from generation.habitat.base_habitat import *
from generation.habitat.habitat_traces import build_reid_trace, REID_ABSENT_VARIANTS, REID_PRESENT_VARIANTS
from generation.option_balance import REID_ABSENT_OPTION, reoption_reid


def format_single_range_to_sec(rng, fps):
    """Helper to convert frame tuples to string second ranges."""
    if not rng:
        return "The object does not appear in this video."
    start_sec = rng[0] / float(fps)
    end_sec = rng[1] / float(fps)
    return f"{start_sec:.1f}s to {end_sec:.1f}s"

def filter_presence_data(
    presence_data, 
    min_range_len=3, 
    min_total_frames=5, 
    total_video_frames=None, 
    drop_boundary_ranges=False
):
    """
    Filters temporal presence data based on length, total visibility, and boundaries.
    
    Args:
        presence_data (dict): e.g., {'cam0': [(20, 25)], 'cam1': [(55, 65)]}
        min_range_len (int): Minimum frames a single continuous range must have.
        min_total_frames (int): Minimum total frames the object must be visible in the video.
        total_video_frames (int): Total length of the video (required if drop_boundary_ranges is True).
        drop_boundary_ranges (bool): If True, removes ranges that touch the start/end of the video.
        
    Returns:
        dict: The filtered presence data. Returns an empty dict if < 2 cameras survive.
    """
    filtered_data = {}
    
    for cam, ranges in presence_data.items():
        valid_ranges = []
        total_visible_frames = 0
        
        for start, end in ranges:
            # Calculate how many frames this specific range covers
            range_len = (end - start) + 1
            
            # Criterion 1: Is the range too short? (Your idea!)
            if range_len < min_range_len:
                continue
                
            # Criterion 2: Does it touch the edge of the video?
            if drop_boundary_ranges and start == 0:
                    continue 
            # if drop_boundary_ranges and total_video_frames is not None:
            #     if start == 0 or end == total_video_frames - 1:
            #         continue 
            
            # If it passed the checks, keep it
            valid_ranges.append((start, end))
            total_visible_frames += range_len
            
        # Criterion 3: Is the object visible long enough in this specific video overall?
        # Also ensures we don't add empty lists [] to our filtered dict.
        if valid_ranges and total_visible_frames >= min_total_frames:
            filtered_data[cam] = valid_ranges

    # Criterion 4: Do we still have at least 2 cameras to make a QA pair?
    if len(filtered_data) < 2:
        return {} # Return completely empty, this object is dead for cross-video QA!
        
    return filtered_data

def generate_single_object_reidentification_qa(scene_path, clean_obj_name, presence_data, ref_cam,
                                               target_cam, fps=30,
                                               include_reasoning=False, frame_counts=None):
    """One ReID item carrying only its correct option. Distractors are drawn across
    the whole list by generation.option_balance.reoption_reid (generate_object_reidentification_qa),
    because the old +1 s / -1 s / +2 s shifts made the answer the second-earliest window."""

    ref_ranges = presence_data.get(ref_cam, [])
    target_ranges = presence_data.get(target_cam, [])

    chosen_ref_range = random.choice(ref_ranges) if ref_ranges else None
    ref_time_str = format_single_range_to_sec(chosen_ref_range, fps)

    # 3. Formulate the Question (Now explicitly asking for FIRST appearance)
    question = (
        "You are provided with two videos recorded at {fps} FPS. "
        f"In Video 1, the '{clean_obj_name}' is clearly visible from {ref_time_str}. "
        f"Select the correct timestamp window for the FIRST appearance of the exact same '{clean_obj_name}' in Video 2."
    )

    first_target_range = None
    if not target_ranges:
        # Edge case: Target object is completely absent
        correct_answer = REID_ABSENT_OPTION
    else:
        # Sort to guarantee we grab the chronological FIRST appearance
        target_ranges.sort(key=lambda x: x[0])
        first_target_range = target_ranges[0]
        correct_answer = format_single_range_to_sec(first_target_range, fps)

    target_frames = (frame_counts or {}).get(target_cam)
    sample = {
        "question": question,
        "options": [f"A) {correct_answer}"],
        "answer": f"A) {correct_answer}",
        # Length of Video 2 in 0.1 s, for reoption_reid; popped before saving.
        "_video_2_tenths": int(round(target_frames * 10 / fps)) if target_frames else None,
        "video_list": [
            {'video_path': os.path.join(scene_path, ref_cam, f'{ref_cam}.mp4')},
            {'video_path': os.path.join(scene_path, target_cam, f'{target_cam}.mp4')}
        ]
    }

    # Structured ground truth for the GRPO process reward (training code, not released):
    # the target object's re-appearance window in seconds. Numeric (not a
    # phrase), so process_reward auto-routes it through tolerance-based
    # number matching -- an exact-string check would rarely fire, since a
    # model reasoning in its own words will rarely emit the precise
    # substring "17.0s" rather than e.g. "around 17 seconds in". Omitted
    # (not set to some placeholder) for the object-absent case: there is no
    # meaningful timestamp to check a model's reasoning against when the
    # correct answer is "the object does not appear in this video."
    #
    # Keys are named "video_2_*", not "first_target_*", because video_list
    # above puts ref_cam first and target_cam (what these seconds describe)
    # second -- i.e. this is Video 2 in the prompt's own numbering. Naming
    # it that way lets process_reward's proximity window anchor the check to
    # text actually near "Video 2" mentions, rather than anywhere in the
    # reasoning body -- otherwise ref_cam's own stated timing (Video 1's,
    # handed to the model as a given) sits close enough in value to satisfy
    # this within numeric_tolerance with no re-identification performed.
    if first_target_range is not None:
        sample['process_gt'] = {
            'video_2_start_sec': round(first_target_range[0] / fps, 1),
            'video_2_end_sec': round(first_target_range[1] / fps, 1),
        }

    if include_reasoning:
        frame_counts = frame_counts or {}
        # Pin one presentation style (picked from whichever pool applies --
        # "absent" vs. "present" is a fact about the ground truth, identical
        # for both calls) so the cof/cot pair differs only in the frame
        # references, not the wording. Which one lands in the training set is
        # decided later by data_qwenvl.py's --cof-ratio.
        variant_pool = REID_ABSENT_VARIANTS if not target_ranges else REID_PRESENT_VARIANTS
        variant_fn = random.choice(variant_pool)
        sample['reasoning_cof'] = build_reid_trace(
            object_name=clean_obj_name,
            ref_range=chosen_ref_range,
            target_ranges=target_ranges,
            answer_range=first_target_range,
            fps=fps,
            ref_total_frames=frame_counts.get(ref_cam),
            target_total_frames=frame_counts.get(target_cam),
            variant_fn=variant_fn,
        )
        sample['reasoning_cot'] = build_reid_trace(
            object_name=clean_obj_name,
            ref_range=chosen_ref_range,
            target_ranges=target_ranges,
            answer_range=first_target_range,
            fps=fps,
            ref_total_frames=None,
            target_total_frames=None,
            variant_fn=variant_fn,
        )
        # _fmt_range_with_frames (see habitat_traces.py) always includes the
        # timestamp and only ADDS the frame-index parenthetical when
        # total_frames is given -- so with ref/target_total_frames=None above,
        # reasoning_cot here is already exactly the chain-of-timestamps
        # content (seconds, no frame index), not qualitative-only phrasing.
        # reasoning_cotime is therefore identical by construction, not a
        # shortcut -- kept as its own key so data_qwenvl.py's 3-way selection
        # works uniformly across CLEVRER (which has a genuinely distinct,
        # numeric-grounding-free reasoning_cot) and Habitat.
        sample['reasoning_cotime'] = sample['reasoning_cot']
        sample['reasoning'] = sample['reasoning_cof']

    return sample
     

def generate_object_reidentification_qa(save=False, verbose=True, split='val',
                                        root_path=f"{DATA_ROOT}/habitat_data",
                                        output=None, include_reasoning=None,
                                        total_num=None):
    # Reasoning traces are for SFT data only; the benchmark split stays unchanged.
    if include_reasoning is None:
        include_reasoning = (split == 'train')

    qa_list = list()
    habitat_obj = Habitat_Dataset(root_path=root_path, section='general', split=split)

    for idx, scene_idx in enumerate(os.listdir(habitat_obj.data_path)):
        if total_num is not None and len(qa_list) >= total_num:
            break

        scene_path = os.path.join(habitat_obj.data_path, scene_idx)
        print(f"Processing scene: {scene_idx} ({idx}/{len(os.listdir(habitat_obj.data_path))}), qa samples so far: {len(qa_list)}")

        if not os.path.isdir(scene_path):
            continue

        common_obj_ids = habitat_obj.get_common_object_ids(scene_idx=scene_idx)
        frame_counts = habitat_obj.get_video_frame_counts(scene_idx=scene_idx)

        for obj_id in common_obj_ids:
            full_object_name = habitat_obj.get_object_name(scene_idx=scene_idx, object_id=obj_id)


            if full_object_name == 'Unknown Object':
                continue

            object_name = ' '.join(full_object_name.split(' ')[1:-1])

            if object_name in objects_to_filter:
                continue

            presence_data = filter_presence_data(
                habitat_obj.get_timely_object_presence(scene_idx=scene_idx, object_id=obj_id),
                min_range_len=3, min_total_frames=5, drop_boundary_ranges=False)

            cam_list = list(presence_data.keys())
            for ref_idx in range(len(cam_list)-1):
                for target_idx in range(ref_idx+1, len(cam_list)):
                    qa_list.append(generate_single_object_reidentification_qa(
                        scene_path=habitat_obj._get_scene_path(scene_idx),
                        clean_obj_name=object_name,
                        presence_data=presence_data,
                        ref_cam=cam_list[ref_idx],
                        target_cam=cam_list[target_idx],
                        fps=5,
                        include_reasoning=include_reasoning,
                        frame_counts=frame_counts,
                    ))
                    if total_num is not None and len(qa_list) >= total_num:
                        break
                if total_num is not None and len(qa_list) >= total_num:
                    break
            if total_num is not None and len(qa_list) >= total_num:
                break


    # Distractors are borrowed across items (see reoption_reid), so they are
    # only final once the whole list exists.
    video_tenths = [row.pop('_video_2_tenths') for row in qa_list]
    if qa_list:
        rebuilt = reoption_reid([row['question'] for row in qa_list],
                                [row['answer'][3:] for row in qa_list], video_tenths, random)
        for row, (options, answer) in zip(qa_list, rebuilt):
            row['options'], row['answer'] = options, answer

    if verbose and qa_list:
        print(f"Generated {len(qa_list)} QA pairs. Here is the first one:")
        for key, val in qa_list[0].items():
            if key == 'options':
                [print(val_i) for val_i in val]
            else:
                print(key, val)
            print()


    if save:
        if output:
            out_path = output
        elif split == 'train':
            out_path = 'json_files/train/Habitat_Object_Re_identification.jsonl'
        else:
            out_path = 'json_files/eval/Spatial_Tracking/Habitat_Object_Re_identification.jsonl'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_jsonl_file(filepath=out_path, data_list=qa_list)
        print(f"Saved {len(qa_list)} samples to {out_path} (split={split})")

    return qa_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Habitat Object Re-identification QA pairs.")

    parser.add_argument("--split", type=str, default='val', choices=['train', 'val'],
                        help="HM3D split to draw from. 'train' emits reasoning traces "
                             "for SFT; 'val' reproduces the benchmark.")
    parser.add_argument("--root-path", type=str, default=f"{DATA_ROOT}/habitat_data",
                        help="Root of the rendered Habitat data tree.")
    parser.add_argument("--output", type=str, default=None,
                        help="Explicit output .jsonl path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the sample dump.")
    parser.add_argument("--total-num", type=int, default=None,
                        help="Cap the number of QA pairs generated. Unbounded (full "
                             "natural yield) if omitted -- used for the benchmark, "
                             "where the exact count is a fixed property of the dataset.")

    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for window choice, trace variants and distractors.")

    args = parser.parse_args()
    random.seed(args.seed)

    generate_object_reidentification_qa(
        save=True,
        verbose=not args.quiet,
        split=args.split,
        root_path=args.root_path,
        output=args.output,
        total_num=args.total_num,
    )


# Eval (unchanged behaviour):
#   python -m generation.habitat.habitat_entity_tracking
# Train:
#   python -m generation.habitat.habitat_entity_tracking --split train