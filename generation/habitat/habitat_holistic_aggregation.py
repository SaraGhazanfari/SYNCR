#python -m generation.habitat.habitat_holistic_aggregation

from utils import DATA_ROOT
import argparse
import os
import random
import itertools

from utils import write_jsonl_file, read_json_file
from generation.habitat.base_habitat import *
from generation.habitat.habitat_traces import (
    build_counting_trace, build_route_trace, COUNTING_VARIANTS, ROUTE_VARIANTS, _fmt_video_list,
)
from generation.option_balance import reoption_counting, reoption_route_v3
import networkx as nx

# Was previously a bare module-level `path`, defined only inside __main__, which
# made these generators raise NameError on import.
DEFAULT_ROOT_PATH = f"{DATA_ROOT}/habitat_data"

def rebalance_qa_list(qa_list):

    group_1 = []
    group_others = []
    target_count = 0
    # 1. Group the items into "1" and ">= 2"
    for item in qa_list:
        try:
            count_key = int(item['answer'].split(') ')[1].strip())
            
            if count_key == 1:
                group_1.append(item)
            elif count_key >= 2:
                # This captures your 2s, 3s, 4s, 5s, etc.
                group_others.append(item)
                if count_key == 2:
                    target_count += 1
                
        except (IndexError, ValueError):
            continue
    print(f"Total samples with count 1 before subsampling: {len(group_1)}")

    # 3. Subsample group 1 to match the size of group_others
    # We add a quick check just in case you actually have fewer 1s than others
    if len(group_1) > target_count:
        balanced_1 = random.sample(group_1, target_count)
        print(f"Subsampled 1s down to {target_count}.")
    else:
        balanced_1 = group_1
        print("Note: You have fewer 1s than >=2s, so all 1s were kept.")

    # 4. Combine the lists
    balanced_qa_list = balanced_1 + group_others

    # 5. Shuffle the final combined list
    random.shuffle(balanced_qa_list)
    return balanced_qa_list

    print(f"New balanced dataset size: {len(balanced_qa_list)}")

import random
import numpy as np

def generate_single_object_counting_qa(habitat_obj, scene_idx, route_idx=None,
                                       include_reasoning=False):
    qa_list = list()
    
    # Get the raw info dictionary to map names to specific instance IDs
    processed_object_info_dict, aggregated_object_count_dict = habitat_obj.get_scene_object_info(scene_idx=scene_idx, route_idx=route_idx)
    
    # Get the semantic masks and video paths
    video_framewise_object_ids_list, video_path_list = habitat_obj._get_raw_item(scene_idx, route_idx)
    num_videos = len(video_path_list)
    
    if num_videos == 0:
        return qa_list 
        
    video_dicts = [{'video_path': path} for path in video_path_list]
    
    for object_name, count in aggregated_object_count_dict.items():
        if count > 10 or object_name in objects_to_filter:
            continue
            
        # 1. Find all specific instance IDs that belong to this object_name
        target_ids = []
        for obj_id, val_str in processed_object_info_dict.items():
            parts = val_str.split(' ')
            if len(parts) >= 3:
                name = ' '.join(parts[1:-1])
                if name == object_name:
                    target_ids.append(obj_id)
                    
        # 2. Verify that EVERY instance ID occupies >= 5% in at least one frame
        all_instances_visible = True
        # {instance_id: [video indices where it clears the threshold]} -- needed to
        # write a deduplication trace, so when tracing we cannot stop at the first hit.
        instance_to_videos = {}

        # {instance_id: {video_idx: (best_frame, video_total_frames)}} -- chain-of-
        # frames grounding: the frame where each instance is most clearly visible in
        # each video, computed as a side effect of the existing threshold check
        # (pixel_counts_per_frame was already being computed, just not kept).
        instance_frame_info = {}

        for t_id in target_ids:
            visible_in = []
            frame_info = {}

            for video_idx, semantic_video in enumerate(video_framewise_object_ids_list):
                # Calculate the 5% pixel threshold for this video resolution
                total_pixels = semantic_video.shape[1] * semantic_video.shape[2]
                threshold_pixels = total_pixels * 0.05

                # Reshape to (num_frames, H*W) for fast numpy calculations
                flat_view = semantic_video.reshape((semantic_video.shape[0], -1))

                # Count how many pixels this specific ID occupies in EVERY frame simultaneously
                pixel_counts_per_frame = np.sum(flat_view == t_id, axis=1)

                # If the max pixel count in any frame beats the threshold, this instance passes
                if np.max(pixel_counts_per_frame) >= threshold_pixels:
                    visible_in.append(video_idx)
                    frame_info[video_idx] = (int(np.argmax(pixel_counts_per_frame)), semantic_video.shape[0])
                    if not include_reasoning:
                        # Acceptance only needs one hit; keep the original early exit
                        # when we are not building a trace.
                        break

            instance_to_videos[t_id] = visible_in
            instance_frame_info[t_id] = frame_info

            # If after checking all cameras, this specific instance never hit 5%, the object fails
            if not visible_in or len(video_framewise_object_ids_list) == 1:
                all_instances_visible = False
                break # Stop checking other instances, skip this object entirely
                
        # If any instance of this object was too small, skip generating the question
        if not all_instances_visible:
            continue
    
        # 3. Proceed with QA Generation
        clean_obj_name = object_name.replace('_', ' ')
        
        question = (
            f"Based on the combined exploration in all {num_videos} videos, "
            f"how many distinct instances of '{clean_obj_name}' are present in this scene?"
        )
        
        correct_answer = str(count)
        # Distractors are chosen over the whole list by reoption_counting in
        # generate_object_counting_qa: count + randint(-3, 3) made the answer the
        # second-smallest option.
        option_list = [f"A) {correct_answer}"]
        correct_label = option_list[0]
        
        qa_pair = {
            "question": question,
            "options": option_list,
            "answer": correct_label,
            "video_list": video_dicts
        }

        if include_reasoning:
            # processed_object_info_dict maps id -> "<n> <name> <region>", where <n>
            # is that instance's index within its object category.
            instance_labels = {}
            for t_id in target_ids:
                parts = processed_object_info_dict[t_id].split(' ')
                instance_labels[t_id] = f"#{parts[0]}" if parts and parts[0].isdigit() else f"#{t_id}"

            # Pin one presentation style so the cof/cot pair differs only in
            # the frame references, not the wording. Which one lands in the
            # training set is decided later by data_qwenvl.py's --cof-ratio.
            variant_fn = random.choice(COUNTING_VARIANTS)
            qa_pair['reasoning_cof'] = build_counting_trace(
                object_name=clean_obj_name,
                instance_labels=instance_labels,
                instance_to_videos=instance_to_videos,
                total_count=count,
                instance_frame_info=instance_frame_info,
                variant_fn=variant_fn,
            )
            qa_pair['reasoning_cot'] = build_counting_trace(
                object_name=clean_obj_name,
                instance_labels=instance_labels,
                instance_to_videos=instance_to_videos,
                total_count=count,
                instance_frame_info=None,
                variant_fn=variant_fn,
            )
            qa_pair['reasoning'] = qa_pair['reasoning_cof']

            # Process reward ground truth: which video(s) each *repeated*
            # instance is seen in -- the actual dedup claim build_counting_trace's
            # DEDUCTION section states (see its "{label} appears in {video list}"
            # line), not the final union count itself (that's the outcome
            # answer). Only repeated instances are included since a non-repeated
            # instance's "appears in exactly one video" isn't a claim the trace
            # bothers to spell out explicitly. Empty when nothing repeats --
            # process_reward treats an empty dict as "not applicable", same as
            # ReID's object-absent case.
            repeated = {i: vids for i, vids in instance_to_videos.items() if len(vids) > 1}
            qa_pair['process_gt'] = {
                f"{instance_labels[i]}_videos": _fmt_video_list(vids)
                for i, vids in repeated.items()
            }

        qa_list.append(qa_pair)

    return qa_list


def _collect_counting_qa_for_section(root_path, section, split, total_num, include_reasoning):
    """One section's worth of counting QA (no rebalancing -- that happens once
    on the combined pool in generate_object_counting_qa, whether the caller
    asked for one section or 'both')."""
    habitat_obj = Habitat_Dataset(root_path=root_path, section=section, split=split)
    qa_list = list()

    for scene_idx in os.listdir(habitat_obj.data_path):
        if len(qa_list) >= total_num:
            break

        scene_path = os.path.join(habitat_obj.data_path, scene_idx)

        # Skip files, only process directories
        if not os.path.isdir(scene_path):
            continue

        if habitat_obj.data_type == 'general':
            qa_list.extend(generate_single_object_counting_qa(
                habitat_obj, scene_idx, route_idx=None,
                include_reasoning=include_reasoning))
            print(f"[{section}] Processed Scene {scene_idx}. QA pairs so far: {len(qa_list)}")
        else:
            # Checked per-route, not just per-scene: a 'route_plan' section
            # scene can have many routes, each adding a batch of QA pairs, so
            # a scene-level-only check can overshoot total_num by an entire
            # scene's worth at real scale.
            for route_idx in os.listdir(scene_path):
                if len(qa_list) >= total_num:
                    break

                route_path = os.path.join(scene_path, route_idx)
                if os.path.isdir(route_path):
                    qa_list.extend(generate_single_object_counting_qa(
                        habitat_obj, scene_idx, route_idx=route_idx,
                        include_reasoning=include_reasoning))
                    print(f"[{section}] Processed Scene {scene_idx}. QA pairs so far: {len(qa_list)}")

    return qa_list


def generate_object_counting_qa(total_num=float('inf'), section='general', save=False,
                                split='val', root_path=DEFAULT_ROOT_PATH, output=None,
                                include_reasoning=None):
    """
    section: 'general', 'route_plan', or 'both'. 'both' pools QA from each
    section (each capped at total_num independently, since route_plan alone
    was observed well under its own cap -- the bottleneck is the per-example
    filters, not the total_num ceiling) and rebalances once over the combined
    pool, for more training volume than either section yields alone. The
    benchmark (--split val) should keep using a single section to match the
    shipped dataset's composition.
    """
    if include_reasoning is None:
        include_reasoning = (split == 'train')

    if output:
        out_path = output
    elif split == 'train':
        out_path = 'json_files/train/Habitat_Object_Counting.jsonl'
    else:
        out_path = 'json_files/eval/Holistic_Synthesis/Habitat_Object_Counting.jsonl'

    def save_progress(qa_list):
        if save:
            # Ensure the target directory exists to prevent FileNotFoundError
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            write_jsonl_file(filepath=out_path, data_list=qa_list)

    sections = ['general', 'route_plan'] if section == 'both' else [section]

    qa_list = []
    for sec in sections:
        qa_list.extend(_collect_counting_qa_for_section(
            root_path, sec, split, total_num, include_reasoning))
        save_progress(qa_list)

    qa_list = rebalance_qa_list(qa_list)
    if qa_list:
        # Distractors are chosen over the whole list (see reoption_counting).
        rebuilt, stats = reoption_counting([int(row['answer'][3:]) for row in qa_list], random)
        for row, (options, answer) in zip(qa_list, rebuilt):
            row['options'], row['answer'] = options, answer
        print(f"Counting options rebuilt: {stats}")
    save_progress(qa_list)

    return qa_list


def generate_single_route_plan_qa(habitat_obj, scene_idx, route_idx, mode="all",
                                  include_reasoning=False):
    """
    mode: "all" (returns QA pairs for every sub-path >= 3 nodes)
          "longest" (returns only the single QA pair for the maximum path)
    """
    
    # 1. Load the video_routes.json and Verify Videos
    scene_path = habitat_obj._get_scene_path(scene_idx, route_idx)
    routes_file = os.path.join(scene_path, "video_routes.json")
    
    # Assuming read_json_file is defined elsewhere
    routes_dict = read_json_file(routes_file) 
    
    video_paths = habitat_obj._get_video_path(scene_idx, route_idx)
    for vp in video_paths:
        if not os.path.exists(vp):
            print(f"Warning: Missing video file {vp}. Skipping this route.")
            return [] 

    # Videos are shuffled before presentation, so remember where each camera landed:
    # a trace must refer to the video numbering the model actually sees.
    cam_to_presented_idx = {}
    for vp in video_paths:
        cam_name = os.path.basename(os.path.dirname(vp))  # ".../cam1/cam1.mp4" -> "cam1"
        cam_to_presented_idx[cam_name] = None

    random.shuffle(video_paths)
    video_list_formatted = [{'video_path': vp} for vp in video_paths]

    for presented_idx, vp in enumerate(video_paths):
        cam_name = os.path.basename(os.path.dirname(vp))
        cam_to_presented_idx[cam_name] = presented_idx

    # 2. Reconstruct the Full Ground-Truth Path
    full_path = []
    num_cams = len(routes_dict)
    cam_chunks = []

    for i in range(num_cams):
        cam_key = f"cam{i}"
        if cam_key not in routes_dict:
            return []

        chunk = routes_dict[cam_key]
        cam_chunks.append((cam_key, chunk))
        if not full_path:
            full_path.extend(chunk)
        else:
            full_path.extend(chunk[1:])

    if len(full_path) < 3:
        return []

    all_nodes_in_video = list(set(full_path))
    generated_qas = []

    # Chain-of-frames: which rendered frame the camera is likely in each region
    # it passes through, computed once per route (identical across every
    # sliding-window QA pair below, since it only depends on cam_chunks). See
    # get_region_object_ids' docstring for why this is a proxy (peak visibility
    # of the region's characteristic objects) rather than exact ground truth --
    # the renderer never recorded true frame->region during the walk, and
    # re-rendering to add that would be expensive.
    region_frame_evidence = {}  # {(cam_key, region_id): (raw_frame, video_total_frames)}
    if include_reasoning:
        video_framewise_object_ids_list, _ = habitat_obj._get_raw_item(scene_idx, route_idx)
        for cam_idx, (cam_key, chunk) in enumerate(cam_chunks):
            if cam_idx >= len(video_framewise_object_ids_list):
                continue
            semantic_video = video_framewise_object_ids_list[cam_idx]
            video_total_frames = semantic_video.shape[0]
            flat_view = semantic_video.reshape((video_total_frames, -1))
            for region_id in chunk:
                key = (cam_key, str(region_id))
                if key in region_frame_evidence:
                    continue
                obj_ids = habitat_obj.get_region_object_ids(scene_idx, route_idx, str(region_id))
                if not obj_ids:
                    continue
                pixel_counts_per_frame = np.isin(flat_view, list(obj_ids)).sum(axis=1)
                if pixel_counts_per_frame.max() > 0:
                    best_frame = int(pixel_counts_per_frame.argmax())
                    region_frame_evidence[key] = (best_frame, video_total_frames)

    # --- MODE CONTROL LOGIC ---
    if mode == "longest":
        # Only process the single maximum length of the path
        lengths_to_process = [len(full_path)]
    elif mode == "all":
        # Process every possible sliding window length from 3 up to the max
        lengths_to_process = range(3, len(full_path) + 1)
        
    elif mode == "medium":
        lengths_to_process = range(max(3,len(full_path))-1 , len(full_path) + 1)
    else:
        print(f"Warning: Unknown mode '{mode}'. Defaulting to 'all'.")
        lengths_to_process = range(3, len(full_path) + 1)

    # 3. SLIDING WINDOW: Extract sub-paths based on the allowed lengths
    for length in lengths_to_process:
        for start_idx in range(len(full_path) - length + 1):
            
            sub_path = full_path[start_idx : start_idx + length]
            
            start_node = sub_path[0]
            end_node = sub_path[-1]
            true_middle = sub_path[1:-1]
            
            # --- Build the Distractor Pool ---
            pool_mids = true_middle.copy()
            
            # Pull in nodes we saw in the video but aren't in this specific sub-path!
            unused_video_nodes = [n for n in all_nodes_in_video if n not in sub_path]
            pool_mids.extend(unused_video_nodes)
            
            # Edge Case Fix: Pad with a valid random room if we lack permutations
            if len(pool_mids) < 2:
                fake_node = None
                attempts = 0
                while not fake_node and attempts < 50:
                    test_node = str(random.randint(0, 50)) 
                    if test_node not in full_path:
                        if habitat_obj.guess_room_name(scene_idx, route_idx, test_node):
                            fake_node = test_node
                    attempts += 1
                if fake_node:
                    pool_mids.append(fake_node)
                    
            # 4. Format strings helper
            def format_path(path_tuple):
                names = []
                for i, r in enumerate(path_tuple):
                    base_name = habitat_obj.guess_room_name(scene_idx, route_idx, str(r))
                    if not base_name:
                        return None

                    if i == 0:
                        names.append(f"Region 1 <{base_name}>")
                    elif i == len(path_tuple) - 1:
                        names.append(f"Region 2 <{base_name}>")
                    else:
                        names.append(base_name)
                return " -> ".join(names)

            # 5. Correct option only. Distractors are built from the named rooms in
            # pool_mids by reoption_route in generate_route_plan_qa; the old ones
            # offered the never-correct direct path and could be the string None.
            correct_str = format_path(tuple(sub_path))
            if not correct_str:
                continue
            pool_names = [name for name in (habitat_obj.guess_room_name(scene_idx, route_idx, str(r))
                                            for r in pool_mids) if name]
            options_list = [f"A) {correct_str}"]
            correct_label = options_list[0]
            
            # 7. Package and Append this specific QA pair
            start_name = habitat_obj.guess_room_name(scene_idx, route_idx, str(start_node))
            end_name = habitat_obj.guess_room_name(scene_idx, route_idx, str(end_node))
            question = (
                f"Based on the spatial layout shown across the videos, what is the shortest possible "
                f"path to travel from Region 1 <{start_name}> to Region 2 <{end_name}>?"
            )
            
            qa_pair = {
                "scene_idx": scene_idx,
                "route_idx": route_idx,
                "question_type": "unseen_shortest_path",
                "question": question,
                "options": options_list,
                "answer": correct_label,
                "video_list": video_list_formatted,
                "_route_pool": pool_names,  # popped by generate_route_plan_qa
            }

            if include_reasoning:
                # What each presented video contributes, in the model's numbering.
                # Region names get a chain-of-frames annotation when evidence
                # exists (region_frame_evidence, computed once above) -- e.g.
                # "kitchen (Frame 8)" -- purely in the per-video evidence lines;
                # the graded answer path (sub_path_names/start_name/end_name
                # below) is untouched by this. Built alongside a plain
                # (frame-free) version of the same segments so the cof/cot
                # pair differs only in that annotation -- which one is used is
                # decided later by data_qwenvl.py's --cof-ratio.
                segments_cof = []
                segments_cot = []
                for cam_key, chunk in cam_chunks:
                    presented_idx = cam_to_presented_idx.get(cam_key)
                    if presented_idx is None:
                        continue
                    names_cof = []
                    names_cot = []
                    any_missing = False
                    for r in chunk:
                        name = habitat_obj.guess_room_name(scene_idx, route_idx, str(r))
                        if name is None:
                            any_missing = True
                            break
                        names_cot.append(name)
                        evidence = region_frame_evidence.get((cam_key, str(r)))
                        if evidence is not None:
                            raw_frame, video_total_frames = evidence
                            sampled_frame = get_sampled_frame_index(raw_frame, video_total_frames) + 1
                            names_cof.append(f"{name} (Frame {sampled_frame})")
                        else:
                            names_cof.append(name)
                    if any_missing:
                        continue
                    segments_cof.append((presented_idx, names_cof))
                    segments_cot.append((presented_idx, names_cot))
                segments_cof.sort(key=lambda x: x[0])
                segments_cot.sort(key=lambda x: x[0])

                sub_path_names = [habitat_obj.guess_room_name(scene_idx, route_idx, str(r))
                                  for r in sub_path]

                # Only attach a trace when every region along the route has a name;
                # a trace with a missing region would assert something unverifiable.
                if segments_cof and all(n is not None for n in sub_path_names):
                    variant_fn = random.choice(ROUTE_VARIANTS)
                    qa_pair['reasoning_cof'] = build_route_trace(
                        segments=segments_cof,
                        sub_path_names=sub_path_names,
                        start_name=start_name,
                        end_name=end_name,
                        variant_fn=variant_fn,
                    )
                    qa_pair['reasoning_cot'] = build_route_trace(
                        segments=segments_cot,
                        sub_path_names=sub_path_names,
                        start_name=start_name,
                        end_name=end_name,
                        variant_fn=variant_fn,
                    )
                    qa_pair['reasoning'] = qa_pair['reasoning_cof']

                    # Process reward ground truth: the intermediate region names
                    # along the correct route (sub_path_names, excluding start/end
                    # since those are already given verbatim in the question
                    # itself -- rewarding a model for repeating the question back
                    # wouldn't test any reasoning). String substring match, same
                    # pattern as Kinematic Comparison's object-name check.
                    middle_names = sub_path_names[1:-1]
                    qa_pair['process_gt'] = {
                        f"hop_{i + 1}": name for i, name in enumerate(middle_names)
                    }

            generated_qas.append(qa_pair)

    return generated_qas


def generate_route_plan_qa(verbose=True, save=False, split='val',
                           root_path=DEFAULT_ROOT_PATH, output=None,
                           include_reasoning=None, total_num=None):

    if include_reasoning is None:
        include_reasoning = (split == 'train')

    habitat_obj = Habitat_Dataset(root_path=root_path, section='route_plan', split=split)
    qa_list = list()

    for scene_idx in os.listdir(habitat_obj.data_path):
        if total_num is not None and len(qa_list) >= total_num:
            break

        scene_path = os.path.join(habitat_obj.data_path, scene_idx)

        if not os.path.isdir(scene_path):
            continue

        for route_idx in os.listdir(scene_path):
            if total_num is not None and len(qa_list) >= total_num:
                break

            route_path = os.path.join(scene_path, route_idx)

            if not os.path.isdir(route_path):
                continue

            qa_list.extend(generate_single_route_plan_qa(habitat_obj, scene_idx,
                                                         route_idx, mode="medium",
                                                         include_reasoning=include_reasoning))

            if verbose:
                print(f"Processed Scene {scene_idx}, Route {route_idx}. Total QA pairs so far: {len(qa_list)}")


    # Distractors are chosen over the whole list (see reoption_route), which also
    # drops items whose start and goal regions share a description or whose
    # options cannot be made distinct.
    pools = [row.pop('_route_pool') for row in qa_list]
    rebuilt, _ = reoption_route_v3([row['answer'][3:] for row in qa_list], pools, random)
    kept = []
    for row, (options, answer) in zip(qa_list, rebuilt):
        if options is not None:
            row['options'], row['answer'] = options, answer
            kept.append(row)
    print(f"Route options rebuilt: kept {len(kept)}/{len(qa_list)} QA pairs")
    qa_list = kept

    if save:
        if output:
            out_path = output
        elif split == 'train':
            out_path = 'json_files/train/Habitat_Route_Planning.jsonl'
        else:
            out_path = 'json_files/eval/Holistic_Synthesis/Habitat_Route_Planning.jsonl'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_jsonl_file(filepath=out_path, data_list=qa_list)
        print(f"Saved {len(qa_list)} samples to {out_path} (split={split})")

    return qa_list

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Habitat Holistic Synthesis QA pairs (route planning / object counting).")

    parser.add_argument("--task", type=str, default='route_plan',
                        choices=['route_plan', 'object_counting'],
                        help="Which Holistic Synthesis task to generate.")
    parser.add_argument("--split", type=str, default='val', choices=['train', 'val'],
                        help="HM3D split to draw from. 'train' emits reasoning traces "
                             "for SFT; 'val' reproduces the benchmark.")
    parser.add_argument("--root-path", type=str, default=DEFAULT_ROOT_PATH,
                        help="Root of the rendered Habitat data tree.")
    parser.add_argument("--output", type=str, default=None,
                        help="Explicit output .jsonl path.")
    parser.add_argument("--section", type=str, default='route_plan',
                        choices=['general', 'route_plan', 'both'],
                        help="Data section for object counting. 'both' pools general + "
                             "route_plan for more training volume (train split only -- "
                             "the benchmark should keep a single section to match the "
                             "shipped dataset's composition).")
    parser.add_argument("--total-num", type=int, default=None,
                        help="Cap the number of QA pairs generated. Unbounded (full "
                             "natural yield) if omitted -- used for the benchmark, "
                             "where the exact count is a fixed property of the dataset.")

    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for sampling, trace variants and distractors.")

    args = parser.parse_args()
    random.seed(args.seed)
    task = args.task
    total_num = args.total_num if args.total_num is not None else float('inf')

    if task == 'route_plan':
        qa_list = generate_route_plan_qa(save=True, split=args.split,
                                         root_path=args.root_path, output=args.output,
                                         total_num=args.total_num)
    else:
        qa_list = generate_object_counting_qa(section=args.section, save=True,
                                              split=args.split, root_path=args.root_path,
                                              output=args.output, total_num=total_num)

    if qa_list:
        print(f"Generated {len(qa_list)} QA pairs. Here is the first one:")
        for key, val in qa_list[0].items():
            if key == 'options':
                [print(val_i) for val_i in val]
            else:
                print(key, val)
            print()
         
        if task == 'object_counting':
            count_dict = dict()   
            for item in qa_list:
                count_key = int(item['answer'].split(') ')[1].strip())
                count_dict[count_key] = count_dict.get(count_key, 0) + 1
                
            print(count_dict)
    else:
        print("No QA pairs were generated. Check dataset paths and contents.")