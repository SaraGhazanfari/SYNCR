import os
import random
import argparse

from utils import write_jsonl_file
from habitat.base_habitat import *


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
            range_len = (end - start) + 1
            
            if range_len < min_range_len:
                continue
                
            if drop_boundary_ranges and start == 0:
                continue 
            
            valid_ranges.append((start, end))
            total_visible_frames += range_len
            
        if valid_ranges and total_visible_frames >= min_total_frames:
            filtered_data[cam] = valid_ranges

    if len(filtered_data) < 2:
        return {} 
        
    return filtered_data


def generate_single_object_reidentification_qa(scene_path, clean_obj_name, presence_data, ref_cam, 
                                               target_cam, fps=30, distractor_shift_sec=1.0):
    
    ref_ranges = presence_data.get(ref_cam, [])
    target_ranges = presence_data.get(target_cam, [])
    
    chosen_ref_range = random.choice(ref_ranges) if ref_ranges else None
    ref_time_str = format_single_range_to_sec(chosen_ref_range, fps)
    
    options_set = set()
    
    question = (
        f"You are provided with two videos recorded at {fps} FPS. "
        f"In Video 1, the '{clean_obj_name}' is clearly visible from {ref_time_str}. "
        f"Select the correct timestamp window for the FIRST appearance of the exact same '{clean_obj_name}' in Video 2."
    )
    
    if not target_ranges:
        correct_answer = "The object does not appear in this video."
        options_set.add(correct_answer)
        
        shift_frames = int(distractor_shift_sec * fps)
        duration_frames = int(2.0 * fps) 
        
        base_start = 0
        while len(options_set) < 4:
            distractor_str = format_single_range_to_sec((base_start, base_start + duration_frames), fps)
            options_set.add(distractor_str)
            base_start += shift_frames
            
    else:
        target_ranges.sort(key=lambda x: x[0])
        first_target_range = target_ranges[0]
        
        correct_answer = format_single_range_to_sec(first_target_range, fps)
        options_set.add(correct_answer)
        
        shift_frames = int(distractor_shift_sec * fps)
        multipliers = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
        
        for k in multipliers:
            if len(options_set) == 4:
                break
                
            d_start = first_target_range[0] + (k * shift_frames)
            d_end = first_target_range[1] + (k * shift_frames)
            
            if d_start >= 0:
                options_set.add(format_single_range_to_sec((d_start, d_end), fps))

    options_list_raw = list(options_set)[:4] 
    random.shuffle(options_list_raw)
    
    labels = ['A', 'B', 'C', 'D']
    options_formatted = []
    correct_label = ""
    
    for i, opt in enumerate(options_list_raw):
        formatted_str = f"{labels[i]}) {opt}"
        options_formatted.append(formatted_str)
        if opt == correct_answer:
            correct_label = formatted_str
            
    return {
        "question": question,
        "options": options_formatted,
        "answer": correct_label,
        "video_list": [
            {'video_path': os.path.join(scene_path, ref_cam, f'{ref_cam}.mp4')},
            {'video_path': os.path.join(scene_path, target_cam, f'{target_cam}.mp4')}
        ]
    }
     

def generate_object_reidentification_qa(root_path, fps=5, min_range_len=3, min_total_frames=5, save=False, verbose=True):
        
    qa_list = list()
    habitat_obj = Habitat_Dataset(root_path=root_path, section='general', split='val')
    scene_directories = os.listdir(habitat_obj.data_path)
    total_scenes = len(scene_directories)
    
    for idx, scene_idx in enumerate(scene_directories):
        scene_path = os.path.join(habitat_obj.data_path, scene_idx)
        print(f"Processing scene: {scene_idx} ({idx + 1}/{total_scenes}) | QA samples so far: {len(qa_list)}", end='\r')

        if not os.path.isdir(scene_path):
            continue
        
        common_obj_ids = habitat_obj.get_common_object_ids(scene_idx=scene_idx)
        
        for obj_id in common_obj_ids:
            full_object_name = habitat_obj.get_object_name(scene_idx=scene_idx, object_id=obj_id)
            
            if full_object_name == 'Unknown Object':
                continue
            
            object_name = ' '.join(full_object_name.split(' ')[1:-1])
            
            if object_name in objects_to_filter:
                continue
            
            presence_data = filter_presence_data(
                habitat_obj.get_timely_object_presence(scene_idx=scene_idx, object_id=obj_id),
                min_range_len=min_range_len, 
                min_total_frames=min_total_frames, 
                drop_boundary_ranges=False
            )
            
            cam_list = list(presence_data.keys())
            for ref_idx in range(len(cam_list) - 1):
                for target_idx in range(ref_idx + 1, len(cam_list)):
                    qa_list.append(generate_single_object_reidentification_qa(
                        scene_path=habitat_obj._get_scene_path(scene_idx),
                        clean_obj_name=object_name, 
                        presence_data=presence_data, 
                        ref_cam=cam_list[ref_idx], 
                        target_cam=cam_list[target_idx], 
                        fps=fps
                    ))
                           
    print(f"\nFinished processing. Total QA pairs generated: {len(qa_list)}")
    
    if verbose and qa_list:
        print("\n" + "-"*30 + " Sample Preview " + "-"*30)
        for key, val in qa_list[0].items():
            if key == 'options':
                print(f"{key}:")
                for opt in val:
                    print(f"  {opt}")
            else:
                print(f"{key}: {val}")
        print("-" * 76)
            
    if save:
        eval_path = 'json_files/Spatial_Tracking/Habitat_Object_Re_identification.jsonl'
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
        write_jsonl_file(filepath=eval_path, data_list=qa_list)
        print(f"File successfully saved to: {eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA pairs for Habitat Object Re-identification.")
    
    parser.add_argument("--root-path", type=str, required=True, 
                        help="Root path for the Habitat dataset.")
    parser.add_argument("--fps", type=int, default=5, 
                        help="Frames per second for time conversions.")
    parser.add_argument("--min-range-len", type=int, default=3, 
                        help="Minimum frames a single continuous range must have.")
    parser.add_argument("--min-total-frames", type=int, default=5, 
                        help="Minimum total frames the object must be visible.")
    
    parser.add_argument("--save", action="store_true", 
                        help="Save the output list to JSONL.")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Enable verbose terminal output to preview the first sample.")
    
    args = parser.parse_args()

    # Pass an empty set for objects_to_filter by default, 
    # or populate it if you intend to add a CLI arg for it later.
    generate_object_reidentification_qa(
        root_path=args.root_path,
        objects_to_filter=set(),
        fps=args.fps,
        min_range_len=args.min_range_len,
        min_total_frames=args.min_total_frames,
        save=args.save,
        verbose=args.verbose
    )