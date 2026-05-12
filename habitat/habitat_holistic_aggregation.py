import os
import random
import itertools
import argparse
import numpy as np
import networkx as nx

from utils import write_jsonl_file, read_json_file
from habitat.base_habitat import *


def rebalance_qa_list(qa_list):
    group_1 = []
    group_others = []
    target_count = 0
    
    for item in qa_list:
        try:
            count_key = int(item['answer'].split(') ')[1].strip())
            
            if count_key == 1:
                group_1.append(item)
            elif count_key >= 2:
                group_others.append(item)
                if count_key == 2:
                    target_count += 1
                
        except (IndexError, ValueError):
            continue
            
    print(f"Total samples with count 1 before subsampling: {len(group_1)}")
    
    if len(group_1) > target_count:
        balanced_1 = random.sample(group_1, target_count)
        print(f"Subsampled 1s down to {target_count}.")
    else:
        balanced_1 = group_1
        print("Note: You have fewer 1s than >=2s, so all 1s were kept.")

    balanced_qa_list = balanced_1 + group_others
    random.shuffle(balanced_qa_list)
    
    # Moved print above return so it is no longer unreachable code
    print(f"New balanced dataset size: {len(balanced_qa_list)}")
    return balanced_qa_list


def generate_single_object_counting_qa(habitat_obj, scene_idx, route_idx=None):
        
    qa_list = list()    
    processed_object_info_dict, aggregated_object_count_dict = habitat_obj.get_scene_object_info(scene_idx=scene_idx, route_idx=route_idx)
    
    video_framewise_object_ids_list, video_path_list = habitat_obj._get_raw_item(scene_idx, route_idx)
    num_videos = len(video_path_list)
    
    if num_videos == 0:
        return qa_list 
        
    video_dicts = [{'video_path': path} for path in video_path_list]
    
    for object_name, count in aggregated_object_count_dict.items():
        if count > 10 or object_name in objects_to_filter:
            continue
            
        target_ids = []
        for obj_id, val_str in processed_object_info_dict.items():
            parts = val_str.split(' ')
            if len(parts) >= 3:
                name = ' '.join(parts[1:-1])
                if name == object_name:
                    target_ids.append(obj_id)
                    
        all_instances_visible = True
        
        for t_id in target_ids:
            instance_passed = False
            
            for semantic_video in video_framewise_object_ids_list:
                total_pixels = semantic_video.shape[1] * semantic_video.shape[2]
                threshold_pixels = total_pixels * 0.05
                
                flat_view = semantic_video.reshape((semantic_video.shape[0], -1))
                pixel_counts_per_frame = np.sum(flat_view == t_id, axis=1)
                
                if np.max(pixel_counts_per_frame) >= threshold_pixels:
                    instance_passed = True
                    break 
            
            if not instance_passed or len(video_framewise_object_ids_list) == 1:
                all_instances_visible = False
                break 
                
        if not all_instances_visible:
            continue
            
        clean_obj_name = object_name.replace('_', ' ')
        
        question = (
            f"Based on the combined exploration in all {num_videos} videos, "
            f"how many distinct instances of '{clean_obj_name}' are present in this scene?"
        )
        
        correct_answer = str(count)
        options = [correct_answer]
        
        while len(options) < 4:
            offset = random.randint(-3, 3)
            distractor = count + offset
            if distractor >= 0 and str(distractor) not in options:
                options.append(str(distractor))
        
        random.shuffle(options)
        labels = ['A', 'B', 'C', 'D']
        option_list = [f"{labels[i]}) {opt}" for i, opt in enumerate(options)]
        
        correct_label = ""
        for opt in option_list:
            if opt.split(') ')[1] == correct_answer:
                correct_label = opt
                break
        
        qa_pair = {
            "question": question,
            "options": option_list,
            "answer": correct_label,
            "video_list": video_dicts
        }
        
        qa_list.append(qa_pair)
        
    return qa_list


def generate_object_counting_qa(root_path, total_num=float('inf'), section='general', save=False):
    habitat_obj = Habitat_Dataset(root_path=root_path, section=section, split='val') 
    qa_list = list()
    
    def save_progress(qa_list):
        if save:        
            eval_path = 'json_files/Holistic_Aggregation/Habitat_Object_Counting.jsonl'
            os.makedirs(os.path.dirname(eval_path), exist_ok=True)
            write_jsonl_file(filepath=eval_path, data_list=qa_list)
        
    for scene_idx in os.listdir(habitat_obj.data_path):
        scene_path = os.path.join(habitat_obj.data_path, scene_idx)
        
        if not os.path.isdir(scene_path):
            continue
            
        if habitat_obj.data_type == 'general':
            qa_list.extend(generate_single_object_counting_qa(
                habitat_obj, scene_idx, route_idx=None))
            print(f"Processed Scene {scene_idx}. Total QA pairs so far: {len(qa_list)}", end='\r')
        else:
            for route_idx in os.listdir(scene_path):
                route_path = os.path.join(scene_path, route_idx)
                if os.path.isdir(route_path):
                    qa_list.extend(generate_single_object_counting_qa(
                        habitat_obj, scene_idx, route_idx=route_idx))
                    print(f"Processed Scene {scene_idx}, Route {route_idx}. Total QA pairs so far: {len(qa_list)}", end='\r')
                    
        if len(qa_list) >= total_num:
            break
        
        save_progress(qa_list)
        
    print(f"\nFinished processing. Initial dataset size: {len(qa_list)}")
    qa_list = rebalance_qa_list(qa_list)
    save_progress(qa_list) 
        
    return qa_list


def generate_single_route_plan_qa(habitat_obj, scene_idx, route_idx, mode="all"):
    """
    mode: "all" (returns QA pairs for every sub-path >= 3 nodes) 
          "longest" (returns only the single QA pair for the maximum path)
          "medium" (returns QA pairs for length max(3, len)-1 to len)
    """
    scene_path = habitat_obj._get_scene_path(scene_idx, route_idx)
    routes_file = os.path.join(scene_path, "video_routes.json")
    
    routes_dict = read_json_file(routes_file) 
    
    video_paths = habitat_obj._get_video_path(scene_idx, route_idx)
    for vp in video_paths:
        if not os.path.exists(vp):
            print(f"Warning: Missing video file {vp}. Skipping this route.")
            return [] 

    random.shuffle(video_paths)
    video_list_formatted = [{'video_path': vp} for vp in video_paths]
    
    full_path = []
    num_cams = len(routes_dict)
    
    for i in range(num_cams):
        cam_key = f"cam{i}"
        if cam_key not in routes_dict:
            return [] 
            
        chunk = routes_dict[cam_key]
        if not full_path:
            full_path.extend(chunk)
        else:
            full_path.extend(chunk[1:])
            
    if len(full_path) < 3:
        return [] 

    all_nodes_in_video = list(set(full_path))
    generated_qas = []
    
    if mode == "longest":
        lengths_to_process = [len(full_path)]
    elif mode == "medium":
        lengths_to_process = range(max(3, len(full_path)) - 1, len(full_path) + 1)
    else: # Default to all
        lengths_to_process = range(3, len(full_path) + 1)

    for length in lengths_to_process:
        for start_idx in range(len(full_path) - length + 1):
            sub_path = full_path[start_idx : start_idx + length]
            
            start_node = sub_path[0]
            end_node = sub_path[-1]
            true_middle = sub_path[1:-1]
            
            pool_mids = true_middle.copy()
            unused_video_nodes = [n for n in all_nodes_in_video if n not in sub_path]
            pool_mids.extend(unused_video_nodes)
            
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
                    
            numeric_options_set = set()
            max_r = min(len(pool_mids), 3) 
            
            for r in range(0, max_r + 1):
                for perm in itertools.permutations(pool_mids, r):
                    candidate_path = [start_node] + list(perm) + [end_node]
                    numeric_options_set.add(tuple(candidate_path))
                    
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

            correct_tuple = tuple(sub_path)
            correct_str = format_path(correct_tuple)
            if not correct_str:
                continue
            
            if correct_tuple in numeric_options_set:
                numeric_options_set.remove(correct_tuple)
                
            distractor_tuples = list(numeric_options_set)
            random.shuffle(distractor_tuples)
            selected_distractors = distractor_tuples[:3]
            
            final_options_strings = [correct_str] + [format_path(tup) for tup in selected_distractors]
            random.shuffle(final_options_strings)
            
            if len(final_options_strings) < 4:
                continue 
                
            labels = ['A', 'B', 'C', 'D']
            options_list = [f"{labels[i]}) {opt}" for i, opt in enumerate(final_options_strings)]
            correct_label = [opt for opt in options_list if opt.split(')')[1].strip() == correct_str][0]
            
            start_name = habitat_obj.guess_room_name(scene_idx, route_idx, str(start_node))
            end_name = habitat_obj.guess_room_name(scene_idx, route_idx, str(end_node))
            question = (
                f"Based on the spatial layout shown across the videos, what is the shortest possible "
                f"path to travel from Region 1 <{start_name}> to Region 2 <{end_name}>?"
            )
            
            generated_qas.append({
                "scene_idx": scene_idx,
                "route_idx": route_idx,
                "question_type": "unseen_shortest_path",
                "question": question,
                "options": options_list,
                "answer": correct_label,
                "video_list": video_list_formatted
            })
            
    return generated_qas


def generate_route_plan_qa(root_path, mode="medium", verbose=True, save=False):
    habitat_obj = Habitat_Dataset(root_path=root_path, section='route_plan', split='val')  
    qa_list = list()
    
    for scene_idx in os.listdir(habitat_obj.data_path):
        scene_path = os.path.join(habitat_obj.data_path, scene_idx)
        
        if not os.path.isdir(scene_path):
            continue
            
        for route_idx in os.listdir(scene_path):
            route_path = os.path.join(scene_path, route_idx)
            
            if not os.path.isdir(route_path):
                continue
                
            qa_list.extend(generate_single_route_plan_qa(habitat_obj, scene_idx, 
                                                         route_idx, mode=mode))
                
            if verbose:
                print(f"Processed Scene {scene_idx}, Route {route_idx}. Total QA pairs so far: {len(qa_list)}", end='\r')
                
    print(f"\nFinished processing. Total QA pairs generated: {len(qa_list)}")

    if save:        
        eval_path = 'json_files/Holistic_Aggregation/Habitat_Route_Planning.jsonl'
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
        write_jsonl_file(filepath=eval_path, data_list=qa_list)
        print(f"File successfully saved to: {eval_path}")
        
    return qa_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA pairs for Habitat Dataset.")
    
    parser.add_argument("--root-path", type=str, required=True, 
                        help="Root path for the Habitat dataset.")
    parser.add_argument("--task", type=str, choices=["object_counting", "route_plan"], required=True,
                        help="Select which generation task to run.")
    
    # Optional arguments for route planning
    parser.add_argument("--route-mode", type=str, choices=["all", "longest", "medium"], default="medium", 
                        help="Mode for determining path length targets in route_plan task.")
    
    # Optional arguments for object counting
    parser.add_argument("--total-num", type=int, default=10000, 
                        help="Maximum number of QA pairs to generate (mostly for object counting).")
    parser.add_argument("--section", type=str, default="general", 
                        help="Data section to initialize Habitat dataset with.")
    
    # Global flags
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose terminal output.")
    parser.add_argument("--save", action="store_true", 
                        help="Save the output list to JSONL.")
    
    args = parser.parse_args()

    if args.task == "route_plan":
        qa_list = generate_route_plan_qa(
            root_path=args.root_path, 
            mode=args.route_mode, 
            verbose=args.verbose, 
            save=args.save
        )
    elif args.task == "object_counting":
        qa_list = generate_object_counting_qa(
            root_path=args.root_path, 
            total_num=args.total_num, 
            section=args.section, 
            save=args.save
        )