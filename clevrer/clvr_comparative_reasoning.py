from utils import write_jsonl_file
from clevrer.base_clevrer import *
import random
import argparse


class CLEVRER_Comparative_Reasoning(CLEVRER_Dataset):
    def __init__(self, root_path, split, sample_num):
        super().__init__(root_path, split, sample_num)

    def get_kinematic_comparison(self, idx, threshold=0.25):
        sample = dict()
        
        v1_path, v1_max, v1_obj, _ = self._get_velocity_item(idx)
        v2_path, v2_max, v2_obj, _ = self._get_velocity_item(idx + 1)
        
        sample['video_list'] = [
            {'video_path': v1_path},
            {'video_path': v2_path}
        ]
        
        sample['question'] = "Given two videos showing the same scene, identify which object has the highest peak velocity."
        
        if abs(v1_max - v2_max) < threshold:
            return None
        
        if v1_max > v2_max:
            correct_answer_raw = f"video_1 -- {v1_obj['color']} {v1_obj['material']} {v1_obj['shape']}"
        else:
            correct_answer_raw = f"video_2 -- {v2_obj['color']} {v2_obj['material']} {v2_obj['shape']}"

        options = []
        
        for v_idx, d_idx in enumerate([idx, idx + 1]):
            ann_dict = list(self.data_list[d_idx].values())[0]
            top_2_objects_dict = get_top_k_fastest_objects(ann_dict, k=2)
            vel_list = list()
            for obj_vel_dict in top_2_objects_dict:
                obj = obj_vel_dict['object_name']
                vel = obj_vel_dict['max_vel']
                options.append(f"video_{v_idx + 1} -- {obj}")
                vel_list.append(vel)
            
            if abs(vel_list[0] - vel_list[1]) < threshold:
                return None
        
        random.shuffle(options)
        
        sample['options'], sample['answer'] = add_alphabet_to_options(options, correct_answer_raw)
        
        return sample
    

    def get_collision_count_QA(self, idx, nframes=32, k=6, time_mode='frame'):
        k = min(max(2, k), 6)
        sample = dict()
        
        base_video, base_collisions = self._get_collision_item(idx)
        
        items_collected = [(base_video, idx, base_collisions)]
        seen_videos = {base_video}
        while len(items_collected) < k:
            if not self.collision_dict:
                break 
                
            c_len = random.choice(list(self.collision_dict.keys()))
            c_idx, c_ann_name = random.choice(self.collision_dict[c_len])
            
            if len(self.collision_dict[c_len]) == 0:
                del self.collision_dict[c_len]
                
            c_vid, c_cols = self._get_collision_item(c_idx)
            if c_vid in seen_videos:
                continue
            seen_videos.add(c_vid)
            items_collected.append((c_vid, c_idx, c_cols))
            
        actual_k = len(items_collected)
        random.shuffle(items_collected)
        
        video_list = []
        collision_counts = []
        video_stats = [] 
        breakdown_lines = [] 
        
        time_unit_label = "timestamp(s)" if time_mode == 'timestamp' else "Frame(s)"
        
        for i, (vid, item_idx, cols) in enumerate(items_collected):
            v_name = f"Video {i+1}"
            v_cnt = len(cols)
            
            video_list.append(vid)
            collision_counts.append(v_cnt)
            video_stats.append({'name': v_name, 'count': v_cnt})
            
            time_strs = [format_time_string(get_time_value(col['frame_id'], time_mode, nframes), time_mode).replace("Frame ", "") for col in cols]
            breakdown_lines.append(f"- {v_name} has {v_cnt} collision(s) at {time_unit_label} [{', '.join(time_strs)}].")
            
        sample['collision_count'] = collision_counts
        sample['video_list'] = [{'video_path': video} for video in video_list]
        
        video_stats.sort(key=lambda x: x['count'], reverse=True)
        max_count = video_stats[0]['count']
        second_count = video_stats[1]['count']
        
        tie_option_str = "most and second most have the same number of collisions -- 0"
        
        if max_count == second_count:
            correct_answer = tie_option_str
            fake_diffs = random.sample(range(1, 6), 4)
            options = [correct_answer]
            for i in range(4):
                vid_name = video_stats[i % actual_k]['name']
                options.append(f"{vid_name} -- {fake_diffs[i]}")
                
        else:
            winner_name = video_stats[0]['name']
            diff = max_count - second_count
            second_place_videos = [v['name'] for v in video_stats if v['count'] == second_count]
            second_place_str = ", ".join(second_place_videos)
            
            correct_answer = f"{winner_name} -- {diff}"
            
            near_diffs = {diff - 1, diff + 1, diff + 2, diff + 3}
            near_diffs = list({d for d in near_diffs if d > 0 and d != diff})
            
            while len(near_diffs) < 2:
                near_diffs.append(max(near_diffs) + 1 if near_diffs else 1)
                
            fake_diff_1, fake_diff_2 = random.sample(near_diffs, 2)
            second_name = second_place_videos[0]
            
            options = [
                correct_answer, tie_option_str, f"{second_name} -- {diff}",        
                f"{winner_name} -- {fake_diff_1}", f"{second_name} -- {fake_diff_2}"  
            ]
        
        sample['question'] = (
            f"Compare the total number of collisions across the {actual_k} videos. "
            f"Which video has the most collisions, and what is the numerical difference "
            f"in collisions between this video and the video with the second most collisions?\n"
            f"Select the correct option below."
        )
        
        random.shuffle(options)
        sample['options'], sample['answer'] = add_alphabet_to_options(options, correct_answer)
        
        return sample
    

    def get_collision_time_QA(self, idx, k=4, nframes=32, mode='earliest', time_mode='frame'):
        assert mode in ['earliest', 'latest'], "mode must be either 'earliest' or 'latest'"
        time_func = min if mode == 'earliest' else max
        fps = 25.0
        
        items_collected = []
        unique_times = set()
        collected_vid_times = [] 
        item_idx = idx
        
        while len(items_collected) < k:
            item = self._get_collision_item(item_idx)
            vid_time = time_func([get_time_value(col['frame_id'], time_mode, nframes, fps) for col in item[1]])
            
            if vid_time not in unique_times:
                unique_times.add(vid_time)
                items_collected.append(item)
                collected_vid_times.append(vid_time)
            
            item_idx += 1
                
        combined = list(zip(items_collected, collected_vid_times))
        random.shuffle(combined)
        items_collected, vid_times = zip(*combined)
        
        sample = dict()
        sample['video_list'] = [{'video_path': item[0]} for item in items_collected]
        sample['question'] = f"Given {k} videos showing the same scene, identify which video has the {mode} collision event."
        
        target_time = time_func(vid_times)
        target_idx = vid_times.index(target_time)
        
        correct_option_str = f"Video {target_idx+1}: {format_time_string(target_time, time_mode)}"
        options_set = {correct_option_str}
        target_num_options = max(4, k)
        
        video_ids = list(set(range(k)) - {target_idx})
        video_id_pool = video_ids.copy() + video_ids.copy()
        video_id_pool.append(target_idx)
        used_gaps = {target_time}
        
        while len(options_set) < target_num_options:
            vid_id = random.choice(video_id_pool) if len(video_ids) == 0 else video_ids.pop(0)
            
            while True:
                offset = random.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
                
                offset_val = offset if time_mode == 'frame' else round(offset * (1.0 / fps), 1)
                fake_time = round(target_time + offset_val, 1)
                
                if fake_time >= 0 and fake_time not in used_gaps:
                    break
                    
            used_gaps.add(fake_time)
            options_set.add(f"Video {vid_id+1}: {format_time_string(fake_time, time_mode)}")
            
        sample['options'] = list(options_set)    
        answer_text = correct_option_str
        sample['options'], sample['answer'] = add_alphabet_to_options(sample['options'], answer_text)
        
        return sample


    def get_collision_duration_QA(self, idx, nframes=32, k=2, time_mode='frame'):
        assert k >= 2, "k must be at least 2 to compare durations."
        fps = 25.0
        
        items_collected = list()
        unique_durations = set()
        item_idx = idx
        
        while len(items_collected) < k:
            item = self._get_collision_item(item_idx)
            times = [get_time_value(col['frame_id'], time_mode, nframes, fps) for col in item[1]]
            
            if not times:
                item_idx += 1
                continue
                
            min_t = min(times)
            max_t = max(times)
            duration = round(max_t - min_t, 1)
            
            if duration not in unique_durations:
                items_collected.append({
                    'video_id': item[0],
                    'min_t': min_t,
                    'max_t': max_t,
                    'duration': duration
                })
                unique_durations.add(duration)
                
            item_idx += 1
            
        random.shuffle(items_collected)
        
        sample = dict()
        sample['video_list'] = [item['video_id'] for item in items_collected]
        
        vid_names = [f"Video {i+1}" for i in range(k)]
        vid_str = "Video 1 and Video 2" if k == 2 else ", ".join(vid_names[:-1]) + f", and {vid_names[-1]}"
        unit_str = "seconds" if time_mode == "timestamp" else "frames"
            
        sample['question'] = f"Analyze the timeline of collisions in {vid_str}. Calculate the time gap (in {unit_str}) between the very first collision and the very last collision in each video. Which video has the longest duration between its first and last collision?"
        
        actual_durations = []
        
        for i, item in enumerate(items_collected):
            actual_durations.append(item['duration'])
            
        target_duration = max(actual_durations)
        target_idx = actual_durations.index(target_duration)
        correct_answer_str = f"Video {target_idx+1} -- {target_duration} {unit_str}"

        options_set = {correct_answer_str}
        used_gaps = {target_duration}
        
        target_num_options = max(4, k) 
        video_ids = list(set(range(k)) - {target_idx})
        video_id_pool = video_ids.copy() + video_ids.copy()
        video_id_pool.append(target_idx)
        
        while len(options_set) < target_num_options:
            vid_id = random.choice(video_id_pool) if len(video_ids) == 0 else video_ids.pop(0)
            
            while True:
                offset = random.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
                
                offset_val = offset if time_mode == 'frame' else round(offset * (1.0 / fps), 1)
                fake_gap = round(target_duration + offset_val, 1)
                
                if fake_gap >= 0 and fake_gap not in used_gaps:
                    break
            used_gaps.add(fake_gap)
            options_set.add(f"Video {vid_id+1} -- {fake_gap} {unit_str}")

        sample['options'] = list(options_set)
        random.shuffle(sample['options']) 
        sample['answer'] = correct_answer_str
        
        sample['options'], sample['answer'] = add_alphabet_to_options(sample['options'], sample['answer'])
        
        return sample

def generate_test_QA(root_path, qa_num=200, total_num=200, k=3, split='eval', verbose=False):
    dataset = CLEVRER_Comparative_Reasoning(
        root_path=root_path, 
        split='validation', 
        sample_num=total_num
    )
    
    data_list = []
    for idx in range(qa_num):
        data_list.append(dataset.get_seq_order_QA(idx=idx, k=3))

    if verbose:
        for sample in data_list:
            for key, val in sample.items():
                print(key, val)
            print('---------------------'*5)
    else:
        random.shuffle(data_list)
        output_path = f"json_files/{split}/mv_seq_order_dataset.jsonl"
        write_jsonl_file(filepath=output_path, data_list=data_list)
        print(f"Successfully saved {len(data_list)} samples to {output_path}")


def generate_comparative_qa(args, dataset):
    data_list = []
    data_idx = 0
    
    while len(data_list) < args.num_samples and len(dataset.data_list) > data_idx + 1:
        
        if args.task_type == 'collision':
            # Utilizing new argparse inputs
            data = dataset.get_collision_count_QA(
                data_idx, 
                nframes=args.nframes, 
                k=args.k, 
                time_mode=args.time_mode
            )
        else:
            # Utilizing new argparse inputs
            data = dataset.get_kinematic_comparison(
                data_idx, 
                threshold=args.threshold
            )
        data_idx += 1
            
        if data:
            data_list.append(data)
            
    print(f"Total processed samples: {len(data_list)}")    
    if data_list:
        print(data_list[0])
        
    if args.task_type == 'collision':
        eval_path = "json_files/Comparative_Reasoning/Clvr_Numerical_Comparison.jsonl"
    else:
        eval_path = "json_files/Comparative_Reasoning/Clvr_Kinematic_Comparison.jsonl"

    if args.save:
        write_jsonl_file(eval_path, data_list)
        print(f"Successfully saved file to: {eval_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multimodal video models.")

    # Fixed missing required arguments that were previously unmapped
    parser.add_argument("--root-path", type=str, required=True, 
                        help="Root path for the CLEVRER dataset.")
    parser.add_argument("--num-samples", type=int, default=100, 
                        help="Number of comparative QA samples to generate.")
    parser.add_argument("--total-num", type=int, required=True, 
                        help="Total number of samples to initialize dataset indexing.")
    parser.add_argument("--task-type", type=str, choices=['collision', 'kinematic'],
                        required=True, help="Task type to generate.")
    parser.add_argument("--split", type=str, default='validation', 
                        help="Split of the dataset to use (e.g. validation, train, test).")
    
    # New customizable arguments previously hardcoded
    parser.add_argument("--k", type=int, default=2, 
                        help="Number of videos/objects to compare (k value).")
    parser.add_argument("--nframes", type=int, default=32, 
                        help="Number of frames for collision data.")
    parser.add_argument("--time-mode", type=str, default='timestamp', choices=['timestamp', 'frame'], 
                        help="Time mode setting for collision tracking.")
    parser.add_argument("--threshold", type=float, default=0.25, 
                        help="Threshold for kinematic max velocity comparison.")
    
    # Using action=store_true for boolean flag
    parser.add_argument("--save", action="store_true", 
                        help="Include this flag to save outputs to JSONL.")
    
    args = parser.parse_args()

    dataset = CLEVRER_Comparative_Reasoning(
            root_path=args.root_path, 
            split=args.split, 
            sample_num=args.total_num * 2
        )

    # Note: Removed the hardcoded 'save=True' to utilize `--save` CLI flag instead
    generate_comparative_qa(args, dataset)