'''
>> Comparative & Contrastive Reasoning 

    This capability involves analyzing two or more videos to identify 
    similarities, differences, improvements, or anomalies.
    To do this effectively, the model must be able to hold the mechanics 
    of one video in its working memory while evaluating the second.
'''

from utils import DATA_ROOT
import os
import json
from utils import write_jsonl_file
from generation.clevrer.base_clevrer import *
from generation.option_balance import reoption_numerical
import random
import argparse
import collections


def _frame_clause(value, lead):
    """value=None (the plain-CoT counterpart to a chain-of-frames/chain-of-
    timestamps trace, see get_kinematic_comparison) omits the clause
    entirely rather than printing a literal 'reached at None'. value is a
    pre-formatted string ('Frame 12' or '2.3s'), not a bare number -- lead
    is just the connector, no unit baked in, so the same variant functions
    serve the frame- and timestamp-grounded traces alike."""
    return f"{lead}{value}" if value is not None else ""


def _kinematic_v0(v1_max, v1_name, v1_frame, v2_max, v2_name, v2_frame, winner_label):
    return (
        f"- Video 1 peak velocity: {v1_max:.2f} (Object: {v1_name}){_frame_clause(v1_frame, ', reached at ')}\n"
        f"- Video 2 peak velocity: {v2_max:.2f} (Object: {v2_name}){_frame_clause(v2_frame, ', reached at ')}\n\n"
        f"Comparison: {winner_label} shows a higher peak velocity."
    )


def _kinematic_v1(v1_max, v1_name, v1_frame, v2_max, v2_name, v2_frame, winner_label):
    return (
        f"The {v1_name} in Video 1 reaches a peak speed of {v1_max:.2f}{_frame_clause(v1_frame, ' at ')}, while the "
        f"{v2_name} in Video 2 peaks at {v2_max:.2f}{_frame_clause(v2_frame, ' at ')}. Comparing the two peak speeds "
        f"directly, {winner_label} is the faster of the two."
    )


def _kinematic_v2(v1_max, v1_name, v1_frame, v2_max, v2_name, v2_frame, winner_label):
    return (
        f"**Peak velocities:** Video 1 = {v1_max:.2f} ({v1_name}{_frame_clause(v1_frame, ', ')}); "
        f"Video 2 = {v2_max:.2f} ({v2_name}{_frame_clause(v2_frame, ', ')})\n"
        f"**Faster:** {winner_label}"
    )


KINEMATIC_VARIANTS = [_kinematic_v0, _kinematic_v1, _kinematic_v2]


def _numerical_v0(breakdown_lines, reasoning_conclusion):
    return "\n\n".join(breakdown_lines) + "\n\n" + reasoning_conclusion


def _numerical_v1(breakdown_lines, reasoning_conclusion):
    body = " ".join(l.lstrip('- ') for l in breakdown_lines)
    return f"Looking at the collision counts across videos: {body} {reasoning_conclusion}"


def _numerical_v2(breakdown_lines, reasoning_conclusion):
    body = "\n".join(f"  {l.lstrip('- ')}" for l in breakdown_lines)
    return f"**Collision counts:**\n{body}\n**Result:** {reasoning_conclusion}"


NUMERICAL_VARIANTS = [_numerical_v0, _numerical_v1, _numerical_v2]


class CLEVRER_Comparative_Reasoning(CLEVRER_Dataset):
    def __init__(self, root_path, split, sample_num):
        super().__init__(root_path, split, sample_num)

    def get_kinematic_comparison(self, idx, threshold=0.25):
        """Video idx paired with video idx + 1, as the benchmark was built."""
        return self.get_kinematic_comparison_pair(idx, idx + 1, threshold=threshold)

    def get_kinematic_comparison_pair(self, idx_a, idx_b, threshold=0.25):
        sample = dict()

        # 1. Retrieve data for both videos
        v1_path, v1_max, v1_obj, v1_frame_id = self._get_velocity_item(idx_a)
        v2_path, v2_max, v2_obj, v2_frame_id = self._get_velocity_item(idx_b)
        
        # 2. Structure video_list
        sample['video_list'] = [
            {'video_path': v1_path},
            {'video_path': v2_path}
        ]
        
        sample['question'] = "Given two videos showing the same scene, identify which object has the highest peak velocity."
        
        if abs(v1_max - v2_max) < threshold:
            return None
        
        # 3. Determine the correct raw answer string
        if v1_max > v2_max:
            correct_answer_raw = f"video_1 -- {v1_obj['color']} {v1_obj['material']} {v1_obj['shape']}"
        else:
            correct_answer_raw = f"video_2 -- {v2_obj['color']} {v2_obj['material']} {v2_obj['shape']}"

        options = []
        
        for v_idx, d_idx in enumerate([idx_a, idx_b]):
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
        
        # 5. Format with Alphabet labels
        sample['options'], sample['answer'] = add_alphabet_to_options(options, correct_answer_raw)
        
        # 6. Add Reasoning
        # Chain-of-frames: peak-velocity frame indices are computed above by
        # get_fastest_speed but were previously discarded -- these videos are
        # used whole (no video_start/video_end trimming, see video_list
        # above), so get_sampled_frame_index's whole-video formula applies
        # directly. +1 for a human-readable 1-indexed frame number, matching
        # the convention build_permanence_trace/build_collision_trace already
        # use in clvr_temporal_alignment.py.
        v1_frame_str = f"Frame {get_sampled_frame_index(v1_frame_id) + 1}"
        v2_frame_str = f"Frame {get_sampled_frame_index(v2_frame_id) + 1}"
        # Chain-of-timestamps: seconds into the (whole, untrimmed) video --
        # sampling-density-independent unlike the frame index above, see
        # build_permanence_trace's use_timestamps docstring in
        # clvr_temporal_alignment.py for why that matters.
        v1_time_str = f"{round(v1_frame_id / 25.0, 1)}s"
        v2_time_str = f"{round(v2_frame_id / 25.0, 1)}s"
        winner_label = 'Video 1' if v1_max > v2_max else 'Video 2' if v2_max > v1_max else 'Both'
        v1_name = f"{v1_obj['color']} {v1_obj['shape']}"
        v2_name = f"{v2_obj['color']} {v2_obj['shape']}"
        # Same style for all three so the cof/cotime/cot triple differs only
        # in the frame/timestamp references (see _frame_clause) -- which one
        # is used in the final training set is decided later by
        # data_qwenvl.py's --cof-ratio/--cotime-ratio.
        variant = random.choice(KINEMATIC_VARIANTS)
        sample['reasoning_cof'] = variant(v1_max, v1_name, v1_frame_str, v2_max, v2_name, v2_frame_str, winner_label)
        sample['reasoning_cotime'] = variant(v1_max, v1_name, v1_time_str, v2_max, v2_name, v2_time_str, winner_label)
        sample['reasoning_cot'] = variant(v1_max, v1_name, None, v2_max, v2_name, None, winner_label)
        sample['reasoning'] = sample['reasoning_cof']

        # Structured ground truth for the GRPO process reward (training code, not released):
        # the same per-video peak-velocity object names already used to build
        # the reasoning text above, exposed directly rather than only baked
        # into template-worded prose -- lets a reward function check whether
        # a model's own generated reasoning names the correct object per
        # video, without regex-parsing it back out of one of three
        # differently-worded templates (KINEMATIC_VARIANTS). Object names
        # only, not frame/timestamp: the GRPO dataset's GRPO prompts show
        # the model sampled frames as plain images with no frame-index or
        # timestamp labels (see that file's docstring for why -- TRL's
        # GRPOTrainer has no native video content-part support), so there is
        # nothing in the prompt for a frame/timestamp claim to be grounded
        # in, unlike the SFT chain-of-frames/chain-of-timestamps traces above.
        sample['process_gt'] = {
            'video_1_object': v1_name,
            'video_2_object': v2_name,
        }

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
        breakdown_lines_cof = []
        breakdown_lines_cotime = []
        breakdown_lines_cot = []

        time_unit_label = "timestamp(s)" if time_mode == 'timestamp' else "Frame(s)"

        for i, (vid, item_idx, cols) in enumerate(items_collected):
            v_name = f"Video {i+1}"
            v_cnt = len(cols)

            video_list.append(vid)
            collision_counts.append(v_cnt)
            video_stats.append({'name': v_name, 'count': v_cnt})

            # Chain-of-frames: append the sampled-frame index (the frame the model
            # actually sees, per the verified get_sampled_frame_index formula)
            # alongside whatever get_time_value already reports (timestamp in
            # production -- time_mode='timestamp' below -- which needs no
            # remapping itself, but pairs naturally with an explicit frame
            # reference). Videos here are used whole (no video_start/video_end in
            # video_list above), so the default total_frames=128 applies directly.
            # Built alongside a frame-free (plain-CoT) version and a genuine
            # chain-of-timestamps version (always real seconds, independent of
            # time_mode -- see build_permanence_trace's use_timestamps
            # docstring in clvr_temporal_alignment.py for why frame indices
            # don't transfer to eval-time sampling but timestamps do) so the
            # three breakdowns differ only in that annotation -- which one is
            # used is decided later by data_qwenvl.py's --cof-ratio/--cotime-ratio.
            time_strs_cof = []
            time_strs_cotime = []
            time_strs_cot = []
            for col in cols:
                time_str = format_time_string(get_time_value(col['frame_id'], time_mode, nframes), time_mode).replace("Frame ", "")
                sampled_frame = get_sampled_frame_index(col['frame_id'], nframes=nframes) + 1
                time_strs_cof.append(f"{time_str} (Frame {sampled_frame})")
                time_strs_cotime.append(f"{round(col['frame_id'] / 25.0, 1)}s")
                time_strs_cot.append(time_str)
            breakdown_lines_cof.append(f"- {v_name} has {v_cnt} collision(s) at {time_unit_label} [{', '.join(time_strs_cof)}].")
            breakdown_lines_cotime.append(f"- {v_name} has {v_cnt} collision(s) at timestamp(s) [{', '.join(time_strs_cotime)}].")
            breakdown_lines_cot.append(f"- {v_name} has {v_cnt} collision(s) at {time_unit_label} [{', '.join(time_strs_cot)}].")
            
        sample['collision_count'] = collision_counts
        sample['video_list'] = [{'video_path': video} for video in video_list]
        
        video_stats.sort(key=lambda x: x['count'], reverse=True)
        max_count = video_stats[0]['count']
        second_count = video_stats[1]['count']
        
        tie_option_str = "most and second most have the same number of collisions -- 0"
        
        if max_count == second_count:
            correct_answer = tie_option_str
            tied_videos = [v['name'] for v in video_stats if v['count'] == max_count]
            tied_str = ", ".join(tied_videos)
            reasoning_conclusion = f"Therefore, {tied_str} tie for the most collisions."
            
            # Distractors are chosen over the whole list by reoption_numerical in
            # generate_comparative_qa (a tie + video x diff grid); the old ones let
            # the answer share its video with one option and its diff with another.
            options = [correct_answer]

        else:
            winner_name = video_stats[0]['name']
            diff = max_count - second_count
            second_place_videos = [v['name'] for v in video_stats if v['count'] == second_count]
            second_place_str = ", ".join(second_place_videos)

            correct_answer = f"{winner_name} -- {diff}"
            reasoning_conclusion = (
                f"Therefore, {winner_name} has the most collisions, "
                f"which is {diff} more than the second most ({second_place_str})."
            )
            options = [correct_answer]
        
        variant = random.choice(NUMERICAL_VARIANTS)
        sample['reasoning_cof'] = variant(breakdown_lines_cof, reasoning_conclusion)
        sample['reasoning_cotime'] = variant(breakdown_lines_cotime, reasoning_conclusion)
        sample['reasoning_cot'] = variant(breakdown_lines_cot, reasoning_conclusion)
        sample['reasoning'] = sample['reasoning_cof']
        sample['question'] = (
            f"Compare the total number of collisions across the {actual_k} videos. "
            f"Which video has the most collisions, and what is the numerical difference "
            f"in collisions between this video and the video with the second most collisions?\n"
            f"Select the correct option below."
        )
        
        random.shuffle(options)
        sample['options'], sample['answer'] = add_alphabet_to_options(options, correct_answer)

        # Structured ground truth for the GRPO process reward: per-video
        # collision count, the intermediate fact this task's reasoning is
        # built from (see get_kinematic_comparison's identical-purpose
        # process_gt for the pattern this follows). Weaker signal than that
        # task's object-name match, worth flagging: a bare digit is a much
        # easier substring to coincidentally match than a color+shape phrase
        # (e.g. "3" could appear in an unrelated timestamp or frame
        # reference the model happens to mention). Accepted for the same
        # first-cut reason as Kinematic Comparison's own documented
        # limitation -- revisit with a stricter check (e.g. requiring the
        # count to appear near that video's own name) if training results
        # show this being gamed.
        sample['process_gt'] = {v['name']: str(v['count']) for v in video_stats}

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
                # Scale offset accurately based on 25 FPS
                offset_val = offset if time_mode == 'frame' else round(offset * (1.0 / fps), 1)
                fake_time = round(target_time + offset_val, 1)
                
                if fake_time >= 0 and fake_time not in used_gaps:
                    break
                    
            used_gaps.add(fake_time)
            options_set.add(f"Video {vid_id+1}: {format_time_string(fake_time, time_mode)}")
            
        sample['options'] = list(options_set)    
        sample['reasoning'] = ""
        for i in range(k):
            sample['reasoning'] += f"- For Video {i+1} the {mode} collision occurs at {format_time_string(vid_times[i], time_mode)}.\n"
            
        sample['reasoning'] += f"\nTherefore, Video {target_idx+1} has the {mode} collision event."
        
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
        sample['reasoning'] = ""
        
        for i, item in enumerate(items_collected):
            actual_durations.append(item['duration'])
            
            t_min_str = format_time_string(item['min_t'], time_mode)
            t_max_str = format_time_string(item['max_t'], time_mode)
            sample['reasoning'] += f"- {vid_names[i]} has its first collision at {t_min_str} and its last collision at {t_max_str}. The time gap is {item['max_t']} - {item['min_t']} = {item['duration']} {unit_str}.\n\n"
            
        target_duration = max(actual_durations)
        target_idx = actual_durations.index(target_duration)
        correct_answer_str = f"Video {target_idx+1} -- {target_duration} {unit_str}"
        
        sample['reasoning'] += f"Comparing the gaps, {vid_names[target_idx]} has the longest duration with {target_duration} {unit_str}."
        
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
                # Scale offset accurately based on 25 FPS
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

def nonconsecutive_kinematic_items(dataset, n_items, threshold=0.25, max_uses=2, rng=random, max_draws=2_000_000):
    """Kinematic Comparison items from random pairs of clear-margin videos.

    get_kinematic_comparison pairs video i with i + 1 only, which leaves 630
    usable pairs among the 10,000 CLEVRER training videos at margin 0.25.
    Pairing any two videos whose own top-2 peak velocities differ by at least
    `threshold` (and whose peaks differ from each other by at least
    `threshold`), each video used at most `max_uses` times, yields ~3100.
    """
    peaks = []
    for i in range(len(dataset.data_list)):
        _, vmax, _, _ = dataset._get_velocity_item(i)
        top2 = get_top_k_fastest_objects(list(dataset.data_list[i].values())[0], k=2)
        clear = len(top2) == 2 and abs(top2[0]['max_vel'] - top2[1]['max_vel']) >= threshold
        peaks.append((vmax, clear))
    good = [i for i, (_, clear) in enumerate(peaks) if clear]
    uses, pairs, items = collections.Counter(), set(), []
    for _ in range(max_draws):
        if len(items) >= n_items or len(good) < 2:
            break
        a, b = rng.sample(good, 2)
        key = (min(a, b), max(a, b))
        if uses[a] >= max_uses or uses[b] >= max_uses or key in pairs:
            continue
        if abs(peaks[a][0] - peaks[b][0]) < threshold:
            continue
        sample = dataset.get_kinematic_comparison_pair(a, b, threshold=threshold)
        if sample is None:
            continue
        pairs.add(key)
        uses[a] += 1
        uses[b] += 1
        items.append(sample)
        if uses[a] >= max_uses:
            good.remove(a)
        if uses[b] >= max_uses:
            good.remove(b)
    return items


def generate_test_QA(qa_num=200, total_num=200, k=3, split='eval', verbose=False):
    
    dataset = CLEVRER_Comparative_Reasoning(
        root_path=f"{DATA_ROOT}/clvr/", 
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


# generate_train_collision_QA(qa_num=[1000, 500, 500], total_num=10_000, 
#                             nframes=32, time_mode='timestamp', verbose=False)

def generate_comparative_qa(args, dataset, save=False):
    data_list = []
    data_idx = 0
    if args.task_type == 'kinematic' and args.kinematic_pairing == 'nonconsecutive':
        data_list = nonconsecutive_kinematic_items(dataset, args.total_num, threshold=args.kinematic_threshold)
        data_idx = len(dataset.data_list)  # skips the consecutive loop below
    while len(data_list) < args.total_num and len(dataset.data_list) > data_idx+1:
        
        if args.task_type == 'collision':
            data = dataset.get_collision_count_QA(data_idx, nframes=32, k=2, time_mode='timestamp')

        else:
            data = dataset.get_kinematic_comparison(data_idx, threshold=args.kinematic_threshold)
        data_idx += 1
        
            
        if data:
            data_list.append(data)
            
    if args.task_type == 'collision' and data_list:
        # Distractors are chosen over the whole list (see reoption_numerical).
        rebuilt, stats = reoption_numerical([d['collision_count'] for d in data_list], random)
        for d, (options, answer) in zip(data_list, rebuilt):
            d['options'], d['answer'] = options, answer
        print(f"Numerical options rebuilt: {stats}")
    print(f"Generated {len(data_list)} samples (split={args.split}, task={args.task_type})")
    if data_list:
        print(data_list[0])

    if args.output:
        out_path = args.output
    elif args.split == 'train':
        # Training data: keep the reasoning traces, one file per task type.
        name = 'Clvr_Numerical_Comparison' if args.task_type == 'collision' else 'Clvr_Kinematic_Comparison'
        out_path = f"json_files/train/{name}.jsonl"
    else:
        # Evaluation data: preserve the existing benchmark layout.
        name = 'Clvr_Numerical_Comparison' if args.task_type == 'collision' else 'Clvr_Kinematic_Comparison'
        out_path = f"json_files/eval/Comparative_Reasoning/{name}.jsonl"

    if save:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_jsonl_file(out_path, data_list)
        print(f"Saved {len(data_list)} samples to {out_path}")
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multimodal video models.")

    parser.add_argument("--total-num", type=int, required=True, 
                        help="Index of the sample to retrieve from the dataset.")
    
    parser.add_argument("--task-type", type=str, choices=['collision', 'kinematic'],
                        required=True, help="Task type.")

    parser.add_argument("--kinematic-threshold", type=float, default=0.25,
                        help="Minimum peak-velocity margin (both between the two videos' "
                             "winners, and between each video's own top-2) for a kinematic "
                             "example to be considered unambiguous. 0.25 matches the "
                             "benchmark; lower it for more training volume at the cost of "
                             "closer, less clear-cut 'winner' margins.")

    parser.add_argument("--kinematic-pairing", choices=['consecutive', 'nonconsecutive'], default='consecutive',
                        help="consecutive: video i with i + 1 (the benchmark); nonconsecutive: random "
                             "clear-margin videos, each used at most twice (training data).")

    parser.add_argument("--split", type=str, default='validation',
                        choices=['train', 'validation'],
                        help="CLEVRER split to draw from. 'train' -> video_train/annotation_train "
                             "(ids 0-9999, used for SFT data); 'validation' -> video_validation/"
                             "annotation_validation (ids 10000-14999, used for the benchmark).")

    parser.add_argument("--root-path", type=str, default=f"{DATA_ROOT}/clvr/",
                        help="Root directory containing video_<split>/ and annotation_<split>/.")

    parser.add_argument("--output", type=str, default=None,
                        help="Explicit output .jsonl path. Defaults to a train/ or eval/ path "
                             "chosen from --split and --task-type.")

    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for video pairing, trace variants and distractors.")

    args = parser.parse_args()
    random.seed(args.seed)

    dataset = CLEVRER_Comparative_Reasoning(
            root_path=args.root_path,
            split=args.split,
            sample_num=args.total_num*2
        )


    generate_comparative_qa(args, dataset, save=True)


# Eval (unchanged behaviour):
#   python -m generation.clevrer.clvr_comparative_reasoning --total-num 1000 --task-type collision
# Train:
#   python -m generation.clevrer.clvr_comparative_reasoning --total-num 1000 --task-type collision  --split train
#   python -m generation.clevrer.clvr_comparative_reasoning --total-num 1000 --task-type kinematic --split train

