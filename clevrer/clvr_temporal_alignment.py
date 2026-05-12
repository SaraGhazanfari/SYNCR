import os
import random
import argparse
import itertools
from collections import defaultdict

from utils import write_jsonl_file
from clevrer.base_clevrer import *
from clevrer.utils import *


class CLEVRER_Temporal_Alignment(CLEVRER_Dataset):
    def __init__(self, root_path, split, sample_num):
        super().__init__(root_path, split, sample_num)

    def get_seq_order_QA(self, idx, total_frames=128, k=3, gap_time=0.2, fps=25.0, min_frames=32):
        (ann_name, ann_dict), = self.data_list[idx].items()

        chronological_parts = list(range(k)) 
        presented_parts = chronological_parts.copy()
        random.shuffle(presented_parts)
        
        sample = dict()
        
        # Calculate times dynamically based on inputs rather than using magic numbers
        min_length = round(min_frames / fps, 2)           # e.g., 32 frames @ 25fps = 1.28s
        total_duration = round(total_frames / fps, 2)     # e.g., 128 frames @ 25fps = 5.12s
        total_gaps = k - 1
        usable_time = round(total_duration - (total_gaps * gap_time), 2)
        
        if usable_time < k * min_length:
             raise ValueError(f"Impossible constraints: {usable_time}s is not enough for {k} segments of min {min_length}s.")
             
        remaining_time = round(usable_time - (k * min_length), 2)
        
        lengths = []
        for i in range(k - 1):
            add_time = round(random.uniform(0, remaining_time), 2)
            lengths.append(round(min_length + add_time, 2))
            remaining_time = round(remaining_time - add_time, 2)
            
        lengths.append(round(min_length + remaining_time, 2))
        random.shuffle(lengths)
        
        original_ranges = []
        current_time = 0.0
        for L in lengths:
            end_time = round(current_time + L, 2)
            original_ranges.append((current_time, end_time))
            current_time = round(end_time + gap_time, 2)
        
        sample['question'] = (
            f"These {k} videos are segments of a single continuous event, "
            "but they are shown out of order. What is the correct chronological order of these video segments?"
        )
        
        correct_order_indices = [presented_parts.index(p) + 1 for p in chronological_parts]
        correct_answer = " -> ".join([f"Video {i}" for i in correct_order_indices])
        correct_tuple = tuple(correct_order_indices) 
        
        sample['video_list'] = [
            {
                'video_path': os.path.join(self.video_path, f"video_{ann_name}.mp4"), 
                'start_sec': start, 
                'end_sec': end
            } for start, end in [original_ranges[p] for p in presented_parts]
        ]
        
        all_permutations = list(itertools.permutations(range(1, k + 1)))
        all_permutations.remove(correct_tuple) 
        
        def calc_distance(p1, p2):
            return sum(1 for x, y in zip(p1, p2) if x != y)
        
        difficulty_groups = defaultdict(list)
        for p in all_permutations:
            dist = calc_distance(p, correct_tuple)
            difficulty_groups[dist].append(p)
            
        hard_wrong_options = []
        for dist in sorted(difficulty_groups.keys()):
            group = difficulty_groups[dist]
            random.shuffle(group) 
            hard_wrong_options.extend(group)
            if len(hard_wrong_options) >= 3:
                break
                
        wrong_options = [" -> ".join([f"Video {i}" for i in p]) for p in hard_wrong_options[:3]]
        
        options = [correct_answer] + wrong_options
        random.shuffle(options)
        
        sample['options'], sample['answer'] = add_alphabet_to_options(options, correct_answer)
        
        return sample


def generate_temporal_alignment_qa(args, dataset):
    data_list = []
    
    for idx in range(args.total_num):
        # Using '\r' to update progress on a single line instead of spamming standard output
        print(f"Processing sample index: {idx+1}/{args.total_num}", end='\r')
        
        data = dataset.get_seq_order_QA(
            idx=idx, 
            gap_time=args.gap_time,
            total_frames=args.total_frames, 
            k=args.k
        )
        data_list.append(data)
        
    print(f"\nGenerated {len(data_list)} samples successfully.")

    if data_list:
        print("\n" + "-"*30 + " Sample Preview " + "-"*30)
        for key, val in data_list[0].items():
            print(f"{key}: {val}")
        print("-" * 76)

    if args.save:
        eval_path = 'json_files/Temporal_Alignment/Clvr_Sequential_Ordering.jsonl'
        # Ensures the directory structure exists before writing to prevent FileNotFoundError
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
        write_jsonl_file(filepath=eval_path, data_list=data_list)
        print(f"File successfully saved to: {eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multimodal video models for temporal alignment.")
    
    parser.add_argument("--root-path", type=str, default='/scratch/sg7457/dataset/clvr/', 
                        help="Root path for the dataset.")
    parser.add_argument("--total-num", type=int, default=10, 
                        help="Number of samples to retrieve from the dataset.")
    parser.add_argument("--split", type=str, default='validation', 
                        help="Dataset split to initialize.")
    
    # New customizable arguments
    parser.add_argument("--k", type=int, default=4, 
                        help="Number of video segments to sequence.")
    parser.add_argument("--total-frames", type=int, default=128, 
                        help="Total frames for sequence generation.")
    parser.add_argument("--gap-time", type=float, default=0.0, 
                        help="Time gap (in seconds) applied between chronological video segments.")
    
    # Save flag
    parser.add_argument("--save", action="store_true", 
                        help="Include this flag to save the generated output to a JSONL file.")
    
    args = parser.parse_args()

    dataset = CLEVRER_Temporal_Alignment(
        root_path=args.root_path, 
        split=args.split, 
        sample_num=args.total_num
    )

    generate_temporal_alignment_qa(args, dataset)