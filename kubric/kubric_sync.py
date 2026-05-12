import os
import random
import argparse
import numpy as np

from kubric.base_kubric import KubricDataset
from utils import write_jsonl_file


class TimeSyncQAGenerator:
    def __init__(self, data_manager: KubricDataset):
        self.data_manager = data_manager

    def generate(self, scene_idx: str, fps=12, original_duration=5, crop_duration=3):
        scene_path = os.path.join(self.data_manager.path, scene_idx)
        total_frames = int(original_duration * fps)
        crop_frames = int(crop_duration * fps)      
        max_start_frame = total_frames - crop_frames
        
        start_frames = random.sample(range(0, max_start_frame + 1), 3)
        t1, t2, t3 = start_frames
        
        cameras = ['cam0', 'cam1', 'cam2']
        random.shuffle(cameras)
        cam_vid1, cam_vid2, cam_vid3 = cameras
        
        offset1_sec = (t2 - t1) / float(fps)
        offset2_sec = (t3 - t1) / float(fps)
        
        video_list = [
            {"video_path": os.path.join(scene_path, f"{cam_vid1}.mp4"), "start_frame": t1, "end_frame": t1 + crop_frames,
             "start_sec": round(t1 / float(fps), 1), "end_sec": round((t1 + crop_frames) / float(fps), 1)},
            {"video_path": os.path.join(scene_path, f"{cam_vid2}.mp4"), "start_frame": t2, "end_frame": t2 + crop_frames,
             "start_sec": round(t2 / float(fps), 1), "end_sec": round((t2 + crop_frames) / float(fps), 1)},
            {"video_path": os.path.join(scene_path, f"{cam_vid3}.mp4"), "start_frame": t3, "end_frame": t3 + crop_frames,
             "start_sec": round(t3 / float(fps), 1), "end_sec": round((t3 + crop_frames) / float(fps), 1)}
        ]
        
        correct_text = f"Video 2 starts at {offset1_sec:+.1f} seconds; Video 3 starts at {offset2_sec:+.1f} seconds."
        distractor1 = f"Video 2 starts at {offset2_sec:+.1f} seconds; Video 3 starts at {offset1_sec:+.1f} seconds."
        distractor2 = f"Video 2 starts at {-offset1_sec:+.1f} seconds; Video 3 starts at {-offset2_sec:+.1f} seconds."
        
        options = list(set([correct_text, distractor1, distractor2]))
        
        while len(options) < 4:
            fuzz1 = offset1_sec + random.choice([0.1, -0.1, 0.2, -0.2])
            fuzz2 = offset2_sec + random.choice([0.3, -0.3, 0.4, -0.4])
            fuzzed_opt = f"Video 2 starts at {fuzz1:+.1f} seconds; Video 3 starts at {fuzz2:+.1f} seconds."
            if fuzzed_opt not in options:
                options.append(fuzzed_opt)
        
        random.shuffle(options)
        labels = ['A', 'B', 'C', 'D']
        options_list = [f"{labels[i]}) {opt}" for i, opt in enumerate(options)]
        correct_label = next((opt for opt in options_list if opt.split(') ')[1] == correct_text), "")
        
        question = (
            f"You are provided with three videos (Video 1, Video 2, and Video 3) recorded at {fps} FPS. "
            f"Each video is exactly {crop_duration}.0 seconds long and shows the same physical event "
            "captured simultaneously from three different camera angles.\n"
            "Identify the exact temporal offset of the latter two videos. At what timestamp in Video 1's timeline "
            "does the very first frame of Video 2 and Video 3 occur? (Note: A negative timestamp means the video started before Video 1)."
        )
        
        return {
            'video_list': video_list,
            'question': question,
            'options': options_list,
            'answer': correct_label
        }
    
    
def generate_multi_angle_sync_dataset(path, total_num, save, output_path):
    data_manager = KubricDataset(path=path)
    generator = TimeSyncQAGenerator(data_manager=data_manager)
    data_list = []

    for i in range(total_num):
        data_list.append(generator.generate(scene_idx=f"scene_{i}"))

    if save:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_jsonl_file(filepath=output_path, data_list=data_list)
        print(f"Successfully saved {len(data_list)} samples to {output_path}")
    else:
        print(f"Generated {len(data_list)} samples. Run with --save to write to disk.")
        if data_list:
            print("Sample 0:", data_list[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kubric Multi-Angle Synchronization QA Dataset")
    
    parser.add_argument(
        "--path", 
        type=str, 
        required=True, 
        help="Path to the base directory containing Kubric scene folders."
    )
    
    parser.add_argument(
        "--total_num", 
        type=int, 
        default=1000, 
        help="Total number of QA samples to generate (default: 1000)."
    )
    
    parser.add_argument(
        "--save", 
        action="store_true", 
        help="Include this flag to save the output to a JSONL file."
    )
    
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="json_files/Temporal_Alignment/Kubric_Multi_Angle_Synchronization.jsonl", 
        help="Path where the output JSONL file will be saved."
    )

    args = parser.parse_args()

    # Run the generator with parsed arguments
    generate_multi_angle_sync_dataset(
        path=args.path, 
        total_num=args.total_num, 
        save=args.save,
        output_path=args.output_path
    )