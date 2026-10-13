import re, json, math, ast
import os

# Root of the raw simulator data (CLEVRER, Kubric and Habitat videos and annotations).
DATA_ROOT = os.environ.get("SYNCR_DATA", "data").rstrip("/")


def save_video_frame(video_path, frame_idx, output_path):
    """
    Extracts a specific frame by index from a video and saves it to disk.
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_path}' does not exist.")
        return False

    # Open the video capture
    import cv2
    import cv2
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open the video.")
        return False

    # Get the total number of frames to ensure the index is valid
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or frame_idx >= total_frames:
        print(f"Error: Frame index {frame_idx} is out of bounds. Video has {total_frames} frames.")
        cap.release()
        return False

    # Jump directly to the desired frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    # Read the frame
    success, frame = cap.read()

    if success:
        # Save the frame to the specified output path
        cv2.imwrite(output_path, frame)
        print(f"Success! Frame {frame_idx} saved to '{output_path}'.")
    else:
        print(f"Error: Could not read frame at index {frame_idx}.")

    # Always release the capture object when done
    cap.release()
    return success


def my_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def format_frame_timestamp(seconds):
    """Qwen3-VL's per-frame time marker, e.g. "<1.5 seconds>" -- the same
    string its processor puts before each video frame pair
    (Qwen3VLProcessor: f"<{curr_time:.1f} seconds>"). the GRPO training code
    writes it before each sampled image so GRPO prompts match eval."""
    return f"<{seconds:.1f} seconds>"


def make_timestamps_clip_relative(video_metadatas):
    """Shift each video's frames_indices so its first sampled frame is 0.

    Qwen3-VL's processor labels frames "<t seconds>" with t = frame index /
    fps, and qwen_vl_utils' readers count frame indices from the start of
    the SOURCE file even when video_start/video_end crop a clip out of it
    (decord: linspace(start_frame, end_frame, nframes)). For cropped clips
    that gives the answer away: Sequential Ordering's shuffled segments
    showed <0.2>, <2.8>, <1.5>, <4.0> seconds (sorting them is the correct
    order) and Multi-Angle Sync's crops started at <0.5>, <2.1>, <1.7>
    seconds (subtracting gives the offsets). With every clip starting at
    0.0, the times still say how far into the clip a frame is, but nothing
    about where the clip sat in the source video.

    Uncropped videos already start at index 0 and are unchanged. Modifies
    the metadata dicts in place and returns the list.
    """
    for metadata in video_metadatas:
        indices = [int(i) for i in metadata["frames_indices"]]
        metadata["frames_indices"] = [i - indices[0] for i in indices]
    return video_metadatas
    
def read_file(filepath):
    """Reads a text file and returns its content as a string."""
    with open(filepath, 'r') as file:
        content = file.read()
    return content

def write_file(filepath, content):
    with open(filepath, 'w') as file:
        file.write(content)
    print(f"Saved {filepath}")

def write_jsonl_file(filepath, data_list):
    """Writes a list of dictionaries to a JSONL file."""
    with open(filepath, 'w') as f:
        for data in data_list:
            f.write(json.dumps(data) + '\n')
    print(f"Saved {filepath}")


def read_jsonl_file(filepath):
    """Reads a JSONL file and returns a list of dictionaries."""
    data_list = []
    with open(filepath, 'r') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))
    return data_list
    
def read_json_file(filepath):
    """Reads a JSON file and returns a list of dictionaries."""
    with open(filepath, 'r') as f:
        return json.load(f)
    return data_list

    
def timestamp_to_seconds(time_str):
    """Converts HH:MM:SS.mmm string to total seconds (float)."""
    h, m, s = time_str.split(':')
    ms = 1 if int(s.split('.')[1]) >= 500 else 0
    return int(h) * 3600 + int(m) * 60 + float(s.split('.')[0]) + ms #+ ()

def filename_to_seconds(filename):
    """
    Extracts time from filename format: ID_keyframe_H-M-S-MS.jpg
    Example: 000007_keyframe_0-0-12-554.jpg -> 12.554 seconds
    """
    # Regex to find the H-M-S-MS pattern at the end of the file
    match = re.search(r'_(\d+)-(\d+)-(\d+)-(\d+)\.jpg$', filename)
    
    if match:
        h, m, s, ms = map(int, match.groups())
        ms = 1 if ms >= 500 else 0
        return h * 3600 + m * 60 + s + ms
    return None


def get_data_statistics():

    all_files = [
        "json_files/eval/Holistic_Synthesis/Habitat_Object_Counting.jsonl",
        "json_files/eval/Holistic_Synthesis/Habitat_Route_Planning.jsonl",
        "json_files/eval/Temporal_Alignment/Clvr_Sequential_Ordering.jsonl",
        "json_files/eval/Temporal_Alignment/Kubric_Multi_Angle_Synchronization.jsonl",
        "json_files/eval/Spatial_Tracking/Habitat_Object_Re_identification.jsonl",
        "json_files/eval/Spatial_Tracking/Kubric_Cross_Video_Spatial_Measurement.jsonl",
        "json_files/eval/Comparative_Reasoning/Clvr_Numerical_Comparison.jsonl",
        "json_files/eval/Comparative_Reasoning/Clvr_Kinematic_Comparison.jsonl",
    ]

    total_lines_all = 0
    video_count_dict = dict()
   

    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = 0
                video_count = 0
                unique_videos = set() 
                # Iterate through every line to extract video names
                for line in f:
                    if not line.strip():
                        continue
                    
                    line_count += 1
                    sample = json.loads(line)
                    videos_in_sample = [item['video_path'] for item in sample.get('video_list', [])]
                    
                    # NEW: Add the videos from this line to our global set
                    unique_videos.update(videos_in_sample)
                    
                    # Keep original logic: use the 1st line's video count for the dictionary
                    if line_count == 1:
                        video_count = len(videos_in_sample)

                # Update the dictionary after finishing the file loop
                video_count_dict[video_count] = video_count_dict.get(video_count, 0) + line_count
            
            
            print(f"{file_path}:")
            print(f"Total unique videos used for any task: {len(unique_videos)}")
            print(f"  --> Lines: {line_count} | Videos in 1st sample: {video_count}")
            total_lines_all += line_count

        except FileNotFoundError:
            print(f"Error: Could not find {file_path}")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {file_path}")

    print("-" * 50)
    print(f"Total lines across all files: {total_lines_all}")
    print(f"Video count dictionary: {video_count_dict}")
    



if __name__ == "__main__":
    get_data_statistics()