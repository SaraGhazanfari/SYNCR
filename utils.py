import re, json, math, ast
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2


def save_video_frame(video_path, frame_idx, output_path):
    """
    Extracts a specific frame by index from a video and saves it to disk.
    """
    
    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_path}' does not exist.")
        return False

    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open the video.")
        return False

    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or frame_idx >= total_frames:
        print(f"Error: Frame index {frame_idx} is out of bounds. Video has {total_frames} frames.")
        cap.release()
        return False

    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    
    success, frame = cap.read()

    if success:
        
        cv2.imwrite(output_path, frame)
        print(f"Success! Frame {frame_idx} saved to '{output_path}'.")
    else:
        print(f"Error: Could not read frame at index {frame_idx}.")

    
    cap.release()
    return success


def my_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)
    
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
    return int(h) * 3600 + int(m) * 60 + float(s.split('.')[0]) + ms 

def filename_to_seconds(filename):
    """
    Extracts time from filename format: ID_keyframe_H-M-S-MS.jpg
    Example: 000007_keyframe_0-0-12-554.jpg -> 12.554 seconds
    """
    
    match = re.search(r'_(\d+)-(\d+)-(\d+)-(\d+)\.jpg$', filename)
    
    if match:
        h, m, s, ms = map(int, match.groups())
        ms = 1 if ms >= 500 else 0
        return h * 3600 + m * 60 + s + ms
    return None


def make_simple_pdf(data, output="output.pdf"):
    pages = []

    for subtitle, images in data.items():
        n = len(images)
        cols = 4
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(4 * cols, 4 * rows)
        )

        
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        axes = axes.flatten()

        for ax in axes:
            ax.axis("off")

        for ax, img_path in zip(axes, images):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis("off")

        
        fig.suptitle(subtitle, fontsize=16, wrap=True)

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        pages.append(fig)

    
    
    with PdfPages(output) as pdf:
        for fig in pages:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved {output}")
    
def draw_histogram_from_dict(count_dict):

    keys = sorted(count_dict.keys())
    frequencies = [count_dict[k] for k in keys]

    
    
    plt.bar(keys, frequencies, width=1.0, edgecolor='black', color='skyblue')

    
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution')

    
    plt.savefig('histogram.png')
    plt.show()


def get_data_statistics():

    all_files = [
        "json_files/Holistic_Aggregation/Habitat_Object_Counting.jsonl",
        "json_files/Holistic_Aggregation/Habitat_Route_Planning.jsonl",
        "json_files/Termporal_Alignment/Clvr_Sequential_Ordering.jsonl",
        "json_files/Termporal_Alignment/Kubric_Multi_Angle_Synchronization.jsonl",
        "json_files/Spatial_Tracking/Habitat_Object_Re_identification.jsonl",
        "json_files/Spatial_Tracking/Kubric_Spatial_Measurement.jsonl",
        "json_files/Comparative_Reasoning/Clvr_Numerical_Comparison.jsonl",
        "json_files/Comparative_Reasoning/Clvr_Kinematic_Comparison.jsonl",
    ]

    total_lines_all = 0
    video_count_dict = dict()
   

    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = 0
                video_count = 0
                unique_videos = set() 
                
                for line in f:
                    if not line.strip():
                        continue
                    
                    line_count += 1
                    sample = json.loads(line)
                    videos_in_sample = [item['video_path'] for item in sample.get('video_list', [])]
                    
                    
                    unique_videos.update(videos_in_sample)
                    
                    
                    if line_count == 1:
                        video_count = len(videos_in_sample)

                
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