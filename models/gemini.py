import json
import time

CLIP_CACHE_DIR = os.environ.get("SYNCR_CLIP_CACHE", ".cache/gemini_clips")
GEMINI_MAX_FPS = 24.0  # the API accepts video_metadata.fps in (0, 24]


def cut_clip(video_path, start_sec, end_sec, cache_dir=CLIP_CACHE_DIR):
    """Write frames [start_sec, end_sec) of video_path to their own mp4.

    The clip starts at t=0, so neither the file nor the prompt reveals where it sat in the source video -- for
    Sequential Ordering and Multi-Angle Sync that position is the answer. (The previous version uploaded the whole
    source video and put "focus on the events between {start}s and {end}s" in the prompt.)"""
    import hashlib
    import os
    import cv2
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1(f"{os.path.abspath(video_path)}|{start_sec}|{end_sec}".encode()).hexdigest()[:16]
    out = os.path.join(cache_dir, f"{key}.mp4")
    if os.path.exists(out):
        return out
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    first, last = int(round(start_sec * fps)), int(round(end_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    writer = None
    for _ in range(max(1, last - first)):
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
    cap.release()
    if writer is None:
        raise RuntimeError(f"no frames in {video_path} [{start_sec}, {end_sec})")
    writer.release()
    os.replace(out + ".tmp.mp4", out)
    return out


class GeminiModel:
    def __init__(self, model_version="gemini-3-flash-preview", *args, **kwargs):
        self.model_version = model_version
        self.client = self._authenticate()
        
        # Cache uploaded files to prevent re-uploading the same video in subsequent batches
        self.uploaded_files = {}

    def _authenticate(self):
        from google import genai
        # Your API key tier dictates whether this is "free" or "pay-as-you-go"
        try:
            with open("tokens/gemini_api_key.json", "r") as f:
                API_KEY = json.load(f)["api_key"]
            return genai.Client(api_key=API_KEY)
        except FileNotFoundError:
            print("Warning: tokens/gemini_api_key.json not found; using the GEMINI_API_KEY environment variable")
            # The client reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment
            return genai.Client()

    def _upload_and_wait(self, video_path):
        """Uploads a video to the Gemini API and waits for processing to complete."""
        if video_path in self.uploaded_files:
            return self.uploaded_files[video_path]
        
        print(f"Uploading {video_path} to Gemini...")
        video_file = self.client.files.upload(file=video_path)
        
        # Wait for the video to be processed by Google's servers
        while video_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            # Refresh the file status
            video_file = self.client.files.get(name=video_file.name)
        print(" Done!")
        
        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed for {video_path}")
            
        self.uploaded_files[video_path] = video_file
        return video_file

    def get_response_batch(self, text_parts, video_lists, gt_list=None, verbose=True, 
                           fps=None, max_new_tokens=2048):
        """Mimics the Qwen batch inference method."""
        if gt_list is None:
            gt_list = [None] * len(text_parts)
            
        output_texts = []
        
        # Iterate through the "batch" sequentially
        for text_part, video_list, gt in zip(text_parts, video_lists, gt_list):
            message_parts = []
            
            # 1. Process and append videos
            for i, video_dict in enumerate(video_list):
                video_path = video_dict.get("video_path")
                if video_path:
                    # Cropped clips are cut into their own file (starting at t=0) before upload; no crop times in
                    # the prompt (see cut_clip).
                    start_sec = video_dict.get('start_sec')
                    end_sec = video_dict.get('end_sec')
                    if start_sec is not None and end_sec is not None and end_sec != float('inf'):
                        video_path = cut_clip(video_path, start_sec, end_sec)
                    gemini_video = self._upload_and_wait(video_path)

                    message_parts.append(f"Video {i+1}:\n")
                    from google.genai import types
                    # Sample at the task's eval fps (capped at the API maximum) instead of the API default of 1 fps.
                    metadata = types.VideoMetadata(fps=min(float(fps), GEMINI_MAX_FPS)) if fps else None
                    message_parts.append(types.Part(
                        file_data=types.FileData(file_uri=gemini_video.uri, mime_type=gemini_video.mime_type),
                        video_metadata=metadata))
                    message_parts.append("\n")
            
            # 2. Append the main text prompt
            message_parts.append(text_part)
            from google.genai import types
            # 3. Configure generation parameters to match Qwen
            # Qwen's `do_sample=False` maps to `temperature=0.0`
            # Qwen's `max_new_tokens=1024` maps to `max_output_tokens=1024`
            config = types.GenerateContentConfig(
                max_output_tokens=max_new_tokens,
                temperature=0.0, 
                # thinking_config=types.ThinkingConfig(thinking_level="LOW")
            )
            
            # 4. Generate the response, retrying transient server errors (a 503 UNAVAILABLE used to fail the
            # item, which then counted as wrong).
            import time
            for attempt in range(6):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_version,
                        contents=message_parts,
                        config=config
                    )
                    break
                except Exception as e:
                    transient = any(code in str(e) for code in ("503", "500", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
                    if not transient or attempt == 5:
                        raise
                    wait = 30 * 2 ** attempt
                    print(f"Warning: transient Gemini error ({str(e)[:80]}); retry {attempt + 1}/5 in {wait}s")
                    time.sleep(wait)
            
            output_texts.append(response.text)
            
            if verbose:
                print(f"GT: {gt}\nAssistant: {response.text}\n---")
        
        return output_texts