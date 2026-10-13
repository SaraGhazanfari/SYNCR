import json
import base64
import cv2
import numpy as np

# Frames per video: sampled at the task's eval fps within the clip (as the open models are), capped at MAX_FRAMES
# (the largest per-video frame count Qwen3-VL gets on SYNCR). Each frame is downscaled to at most MAX_SIDE pixels
# on its long side; with detail="low" the API bills a fixed ~85 tokens per image regardless of size.
MAX_FRAMES = 40
MAX_SIDE = 512


class GPTModel:
    def __init__(self, model_version="gpt-4o", *args, **kwargs):
        self.model_version = model_version
        self.client = self._authenticate()

        # Cache extracted base64 frames to prevent re-processing the same video/slice
        self.cached_frames = {}

    def _authenticate(self):
        from openai import OpenAI
        try:
            with open("tokens/gpt_api_key.json", "r") as f:
                API_KEY = json.load(f)["api_key"]
            return OpenAI(api_key=API_KEY)
        except FileNotFoundError:
            print("Warning: API key file not found. Ensure it exists at tokens/gpt_api_key.json")
            # Fall back to checking the OPENAI_API_KEY environment variable
            return OpenAI()

    def _extract_and_cache_frames(self, video_path, start_sec=None, end_sec=None, target_fps=None):
        """Frames of [start_sec, end_sec) at target_fps (previously a fixed 10 frames per video), capped at
        MAX_FRAMES, as (seconds from the clip start, base64 JPEG) pairs."""
        cache_key = f"{video_path}_{start_sec}_{end_sec}_{target_fps}"
        if cache_key in self.cached_frames:
            return self.cached_frames[cache_key]

        video = cv2.VideoCapture(video_path)
        native_fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if native_fps <= 0 or total_frames <= 0:
            video.release()
            return []

        start_frame = int(round(start_sec * native_fps)) if start_sec is not None else 0
        end_frame = (int(round(end_sec * native_fps)) if end_sec is not None and end_sec != float('inf')
                     else total_frames)
        start_frame, end_frame = max(0, start_frame), min(total_frames, end_frame)
        frames_in_range = end_frame - start_frame
        if frames_in_range <= 0:
            video.release()
            return []

        duration = frames_in_range / native_fps
        count = int(duration * target_fps) if target_fps else 10
        count = min(max(1, count), MAX_FRAMES, frames_in_range)
        indices = np.linspace(start_frame, end_frame - 1, count).round().astype(int)

        frames = []
        for idx in indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            success, frame = video.read()
            if not success:
                continue
            h, w = frame.shape[:2]
            scale = MAX_SIDE / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frames.append(((idx - start_frame) / native_fps, base64.b64encode(buffer).decode("utf-8")))
        video.release()

        self.cached_frames[cache_key] = frames
        return frames

    def get_response_batch(self, text_parts, video_lists, gt_list=None, verbose=True,
                           fps=None, max_new_tokens=2048):
        """Mimics the Qwen/Gemini batch inference method for GPT."""
        if gt_list is None:
            gt_list = [None] * len(text_parts)

        output_texts = []

        # Iterate through the "batch" sequentially
        for text_part, video_list, gt in zip(text_parts, video_lists, gt_list):
            content_payload = []

            # 1. Videos first (as for the other models): "Video i:", then each frame with a clip-relative time marker
            for i, video_dict in enumerate(video_list):
                video_path = video_dict.get("video_path")
                if not video_path:
                    continue
                frames = self._extract_and_cache_frames(
                    video_path, video_dict.get('start_sec'), video_dict.get('end_sec'), target_fps=fps)
                content_payload.append({"type": "text", "text": f"Video {i+1}:"})
                for t, frame in frames:
                    content_payload.append({"type": "text", "text": f"<{t:.1f} seconds>"})
                    content_payload.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame}", "detail": "low"}
                    })

            # 2. The question, options and answer format
            content_payload.append({"type": "text", "text": text_part})

            # 3. Generate the response. Deterministic decoding (temperature 0) where the model allows it; some models
            # (e.g. gpt-6-astra) only accept their default temperature, so on that error retry without it (
            # those results are sampled, not greedy).
            request = dict(model=self.model_version, messages=[{"role": "user", "content": content_payload}],
                           max_completion_tokens=max_new_tokens)
            if not getattr(self, "default_temperature_only", False):
                request["temperature"] = 0.0
            try:
                response = self.client.chat.completions.create(**request)
            except Exception as e:
                if "temperature" not in str(e) or "temperature" not in request:
                    raise
                print(f"Warning: {self.model_version} rejects temperature=0; using its default temperature")
                self.default_temperature_only = True
                request.pop("temperature")
                response = self.client.chat.completions.create(**request)

            choice = response.choices[0]
            output_text = choice.message.content
            if choice.finish_reason == "length":
                print(f"Warning: output hit max_completion_tokens={max_new_tokens}")
            output_texts.append(output_text)

            if verbose:
                print(f"GT: {gt}\nAssistant: {output_text}\n---")

        return output_texts
