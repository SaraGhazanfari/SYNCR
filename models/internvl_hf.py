"""InternVL3.5 through the transformers-native classes (e.g. OpenGVLab/InternVL3_5-8B-HF).

Used for fine-tuning (train/internvl/train_internvl_hf.py) and for evaluating the base model and its LoRA adapters on
the same input path. Each video is sampled at the task fps inside its crop window, capped at INTERNVL_MAX_FRAMES frames
(default 16), and every frame is one 448x448 tile (crop_to_patches=False, 256 image tokens) behind its own
"Frame{k}: <IMG_CONTEXT>" line, matching the prompt layout of models/internvl.py.
"""
import os

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, InternVLForConditionalGeneration

from utils import my_print

IMAGE_PLACEHOLDER = "<IMG_CONTEXT>"


def max_frames_default():
    return int(os.environ.get("INTERNVL_MAX_FRAMES", "16"))


def load_video_frames(video_path, fps=1.0, max_frames=16, start_sec=None, end_sec=None):
    """Frames sampled at `fps` within [start_sec, end_sec), evenly spaced, capped at max_frames (same as models/internvl.py)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if original_fps <= 0 or total_frames <= 0:
        cap.release()
        return []
    first, last = 0, total_frames - 1
    if start_sec is not None and end_sec is not None and end_sec != float("inf"):
        first = max(0, int(round(start_sec * original_fps)))
        last = min(total_frames - 1, max(first, int(round(end_sec * original_fps)) - 1))
    duration = (last - first + 1) / original_fps
    count = max(1, int(duration * (fps or 1.0)))
    if max_frames is not None:
        count = min(count, max_frames)
    frames = []
    for idx in np.linspace(first, last, count).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def video_block(index, frames):
    """Prompt text for one video: its label, then one placeholder line per frame."""
    return f"Video {index}:\n" + "".join(f"Frame{k + 1}: {IMAGE_PLACEHOLDER}\n" for k in range(len(frames))) + "\n"


class InternVLHFModel:
    def __init__(self, model_version, *args, **kwargs):
        cache_dir = kwargs.get("cache_dir")
        self.processor = AutoProcessor.from_pretrained(model_version, cache_dir=cache_dir)
        self.processor.tokenizer.padding_side = "left"
        self.model = InternVLForConditionalGeneration.from_pretrained(
            model_version, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir,
            attn_implementation="sdpa").eval()
        if kwargs.get("adapter_path") is not None:
            print(f"Applying LoRA adapters from {kwargs.get('adapter_path')}...")
            self.model = PeftModel.from_pretrained(self.model, kwargs.get("adapter_path"))
        self.max_frames = max_frames_default()

    @torch.inference_mode()
    def get_response_batch(self, text_parts, video_lists, gt_list, verbose=True, fps=1.0, max_new_tokens=2048):
        outputs = []
        for text_part, video_list in zip(text_parts, video_lists):
            prompt, frames = "", []
            for i, video in enumerate(video_list):
                # CrossVid BU/CC/NC list plain paths; MVU-Eval, CVBench and the clip files list dicts.
                if isinstance(video, str):
                    video = {"video_path": video}
                clip = load_video_frames(video["video_path"], fps=fps or 1.0, max_frames=self.max_frames,
                                         start_sec=video.get("start_sec"), end_sec=video.get("end_sec"))
                if not clip:
                    print(f"Warning: Could not read frames from {video['video_path']}")
                    continue
                prompt += video_block(i + 1, clip)
                frames += clip
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt + text_part}]}]
            chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[chat], images=frames or None, crop_to_patches=False, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            outputs.append(self.processor.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        if verbose:
            for answer, gt in zip(outputs, gt_list):
                my_print(f"GT: {gt}\nAssistant: {answer}\n---")
        return outputs
