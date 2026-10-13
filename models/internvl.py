import torch
import os
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from peft import PeftModel
from PIL import Image
import numpy as np
# Assuming your local utils still exists
from utils import my_print 
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

def _remote_code_tied_weights_shim():
    """Let transformers 5.x load InternVL's remote-code checkpoints.

    InternVLChatModel ships its own modelling file and never reaches
    PreTrainedModel.post_init(), which is where transformers 5.x assigns the
    instance attribute all_tied_weights_keys. device_map="auto" reads that
    attribute while inferring the device map, so the load dies with
    AttributeError: 'InternVLChatModel' object has no attribute
    'all_tied_weights_keys' before a single weight is placed.

    A descriptor on PreTrainedModel supplies the default the remote class never
    sets, while still letting post_init() assign the real value on every normal
    model (a bare property with no setter would make that assignment raise). The
    value comes from the class's own
    _tied_weights_keys where it declares one, so accelerate still learns
    which weights are tied and keeps them on one device.

    Returns a callable that removes the shim again, so this only applies to the
    remote-code load it wraps.
    """
    from transformers import PreTrainedModel
    if hasattr(PreTrainedModel, 'all_tied_weights_keys'):
        return lambda: None

    class _TiedKeysDefault:
        """Descriptor supplying the default only when nothing was assigned.

        Defining __set__ makes this a data descriptor, which takes priority over
        the instance __dict__ on reads too -- so __get__ has to consult
        obj.__dict__ itself, or post_init()'s assignment on a normal model would
        be silently ignored and every model would report no tied weights.
        """

        def __get__(self, obj, owner=None):
            if obj is None:
                return {}
            if 'all_tied_weights_keys' in obj.__dict__:
                return obj.__dict__['all_tied_weights_keys']
            # transformers 5.x treats this as a mapping (it calls .keys() on it),
            # so the default is a dict keyed by the remote class's own declared
            # tied-weight names, not the list that class stores.
            declared = getattr(type(obj), '_tied_weights_keys', None) or ()
            if isinstance(declared, dict):
                return dict(declared)
            return {k: k for k in declared}

        def __set__(self, obj, value):
            obj.__dict__['all_tied_weights_keys'] = value

    PreTrainedModel.all_tied_weights_keys = _TiedKeysDefault()
    return lambda: delattr(PreTrainedModel, 'all_tied_weights_keys')


class InternVLModel:
    def __init__(self, model_version, *args, **kwargs):
        undo_shim = _remote_code_tied_weights_shim()
        try:
            self.model = AutoModel.from_pretrained(
                model_version,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                low_cpu_mem_usage=False,
                cache_dir=kwargs.get('cache_dir')
            ).eval()
        finally:
            undo_shim()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_version, 
            trust_remote_code=True, 
            use_fast=False,
            cache_dir=kwargs.get('cache_dir')
        )
        
        if kwargs.get('adapter_path') is not None:
            print(f"Applying LoRA adapters from {kwargs.get('adapter_path')}...")
            self.model = PeftModel.from_pretrained(self.model, kwargs.get('adapter_path'))
        
        self.img_size = 448 # Standard for InternViT
        self.transform = self._build_transform(input_size=self.img_size)


    def _build_transform(self, input_size):
        MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        
    def _load_video(self, video_path, fps=1.0, max_frames=64, start_sec=None, end_sec=None):
        """
        Loads video frames based on a target FPS, restricted to [start_sec, end_sec) for cropped clips (the
        previous version ignored the crop and loaded the whole source video, so every clip of a Sequential Ordering or
        Multi-Angle Sync item showed the same full video).
        max_frames caps the frames per video to prevent Out-Of-Memory (OOM) errors on very long videos.
        """
        import cv2
        cap = cv2.VideoCapture(video_path)

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Handle edge cases for corrupted videos
        if original_fps <= 0 or total_frames <= 0:
            cap.release()
            return []

        first, last = 0, total_frames - 1
        if start_sec is not None and end_sec is not None and end_sec != float('inf'):
            first = max(0, int(round(start_sec * original_fps)))
            last = min(total_frames - 1, max(first, int(round(end_sec * original_fps)) - 1))

        # Calculate how many frames we need based on duration and target fps
        video_duration = (last - first + 1) / original_fps
        target_frame_count = int(video_duration * fps)

        # Ensure we always get at least 1 frame
        target_frame_count = max(1, target_frame_count)

        # Optional: Cap the maximum number of frames to prevent VRAM OOM
        if max_frames is not None:
            target_frame_count = min(target_frame_count, max_frames)

        # Distribute the frames evenly across the clip
        indices = np.linspace(first, last, target_frame_count).astype(int)
        
        frames = []
        for idx in indices:
            # Jump directly to the frame index (much faster than reading sequentially)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                
        cap.release()
        return frames

    @torch.inference_mode()
    def get_response_batch(self, text_parts, video_lists, gt_list, verbose=True, 
                           fps=1.0, max_new_tokens=2048):
        # 1. Preparation Phase
        batched_inputs = []
        for text_part, video_list in zip(text_parts, video_lists):
            all_video_tensors = []
            video_prefix = ""
            num_patches_list = []

            for i, video_dict in enumerate(video_list):
                video_path = video_dict['video_path']
                frames = self._load_video(video_path, fps=fps if fps else 1.0,
                                          start_sec=video_dict.get('start_sec'), end_sec=video_dict.get('end_sec'))

                # Failsafe in case a video was corrupted and returned 0 frames
                if not frames:
                    print(f"Warning: Could not read frames from {video_path}")
                    continue

                video_tensor = torch.stack([self.transform(f) for f in frames])
                video_tensor = video_tensor.to(self.model.device).to(torch.bfloat16)

                all_video_tensors.append(video_tensor)
                # InternVL's video format: one <image> placeholder per frame, one patch per frame (the previous
                # "<video>" tag is not an InternVL token, so chat() prepended a single <image> and fed every frame of
                # every video as tiles of one image).
                video_prefix += f"Video {i+1}:\n" + "".join(f"Frame{k+1}: <image>\n" for k in range(len(frames)))
                num_patches_list += [1] * len(frames)

            combined_pixel_values = torch.cat(all_video_tensors, dim=0) if all_video_tensors else None
            full_prompt = f"{video_prefix}{text_part}"

            batched_inputs.append({
                "pixel_values": combined_pixel_values,
                "question": full_prompt,
                "num_patches_list": num_patches_list or None,
            })

        # 2. Inference Phase 
        output_texts = []
        for inputs in batched_inputs:
            response = self.model.chat(
                self.tokenizer,
                pixel_values=inputs["pixel_values"],
                question=inputs["question"],
                num_patches_list=inputs["num_patches_list"],
                generation_config={
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "repetition_penalty": 1.15
                }
            )
            output_texts.append(response)
            
        # 3. Output/Logging Phase
        if verbose:
            for q, a, gt in zip(text_parts, output_texts, gt_list):
                my_print(f"GT: {gt}\nAssistant: {a}\n---")
            
        return output_texts