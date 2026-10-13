import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from utils import make_timestamps_clip_relative, my_print
from peft import PeftModel

# qwen_vl_utils.vision_process.fetch_video unconditionally falls back to its
# "torchvision" backend on ANY exception from the primary backend (decord),
# including the routine "Resource temporarily unavailable" transient this
# codebase already treats as a normal, catchable per-batch failure under
# multi-tenant cluster load. That torchvision fallback decodes BOTH video and
# audio via torchvision.io.read_video -- but this codebase never uses audio,
# and PyAV's audio-frame decode (io.read_video -> AudioFrame.to_ndarray) has
# been observed to raise numpy._ArrayMemoryError and then segfault the whole
# process instead of propagating a catchable Python exception (we lost entire
# evaluation tasks this way -- a segfault kills every in-flight prefetch thread and the
# whole eval.evaluate subprocess, not just the one sample/batch that
# triggered it, unlike every other decode error this codebase already
# tolerates gracefully). Patched to decode video only via PyAV directly,
# skipping the audio stream -- and the crash-prone code path -- entirely.
import av
from qwen_vl_utils import vision_process as _qwen_vision_process


def _read_video_torchvision_no_audio(ele):
    """Video-only replacement for qwen_vl_utils' torchvision backend (see
    module comment above). Decodes selectively rather than buffering every
    frame before subsampling: CrossVid's "cook" category alone had videos
    with 16000+ raw frames (confirmed: PQxzLMP99BI.mp4, 640x480, 16086
    frames) that OOM'd a 16GB test allocation just building the full frame
    list, of which typically <5% end up in the final sample (e.g. 768/16086
    here) -- buffering all of them (as torchvision.io.read_video's own
    video+audio decode already does, which is the established root cause of
    this same category needing 256GB job-wide with 5-deep concurrent
    prefetch) is wasted peak memory this rewrite doesn't need to inherit.
    """
    from collections import Counter

    video_path = ele["video"]
    if isinstance(video_path, str) and video_path.startswith("file://"):
        video_path = video_path[7:]

    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate)
    total_frames = stream.frames
    if not total_frames:
        # Container doesn't carry a frame-count index (some mp4s don't) --
        # fall back to estimating from duration, same approach ffmpeg-based
        # tooling commonly uses when the exact index is unavailable.
        total_frames = max(1, round(float(stream.duration * stream.time_base) * video_fps))

    nframes = _qwen_vision_process.smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    wanted = Counter(idx.tolist())
    last_wanted = idx[-1].item()
    frames = []
    for pos, frame in enumerate(container.decode(video=0)):
        n = wanted.get(pos, 0)
        if n:
            arr = torch.from_numpy(frame.to_ndarray(format="rgb24"))
            frames.extend([arr] * n)
        if pos >= last_wanted:
            break
    container.close()

    # A container's declared frame count can be slightly optimistic (rare,
    # but seen with some mp4 indexes) -- pad by repeating the last decoded
    # frame rather than crashing on a short read.
    while len(frames) < nframes:
        frames.append(frames[-1])

    video = torch.stack(frames).permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="torchvision",
    )
    return video, video_metadata, sample_fps


_qwen_vision_process.VIDEO_READER_BACKENDS["torchvision"] = _read_video_torchvision_no_audio


class QwenVLModel:
    def __init__(self, model_version, *args, **kwargs):
        self.is_qwen3 = 'Qwen3' in model_version
        
        if self.is_qwen3:
            from transformers import Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_version,
                torch_dtype=torch.bfloat16,  
                attn_implementation="flash_attention_2",
                device_map="auto",
                cache_dir=kwargs.get('cache_dir')
            )
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_version, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                cache_dir=kwargs.get('cache_dir')
            )

        self.processor = AutoProcessor.from_pretrained(
            model_version,
            cache_dir=kwargs.get('cache_dir')
        )
        
        if kwargs.get('adapter_path') is not None:
            print("Applying LoRA adapters...")
            self.model = PeftModel.from_pretrained(self.model, kwargs.get('adapter_path'))
        
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'left'
        
    @torch.inference_mode()
    def prepare_batch_inputs(self, text_parts, video_lists, verbose=True, fps=None):
        """
        CPU-bound half of get_response_batch: chat-template formatting and
        process_vision_info (decord video decode) -- everything up to but not
        including model.generate(). Split out so eval/evaluate.py can prefetch
        batch N+1's video decoding on a background thread while batch N's
        generate_from_inputs runs on GPU. Without this, the eval loop's plain
        synchronous for-loop left the GPU idle during every batch's decode step
        with zero overlap, which is what actually tripped the cluster's
        GPU-utilization cancellation policy on eval jobs (confirmed: eval only
        ever requests 1 GPU, so it can't be the multi-GPU pipeline-sharding
        idle-bubble that caused the training-job cancellations fixed earlier).
        Returns the processed `inputs` object, ready for generate_from_inputs.
        """
        messages_batch = []
        
        for text_part, video_list in zip(text_parts, video_lists):
            message = {"role": "user", "content": []}
            for i, video in enumerate(video_list):
                message['content'].append({"type": "text", "text": f"Video {i+1}:\n"})
                # video_list entries aren't a uniform format across benchmarks:
                # MVU-Eval gives {"video_path": ..., "start_sec": ..., "end_sec": ...}
                # dicts, but CrossVid gives bare path strings. `video["video_path"]`
                # on a plain string raised "TypeError: string indices must be
                # integers" on every single CrossVid example (100% failure rate
                # across all 5 tasks, not just the ones with missing video files).
                if isinstance(video, dict):
                    video_path = video["video_path"]
                    if 'start_sec' in video and 'end_sec' in video:
                        start_, end_ = video['start_sec'], video['end_sec']
                    else:
                        start_ = 0.0
                        end_ = float('inf')
                else:
                    video_path = video
                    start_ = 0.0
                    end_ = float('inf')

                # qwen_vl_utils' smart_nframes does `ele.get("fps", FPS)`, which
                # only falls back to its own sensible default when the "fps" key
                # is ABSENT -- not when it's present but None. Passing fps=None
                # (MVU-Eval/CrossVid never pass --fps) previously set an explicit
                # None here, which propagated into `total_frames / video_fps *
                # fps` and crashed with a TypeError on every single video.
                # Omitting the key when unset lets the library's own default
                # take over, matching what this code path was always meant to do.
                video_ele = {
                    "type": "video",
                    "video": video_path,
                    "video_start": start_,
                    "video_end": end_,
                    "max_pixels": 100352,
                }
                if fps is not None:
                    video_ele["fps"] = fps
                message['content'].append(video_ele)
                message['content'].append({"type": "text", "text": "\n"})
            message['content'].append({"type": "text", "text": text_part})
            messages_batch.append([message]) 
            
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
            for msg in messages_batch
        ]

        # [FIX 2] Qwen3-VL uses patch_size=16, Qwen2.5-VL uses 14
        #patch_size = getattr(self.processor.image_processor, "patch_size", 16 if self.is_qwen3 else 14)
       
        # 2. Extract and process the vision info for the batch
        if self.is_qwen3:
            # [FIX 3] Qwen3-VL requires video_metadata and kwargs
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages_batch,
                return_video_kwargs=True,
                return_video_metadata=True,
                #image_patch_size=patch_size
            )
            # For Qwen3-VL, video_inputs is returned as a list of tuples: (video_tensor, video_metadata)
            if video_inputs is not None and len(video_inputs) > 0:
                videos, video_metadatas = zip(*video_inputs)
                videos = list(videos)
                # Timestamps must count from each clip's own start, not the
                # source file's -- see utils.make_timestamps_clip_relative.
                video_metadatas = make_timestamps_clip_relative(list(video_metadatas))
            else:
                videos = None
                video_metadatas = None
                
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=videos,
                video_metadata=video_metadatas,
                padding=True,
                return_tensors="pt",
                **video_kwargs
            ).to(self.model.device)
            
        else:
            # Standard extraction for Qwen2.5-VL
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages_batch,
                return_video_kwargs=True,
                #image_patch_size=patch_size
            )
            
            video_kwargs['fps'] = video_kwargs['fps'][0]
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs
            ).to(self.model.device)
        
        if verbose:
            my_print(f"Batch video pixel values shape: {inputs.get('pixel_values_videos', torch.empty(0)).shape}")

        return inputs

    @torch.inference_mode()
    def generate_from_inputs(self, inputs, text_parts, gt_list, verbose=True, max_new_tokens=2048):
        """GPU-bound half of get_response_batch -- see prepare_batch_inputs' docstring."""
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # Suppresses console warnings about missing token mappings during padding
            pad_token_id=self.processor.tokenizer.pad_token_id,
            repetition_penalty=1.15,
        )

        # Trim the prompts from the generated IDs.
        # (This accurately isolates the prompt because left-padding flawlessly mirrors in_ids lengths)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        if verbose:
            for q, a, gt in zip(text_parts, output_texts, gt_list):
                my_print(f"GT: {gt}\nAssistant: {a}\n---")

        return output_texts

    def get_response_batch(self, text_parts, video_lists, gt_list, verbose=True,
                           fps=None, max_new_tokens=2048):
        """Unchanged behavior/API for any caller that doesn't need prefetching --
        just runs prepare_batch_inputs then generate_from_inputs in sequence."""
        inputs = self.prepare_batch_inputs(text_parts, video_lists, verbose=verbose, fps=fps)
        return self.generate_from_inputs(inputs, text_parts, gt_list, verbose=verbose, max_new_tokens=max_new_tokens)