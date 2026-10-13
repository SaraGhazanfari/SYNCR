import torch
import numpy as np
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
from peft import PeftModel
import os
from utils import my_print # Assuming this is your custom utility


class LlavaOneVisionModel:
    def __init__(self, model_version, *args, **kwargs):
        # Load LLaVA-OneVision model
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
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
        
        # Left-padding is essential for batched generation and easy prompt trimming
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'left'

    def _extract_frames(self, video_path, num_frames=16, start_sec=0.0, end_sec=None):
        from decord import VideoReader, cpu
        """Helper method to read video and uniformly sample frames using decord."""
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        # Calculate start and end frames
        start_frame = int(start_sec * fps)
        if end_sec is not None and end_sec < float('inf'):
            end_frame = int(end_sec * fps)
        else:
            end_frame = total_frames
            
        # Keep bounds safe
        end_frame = min(end_frame, total_frames)
        start_frame = min(start_frame, end_frame - 1)
        
        # Uniformly sample exactly `num_frames` from the target window
        frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        frames = vr.get_batch(frame_indices).asnumpy() # Shape: (num_frames, H, W, C)
        
        return frames

    @torch.inference_mode()
    def get_response_batch(self, text_parts, video_lists, gt_list, verbose=True, fps=None,
                           num_frames=16, max_new_tokens=2048):
        # Note: Replaced `fps` argument with `num_frames` as LLaVA relies 
        # on a fixed context window of sampled frames (typically 16 or 32).
        
        messages_batch = []
        all_videos = []
        
        for text_part, video_list in zip(text_parts, video_lists):
            message = {"role": "user", "content": []}
            for i, video in enumerate(video_list):
                message['content'].append({"type": "text", "text": f"Video {i+1}:\n"})
                
                start_ = video.get('start_sec', 0.0)
                end_ = video.get('end_sec', None)
                
                # Extract frames and add to our flat batch-level list
                frames = self._extract_frames(
                    video["video_path"], 
                    num_frames=num_frames, 
                    start_sec=start_, 
                    end_sec=end_
                )
                all_videos.append(frames)
                
                # Add the generic <video> marker for the processor
                message['content'].append({"type": "video"})
                message['content'].append({"type": "text", "text": "\n"})
                
            message['content'].append({"type": "text", "text": text_part})
            messages_batch.append([message]) 
            
        # 1. Apply the standard chat template 
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
            for msg in messages_batch
        ]

        # 2. Process text and raw frames into model inputs
        processor_kwargs = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt"
        }
        if all_videos:
            processor_kwargs["videos"] = all_videos
            
        inputs = self.processor(**processor_kwargs).to(self.model.device)
        
        if verbose:
            if 'pixel_values_videos' in inputs:
                my_print(f"Batch video pixel values shape: {inputs['pixel_values_videos'].shape}") 
            else:
                my_print("No video pixel values found (text-only inputs).")
            
        # 3. Generate response
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # Suppresses console warnings about missing token mappings during padding
            pad_token_id=self.processor.tokenizer.pad_token_id, 
            repetition_penalty=1.15, 
        )
        
        # 4. Trim the prompts from the generated IDs
        # (Accurately isolates the prompt because left-padding cleanly mirrors in_ids lengths)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # 5. Decode outputs
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        if verbose:
            for q, a, gt in zip(text_parts, output_texts, gt_list):
                my_print(f"GT: {gt}\nAssistant: {a}\n---")
        
        return output_texts