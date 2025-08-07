import logging
import sys
import numpy as np
import cv2
import torch
import json
import datetime
import time
import base64
import os
import imageio
import glob

from collections import deque
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import get_peft_model, LoraConfig, PeftModel
from PIL import Image


MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
ORIGINAL = True
PEFT = True
FINETUNED_DIR = ""
MIN_PIXELS = 224 * 28 * 28
MAX_PIXELS = 840 * 28 * 28
INFERENCE_NUM_FRAME_PER_TIME = 10
INFERENCE_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "inference_images")
FPS = 10


class VLM():
    def __init__(self):
        self.model = None
        self.processor = None
        self.conversation = None
        self.queue = deque()

        print("--- Load Model and Processor ---")
        if MIN_PIXELS is not None and MAX_PIXELS is not None:
            processor = AutoProcessor.from_pretrained(
                MODEL_NAME, 
                torch_dtype=torch.bfloat16,
                trust_remote_code=True, 
                min_pixels=MIN_PIXELS, 
                max_pixels=MAX_PIXELS, 
            )
        else:
            processor = AutoProcessor.from_pretrained(
                MODEL_NAME, 
                torch_dtype=torch.bfloat16,
                trust_remote_code=True, 
            )
        processor.tokenizer.padding_side  = "left"
        self.processor = processor

        if ORIGINAL:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_NAME, 
                torch_dtype=torch.bfloat16, 
                attn_implementation="flash_attention_2", 
                device_map="cuda:0", 
            ).eval()
            self.model = model
        else:
            if PEFT:
                finetuned_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    MODEL_NAME, 
                    torch_dtype=torch.bfloat16, 
                    attn_implementation="flash_attention_2", 
                    device_map="cuda:0", 
                )
                finetuned_model = PeftModel.from_pretrained(finetuned_model, FINETUNED_DIR).eval()
            else:
                finetuned_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    FINETUNED_DIR, 
                    torch_dtype=torch.bfloat16, 
                    attn_implementation="flash_attention_2", 
                    device_map="cuda:0", 
                ).eval()
            self.model = finetuned_model
        print("--- Load Completely. ---")

        print("--- Load Conversation Template. ---")
        conversation = [
            {
                "role":"system",
                "content":[
                    {
                        "type": "text",
                        "text": """You are monitoring a worker's current action. The worker can be in one of the following six actions:\n- Use Barcode Scanner\n- Using Long Wrench\n- Using Screwdriver\n- Using Mark Pen\n- Doing Oiling\n- Using Wires\nPlease determine the worker's current action, with no extra explanation. If you are not sure, please answer "Unknown"."""
                    }
                ]
            }, 
            {
                "role":"user",
                "content":[
                    {
                        "type": "video",
                    },
                    {
                        "type": "text",
                        "text": "What is the worker's current action?"
                    }
                ]
            }
        ]

        self.conversation = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        print("--- Complete Conversation Template. ---")


    def frame_process(self, PIL_image, text):
        cv2_image = cv2.cvtColor(np.asarray(PIL_image), cv2.COLOR_RGB2BGR)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        text_w, text_h = text_size
        cv2.rectangle(cv2_image, (0, 0), (text_w, text_h), (34, 139, 34), -1)
        cv2.putText(cv2_image, text, (0, text_h - 1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        return cv2_image
    
    
    def inference(self, curr_frame_number):
        inputs = self.processor(
            text=[self.conversation], 
            images=None, 
            videos=list(self.queue), 
            padding=True, 
            return_tensors="pt", 
        ).to(self.model.device, self.model.dtype)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=256, eos_token_id=self.processor.tokenizer.eos_token_id, do_sample=False)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(f"frame_{curr_frame_number-1}_{output_text[0]}")
        print(f"frame_{curr_frame_number}_{output_text[0]}")

        for index in range(-2, 0):
            image = self.frame_process(self.queue[index], output_text[0])
            fname = f"frame_{curr_frame_number + index + 1:07d}.png"
            path = os.path.join(INFERENCE_IMAGES_DIR, fname)
            cv2.imwrite(path, image)


if __name__ == '__main__':
    os.makedirs(INFERENCE_IMAGES_DIR, exist_ok=True)
    vlm = VLM()

    print("Video opens ...")
    cap = cv2.VideoCapture('/home/arthur/other_test/video_stream_inference/old_angle_cam13.mp4')
    if not cap.isOpened():
        print("Video file is wrong.")
        exit()
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print(f"Read_Frame_{frame_count}")

        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        vlm.queue.append(frame)
        if len(vlm.queue) == INFERENCE_NUM_FRAME_PER_TIME:
            vlm.inference(frame_count)
            vlm.queue.popleft()
            vlm.queue.popleft()
        else:
            image = vlm.frame_process(frame, "None")
            fname = f"frame_{frame_count:07d}.png"
            path = os.path.join(INFERENCE_IMAGES_DIR, fname)
            cv2.imwrite(path, image)

        frame_count += 1

    cap.release()
    print("Video closes ...")
    
    print("Merging the inference images to the video, if possible.")
    frame_paths = sorted(glob.glob(os.path.join(INFERENCE_IMAGES_DIR, "*.png")))
    if not frame_paths:
        print("No inference Images.")
        exit()
    output_path = os.path.join(os.path.dirname(__file__), "inference.mp4")
    out = imageio.get_writer(output_path, fps=FPS)
    for path in frame_paths:
        frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        out.append_data(frame)
    out.close()
    print("Complete.")