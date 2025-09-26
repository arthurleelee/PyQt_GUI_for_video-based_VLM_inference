import sys
import os
import cv2
import time
import torch
import numpy as np
import imageio
import gc
import subprocess
import shutil
import math
import torchvision.transforms as T
import supervision as sv
import multiprocessing as mp
import signal

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QFileDialog, QSlider, QSpinBox, QComboBox, QTextEdit, 
    QGroupBox, QFormLayout, QMessageBox, QProgressBar, QStyle, QSizePolicy, 
    QTabWidget, QColorDialog, QScrollArea, QLineEdit, QProgressDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import QImage, QPixmap, QColor
from collections import deque
from transformers import Qwen2_5_VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration, AutoModelForImageTextToText, AutoTokenizer, AutoProcessor, AutoModel, AutoModelForZeroShotObjectDetection
from qwen_vl_utils import process_vision_info
from PIL import ImageFont, Image, ImageDraw
from torchvision.transforms.functional import InterpolationMode
from scipy.spatial import cKDTree
from ffmpeg_progress_yield import FfmpegProgress
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pynvml import *


STYLE_SHEET = """
/* Global Settings */
QWidget {
    font-family: "Segoe UI", "Roboto", "Helvetica Neue", sans-serif;
    font-size: 16px;
    color: #f0f0f0; /* Light Gray Text */
    background-color: #1a1a1a; /* Dark Background */
}

/* Main Window */
QMainWindow {
    background-color: #121212;
}

/* GroupBox */
QGroupBox {
    background-color: #242424;
    border: 1px solid #333333;
    border-radius: 8px;
    margin-top: 1em;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 10px;
    background-color: #242424;
    color: #39ff14; /* Neon Green Accent Color */
}

/* Button */
QPushButton {
    background-color: #333333;
    border: 1px solid #444444;
    padding: 8px;
    border-radius: 4px;
    min-height: 20px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #404040;
    border: 1px solid #39ff14;
}
QPushButton:pressed {
    background-color: #39ff14;
    color: #1a1a1a; /* Invert Color When Pressed */
}
QPushButton:disabled {
    background-color: #2a2a2a;
    color: #555555;
}

/* TextEdit, SpinBox, ComboBox */
QTextEdit, QSpinBox, QComboBox {
    background-color: #1a1a1a;
    border: 1px solid #444444;
    border-radius: 4px;
    padding: 5px;
}
QTextEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #39ff14;
}

/* QSpinBox*/
QSpinBox::up-button, QSpinBox::down-button {
    subcontrol-origin: border;
    background-color: #525252;
    border-radius: 4px;
    border-left: 1px solid #242424;
    width: 20px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: #39ff14;
}
QSpinBox::up-button {
    subcontrol-position: top right;
    margin-bottom: 1px;
}
QSpinBox::down-button {
    subcontrol-position: bottom right;
    margin-top: 1px;
}

/* ComboBox */
QComboBox:hover {
    border: 1px solid #39ff14;
}
QComboBox::item:selected {
    border: 1px solid #39ff14;
}

/* Slider */
QSlider::groove:horizontal {
    border: 1px solid #333333;
    background: #1a1a1a;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #39ff14;
    border: 2px solid #39ff14;
    width: 14px;
    height: 14px;
    margin: -7px 0;
    border-radius: 8px;
}

/* ProgressBar */
QProgressBar {
    background: #f0f0f0;
    border: 1px solid #444444;
    border-radius: 4px;
    text-align: center;
    color: #1a1a1a;
    font-weight: bold;
}
QProgressBar::chunk {
    background-color: #39ff14;
    border-radius: 4px;
}

/* VideoDisplayLabel */
VideoDisplayLabel {
    background-color: black;
    border: 2px solid #242424;
    border-radius: 8px;
}
"""


MODEL_MAP = {
    "Qwen/Qwen2.5-VL-3B-Instruct": Qwen2_5_VLForConditionalGeneration, 
    "Qwen/Qwen2.5-VL-7B-Instruct": Qwen2_5_VLForConditionalGeneration, 
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": LlavaOnevisionForConditionalGeneration, 
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": LlavaOnevisionForConditionalGeneration, 
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct": AutoModelForImageTextToText, 
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct": AutoModelForImageTextToText, 
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct": AutoModelForImageTextToText,
    "OpenGVLab/InternVL3_5-1B": AutoModel,
    "OpenGVLab/InternVL3_5-2B": AutoModel,
    "OpenGVLab/InternVL3_5-4B": AutoModel,
    "OpenGVLab/InternVL3_5-8B": AutoModel,
    "openbmb/MiniCPM-V-4_5": AutoModel,
}


class VLM_Inference():
    def __init__(self, params, progress_queue, finished_queue, error_queue):
        super().__init__()
        self.params = params
        self.progress_queue = progress_queue
        self.finished_queue = finished_queue
        self.error_queue = error_queue
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.conversation = None

        print("--- Loading Conversation Template. ---")
        if "InternVL3_5" in self.params["model_name"]:
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(self.params["frame_accumulation"])])
            self.conversation = video_prefix + self.params["user_prompt"]
        elif "MiniCPM-V-4_5" in self.params["model_name"]:
            pass
        else:
            if (
                "Qwen2.5-VL" in self.params["model_name"] or 
                "llava-onevision-qwen2" in self.params["model_name"]
            ):
                conversation = [
                    {
                        "role":"system",
                        "content":[
                            {
                                "type": "text",
                                "text": self.params["system_prompt"]
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
                                "text": self.params["user_prompt"]
                            }
                        ]
                    }
                ]
            else:
                # Actually, these model will directly convert the video into the multiple images without any time handling.
                conversation = [
                    {
                        "role":"system",
                        "content":[
                            {
                                "type": "text",
                                "text": self.params["system_prompt"]
                            }
                        ]
                    }, 
                    {
                        "role":"user",
                        "content":[{"type": "image",} for _ in range(self.params["frame_accumulation"])] + [
                            {
                                "type": "text",
                                "text": self.params["user_prompt"]
                            }
                        ]
                    }
                ]

            self.conversation = conversation
        print("--- Complete Conversation Template. ---")

    def wrap_text_pixelwise(self, text, font, max_width):
        lines = []
        for para in text.split("\n"):
            words = para.split()
            line = ""
            for word in words:
                test_line = f"{line} {word}".strip()
                width = font.getlength(test_line)
                if width <= max_width:
                    line = test_line
                else:
                    lines.append(line)
                    line = word
            lines.append(line)
        return lines
    
    def frame_process(self, PIL_image, text):
        font_path = "./Bitcount_Prop_Single/static/BitcountPropSingle_Roman-SemiBold.ttf"
        font_size = 28
        padding = 10
        line_spacing = 10
        text_area_height = 160
        orig_w, orig_h = PIL_image.size
        new_h = orig_h + text_area_height
        new_img = Image.new("RGB", (orig_w, new_h), color=(34, 139, 34))
        new_img.paste(PIL_image, (0, 0))
        draw = ImageDraw.Draw(new_img)
        font = ImageFont.truetype(font_path, font_size) 
        wrapped_lines = self.wrap_text_pixelwise(text, font, orig_w - 2 * padding)
        text_x = padding
        text_y = orig_h + padding
        for line in wrapped_lines:
            bbox = font.getbbox(line)
            line_height = bbox[3] - bbox[1]
            draw.text((text_x, text_y), line, font=font, fill=(255, 255, 255), spacing=line_spacing)
            text_y += line_height + line_spacing
        cv2_image = cv2.cvtColor(np.asarray(new_img), cv2.COLOR_RGB2BGR)
        return cv2_image

    def build_transform(self, input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def load_video_frames(self, video_frames, input_size=448, max_num=1):
        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)
        for img in video_frames:
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def map_to_nearest_scale(self, values, scale):
        tree = cKDTree(np.asarray(scale)[:, None])
        _, indices = tree.query(np.asarray(values)[:, None])
        return np.asarray(scale)[indices]

    def group_array(self, arr, size):
        return [arr[i:i+size] for i in range(0, len(arr), size)]

    def encode_video(self, video_frames, input_fps, choose_fps=3, force_packing=None):
        MAX_NUM_FRAMES=180 # Indicates the maximum number of frames received after the videos are packed. The actual maximum number of valid frames is MAX_NUM_FRAMES * MAX_NUM_PACKING.
        MAX_NUM_PACKING=3  # indicates the maximum packing number of video frames. valid range: 1-6
        TIME_SCALE = 0.1

        frames = video_frames
        video_duration = len(frames) / input_fps
        if choose_fps * int(video_duration) <= MAX_NUM_FRAMES:
            packing_nums = 1      
        else:
            packing_nums = math.ceil(video_duration * choose_fps / MAX_NUM_FRAMES)
            if packing_nums > MAX_NUM_PACKING:
                packing_nums = MAX_NUM_PACKING
        frame_idx = [i for i in range(0, len(frames))]      
        frame_idx =  np.array(frame_idx)
        if force_packing:
            packing_nums = min(force_packing, MAX_NUM_PACKING)
        # print(f"get video frames={len(frame_idx)}, duration={video_duration}, packing_nums={packing_nums}")
        
        frame_idx_ts = frame_idx / input_fps
        scale = np.arange(0, video_duration, TIME_SCALE)

        frame_ts_id = self.map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
        frame_ts_id = frame_ts_id.astype(np.int32)
        assert len(frames) == len(frame_ts_id)
        frame_ts_id_group = self.group_array(frame_ts_id, packing_nums)
        
        return frames, frame_ts_id_group
    
    def run_inference_on_frames(self, queue, input_fps, curr_frame_number):
        """
        Args:
            queue (deque): Accumulated PIL Frames.
            input_fps (int): Input video FPS.

        Returns:
            inference_text (str): Generated Text.
        """
        # print(f"Processing Frame {curr_frame_number - self.frame_accumulation + 1} ~ Frame {curr_frame_number}")
        # print(f"Used Model Name: {self.model_name}")
        # print(f"System Prompt: {self.system_prompt}")
        # print(f"User Prompt: {self.user_prompt}")

        if "InternVL3_5" in self.params["model_name"]:
            pixel_values, num_patches_list = self.load_video_frames(list(queue), max_num=1)
            pixel_values = pixel_values.to(self.model.dtype).to(self.model.device)
            generation_config = dict(max_new_tokens=256, do_sample=False, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
            with torch.no_grad():
                response, history = self.model.chat(
                    self.tokenizer, 
                    pixel_values, 
                    self.conversation, 
                    generation_config, 
                    num_patches_list=num_patches_list, 
                    history=None, 
                    return_history=True, 
                )
            inference_text = response
            # print(f"The Results of Frame {curr_frame_number-1} and Frame {curr_frame_number}: {response}")
        elif "MiniCPM-V-4_5" in self.params["model_name"]:
            force_packing = None # You can set force_packing to ensure that 3D packing is forcibly enabled; otherwise, encode_video will dynamically set the packing quantity based on the duration.
            frames, frame_ts_id_group = self.encode_video(
                list(queue), 
                input_fps, 
                choose_fps=input_fps, 
                force_packing=force_packing)
            msgs = [
                {"role":"user", "content":frames + [self.params["user_prompt"]]}, 
            ]
            generation_config = dict(max_new_tokens=256, do_sample=False, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
            answer = self.model.chat(
                msgs=msgs, 
                system_prompt=self.params["system_prompt"], 
                tokenizer=self.tokenizer, 
                use_image_id=False, 
                max_slice_nums=1, 
                temporal_ids=frame_ts_id_group, 
                generation_config=generation_config, 
            )
            inference_text = answer
            # print(f"The Results of Frame {curr_frame_number-1} and Frame {curr_frame_number}: {answer}")
        else:
            inputs = self.processor(
                text=[self.conversation], 
                images=None, 
                videos=[list(queue)], 
                padding=True, 
                return_tensors="pt", 
            ).to(self.model.device, self.model.dtype)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=256, eos_token_id=self.processor.tokenizer.eos_token_id, pad_token_id=self.processor.tokenizer.eos_token_id, do_sample=False)
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
                output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            inference_text = output_text[0]
            # print(f"The Results of Frame {curr_frame_number-1} and Frame {curr_frame_number}: {output_text[0]}")
        return inference_text
    
    def run(self):
        self.cap = None
        self.out = None
        try:
            video_path = self.params["video_path"]
            start_frame = self.params["start_frame"]
            end_frame = self.params["end_frame"]
            output_fps = self.params["output_fps"]
            frame_accumulation = self.params["frame_accumulation"]
            stride_size = self.params["stride_size"]
            system_prompt = self.params["system_prompt"]
            user_prompt = self.params["user_prompt"]
            model_name = self.params["model_name"]

            try:
                print(f"Loading Processor...")
                if "Qwen2.5-VL" in model_name:
                    # Temporarily fix this setting
                    MIN_PIXELS = None # 224 * 28 * 28
                    MAX_PIXELS = None # 840 * 28 * 28
                    if MIN_PIXELS is not None and MAX_PIXELS is not None:
                        self.processor = AutoProcessor.from_pretrained(
                            model_name, 
                            torch_dtype=torch.bfloat16,
                            trust_remote_code=True, 
                            min_pixels=MIN_PIXELS, 
                            max_pixels=MAX_PIXELS, 
                        )
                    else:
                        self.processor = AutoProcessor.from_pretrained(
                            model_name, 
                            torch_dtype=torch.bfloat16,
                            trust_remote_code=True, 
                        )
                elif "InternVL3_5" in model_name or "MiniCPM-V-4_5" in model_name:
                    pass
                else:
                    self.processor = AutoProcessor.from_pretrained(
                        model_name, 
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True, 
                    )
                if self.processor: self.processor.tokenizer.padding_side = "left"
                if "InternVL3_5" in model_name:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        trust_remote_code=True, 
                        use_fast=False, 
                    )
                elif "MiniCPM-V-4_5" in model_name:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        trust_remote_code=True, 
                    )
                if self.tokenizer: self.tokenizer.padding_side = "left"
                
                print(f"Loading the model: {model_name}")
                self.model = MODEL_MAP[model_name].from_pretrained(
                    model_name, 
                    torch_dtype=torch.bfloat16, 
                    attn_implementation="flash_attention_2", 
                    device_map=f"cuda:{torch.cuda.current_device()}", 
                    trust_remote_code=True, 
                ).eval()
                print(f"Successfully loaded {model_name}.")
            except Exception as e:
                self.error_queue.put(e)
                if self.cap: self.cap.release()
                if self.out: self.out.close()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
                return
            
            if "InternVL3_5" not in model_name and "MiniCPM-V-4_5" not in model_name:
                self.conversation = self.processor.apply_chat_template(
                    self.conversation, tokenize=False, add_generation_prompt=True
                )
            if "InternVL3_5" in model_name:
                self.model.system_message = self.params["system_prompt"]

            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                self.error_queue.put("The video file can't be opened.")
                if self.cap: self.cap.release()
                if self.out: self.out.close()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
                return

            input_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            end_frame = min(end_frame, total_frames - 1)
            total_inference_frames = end_frame - start_frame + 1
            
            output_dir = "inference_results"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_filename = f"result_{os.path.splitext(os.path.basename(video_path))[0]}_{timestamp}.mp4"
            output_video_path = os.path.join(output_dir, output_filename)            

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame_index = start_frame
            queue = deque()
            generated_texts = []
            
            # The QThread signal may be executed first and then "finally", so you must use "with" to close the writer.
            with imageio.get_writer(output_video_path, fps=output_fps) as self.out:
                while current_frame_index <= end_frame:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    queue.append(frame)

                    if len(queue) >= frame_accumulation:
                        inference_text = self.run_inference_on_frames(queue, input_fps, current_frame_index)

                        for index in range(-stride_size, 0):
                            image = self.frame_process(queue[index], inference_text)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            self.out.append_data(image)

                            generated_texts.append({
                                "current_frame": current_frame_index + index + 1,
                                "text": inference_text
                            })
                        
                        for _ in range(stride_size):
                            queue.popleft()
                    elif len(queue) <= frame_accumulation - stride_size:
                        image = self.frame_process(frame, "None")
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        self.out.append_data(image)

                        generated_texts.append({
                            "current_frame": current_frame_index,
                            "text": "None"
                        })

                    current_frame_index += 1
                    progress = int((current_frame_index - start_frame) / total_inference_frames * 100)
                    self.progress_queue.put(progress)

                if len(queue) > frame_accumulation - stride_size:
                    for index in range(-(len(queue) - frame_accumulation + stride_size), 0):
                        image = self.frame_process(queue[index], "None")
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        self.out.append_data(image)

                        generated_texts.append({
                            "current_frame": current_frame_index - 1 + index + 1,
                            "text": "None"
                        })
            
            self.finished_queue.put((output_video_path, generated_texts))
        except Exception as e:
            self.error_queue.put(e)
        finally:
            if self.cap: self.cap.release()
            if self.out: self.out.close()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()


def run_VLM_in_process(params, progress_queue, finished_queue, error_queue):
    try:
        VLM_process = VLM_Inference(params, progress_queue, finished_queue, error_queue)
        VLM_process.run()
    except Exception as e:
        if error_queue.empty():
            error_queue.put(e)


class InferenceWorker(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str, list) # (output_video_path, generated_texts)
    error_signal = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.is_running = True

    def stop(self):
        self.is_running = False
    
    def poll_progress(self):
        while not self.progress_queue.empty():
            value = self.progress_queue.get()
            self.progress_signal.emit(value)

        if not self.process.is_alive():
            if not self.error_queue.empty():
                e = self.error_queue.get()
                print(e)
                self.error_signal.emit(f"An error occurred during processing: {e}")
            else:
                self.progress_signal.emit(100)
                if self.is_running:
                    output_video_path, generated_texts = self.finished_queue.get()
                    self.finished_signal.emit(output_video_path, generated_texts)
            self.quit()

    def run(self):
        try:
            self.progress_queue = mp.Queue()
            self.finished_queue = mp.Queue()
            self.error_queue = mp.Queue()

            self.process = mp.Process(target=run_VLM_in_process, args=(self.params, self.progress_queue, self.finished_queue, self.error_queue))
            self.process.start()

            self.timer = QTimer()
            self.timer.timeout.connect(self.poll_progress)
            self.timer.start(100)

            self.exec()
            
        except Exception as e:
            print(e)
            self.error_signal.emit(f"An error occurred during processing: {e}")


def run_Grounded_SAM2_steps(params, progress_queue, error_queue):
    try:
        # `input_path` a path of video 
        input_path = params["input_path"]
        output_path = params["output_path"]
        fps = params["fps"]
        generate_box = params.get("generate_box", True)
        generate_label = params.get("generate_label", False)
        generate_mask = params.get("generate_mask", False)
        # setup the input image and text prompt for SAM 2 and Grounding DINO
        # VERY important: text queries need to be lowercased + end with a dot
        text = params["text"]
        draw_color = params["draw_color"]
        box_thickness = params["box_thickness"]

        cap = None
        out = None

        """
        Step 1: Environment settings and model initialization
        """
        sam2_checkpoint = "sam2.1_hiera_large.pt"
        sam2p1_cfg_map = {
            "sam2.1_hiera_tiny.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "sam2.1_hiera_small.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml",
        }
        model_cfg = sam2p1_cfg_map[sam2_checkpoint]
        if not os.path.exists(sam2_checkpoint):
            # SAM 2.1 checkpoints
            SAM2p1_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
            sam2p1_hiera_t_url = f"{SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt"
            sam2p1_hiera_s_url = f"{SAM2p1_BASE_URL}/sam2.1_hiera_small.pt"
            sam2p1_hiera_b_plus_url = f"{SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt"
            sam2p1_hiera_l_url = f"{SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"
            print(f"Downloading {sam2_checkpoint} checkpoint...")
            try:
                command = f"wget {sam2p1_hiera_l_url} -O {sam2_checkpoint}"
                subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                error_queue.put(f"An error occurred: {e.stderr.decode()}\n" + f"Failed to download {sam2_checkpoint} checkpoint.")
                return

        torch.autocast(device_type=f"cuda:{torch.cuda.current_device()}", dtype=torch.bfloat16).__enter__()

        video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        # init video predictor state
        try:
            inference_state = video_predictor.init_state(video_path=input_path)
        except Exception as e:
            error_queue.put(f"Error - {e} or The problem may be that the video is too long and causes out of memory (DRAM).")
            if cap: cap.release()
            if out: out.close()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            return
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        image_predictor = SAM2ImagePredictor(sam2_image_model)

        # init grounding dino model from huggingface
        model_id = "IDEA-Research/grounding-dino-base"
        device = f"cuda:{torch.cuda.current_device()}"
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        # read all the frame names from video
        images = []
        cap = cv2.VideoCapture(input_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        ann_frame_idx = -1  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        progress_queue.put(20)

        """
        Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
        """
        # Try Grounding DINO on every frame until objects are detected.
        for idx in range(ann_frame_idx+1, len(images)):
            print(f"Attempting detection on frame {idx}...")
            # prompt grounding dino to get the box coordinates on specific frame
            image = images[idx]

            # run Grounding DINO on the image
            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = grounding_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.25,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )

            # Apply NMS
            boxes_raw = results[0]["boxes"]
            scores_raw = results[0]["scores"]
            labels_raw = results[0]["labels"]

            mask = np.array([label != "" for label in labels_raw])
            boxes_raw = boxes_raw[mask]
            scores_raw = scores_raw[mask]
            labels_raw = [label for i, label in enumerate(labels_raw) if mask[i]]

            if boxes_raw.shape[0] == 0:
                continue
            
            print(f"Success! Objects are detected on frame {idx}. Starting tracking from here.")
            ann_frame_idx = idx

            unique_labels = sorted(list(set(labels_raw)))
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            class_ids_raw = np.array([label_to_id[label] for label in labels_raw])
            
            detections_gd = sv.Detections(
                xyxy=boxes_raw.cpu().numpy(), 
                confidence=scores_raw.cpu().numpy(), 
                class_id=class_ids_raw, 
                data={'labels': labels_raw}
            )
            NMS_IOU_THRESHOLD = 0.5
            detections_filtered = detections_gd.with_nms(threshold=NMS_IOU_THRESHOLD)
            print(f"Originally detect {len(detections_gd)} bounding boxes. Remain {len(detections_filtered)} bounding boxes after NMS.")

            # process the detection results
            input_boxes = detections_filtered.xyxy
            OBJECTS = detections_filtered.data['labels']
            break
        
        if ann_frame_idx == -1:
            error_queue.put("Grounding DINO did not detect any objects on all frames! Exiting.")
            if cap: cap.release()
            if out: out.close()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            return
        
        # prompt SAM image predictor to get the mask for the object
        image_predictor.set_image(np.array(images[ann_frame_idx].convert("RGB")))

        progress_queue.put(40)

        """
        Step 3: Register each object's positive points to video predictor with seperate add_new_points call
        """
        # Using box prompt
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )

        progress_queue.put(60)
        
        """
        Step 4: Propagate the video predictor to get the segmentation results for each frame
        """
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for idx, (out_frame_idx, out_obj_ids, out_mask_logits) in enumerate(video_predictor.propagate_in_video(inference_state)):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            progress_queue.put(int(60 + (idx + 1) / len(images) * 20))
        
        """
        Step 5: Draw the segment results and save video
        """
        default_palette = sv.ColorPalette.LEGACY
        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
        for progress, (frame_idx, segments) in enumerate(video_segments.items()):
            img = cv2.cvtColor(np.asarray(images[frame_idx]), cv2.COLOR_RGB2BGR)
            
            object_ids = list(segments.keys())
            if not object_ids:
                continue

            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)
            
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                mask=masks, # (n, h, w)
                tracker_id=np.array(object_ids, dtype=np.int32),
            )
            annotated_frame = img.copy()
            if generate_box:
                box_annotator = sv.BoxAnnotator(
                    color=sv.Color(r=draw_color[0], g=draw_color[1], b=draw_color[2]) if len(object_ids) == 1 else default_palette, 
                    thickness=box_thickness,
                    color_lookup=sv.ColorLookup.TRACK
                )
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                if generate_label:
                    label_annotator = sv.LabelAnnotator(
                        color=sv.Color(r=draw_color[0], g=draw_color[1], b=draw_color[2]) if len(object_ids) == 1 else default_palette, 
                        text_color=sv.Color.BLACK, 
                        color_lookup=sv.ColorLookup.TRACK
                    )
                    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
            if generate_mask:
                mask_annotator = sv.MaskAnnotator(
                    color=sv.Color(r=draw_color[0], g=draw_color[1], b=draw_color[2]) if len(object_ids) == 1 else default_palette, 
                    color_lookup=sv.ColorLookup.TRACK
                )
                annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            images[frame_idx] = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

            progress_queue.put(int(80 + (progress + 1) / len(video_segments) * 10))
        
        # The QThread signal may be executed first and then "finally", so you must use "with" to close the writer.
        with imageio.get_writer(output_path, fps=fps) as out:
            for progress, image in enumerate(images):
                out.append_data(np.asarray(image))

                progress_queue.put(int(90 + (progress + 1) / len(images) * 10))

    except Exception as e:
        error_queue.put(e)
    finally:
        if out: out.close()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


class VideoProcessingWorker(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str) # output_path
    error_signal = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.is_running = True

    def stop(self):
        self.is_running = False
        
    def run(self):
        mapping = {
            "Clip": self.run_clip_operation,
            "Crop": self.run_crop_operation,
            "Resize": self.run_resize_operation,
            "Draw": self.run_draw_operation,
            "Grounded-SAM2-Tracking": self.run_Grounded_SAM2_operation,
        }
        operation = self.params.get("operation", None)
        if operation and operation in mapping:
            mapping[operation]()
        else:
            self.error_signal.emit(f"Unknown operation: {operation}")

    def run_clip_operation(self):
        input_path = self.params["input_path"]
        output_path = self.params["output_path"]
        start = self.params["start"]
        end = self.params["end"]
        fps = self.params["fps"]
        if start > end:
            self.error_signal.emit(f"Error during clipping operation: The start frame number must be less than or equal to the end frame number.")

        cmd = [
            "ffmpeg", "-i", input_path, 
            "-vf", f"select='between(n\,{start}\,{end})',setpts=N/{fps}/TB", 
            "-af", f"aselect='between(n\,{start}\,{end})',asetpts=N/SR/TB", 
            "-r", f"{fps}", output_path, "-y"]

        try:
            with FfmpegProgress(cmd) as ff:
                for ff_progress in ff.run_command_with_progress():
                    progress = int(ff_progress)
                    self.progress_signal.emit(progress)

            if self.is_running:
                self.finished_signal.emit(output_path)
        except Exception as e:
            print(e)
            self.error_signal.emit(f"Error during clipping operation: {e}")
    
    def run_crop_operation(self):
        input_path = self.params["input_path"]
        output_path = self.params["output_path"]
        w = self.params["width"]
        h = self.params["height"]
        coords = self.params["coords"]

        cmd = ["ffmpeg", "-i", input_path, "-vf", f"crop={w}:{h}:{coords[0]}:{coords[1]}", "-c:a", "copy", output_path, "-y"]

        try:
            with FfmpegProgress(cmd) as ff:
                for ff_progress in ff.run_command_with_progress():
                    progress = int(ff_progress)
                    self.progress_signal.emit(progress)

            if self.is_running:
                self.finished_signal.emit(output_path)
        except Exception as e:
            print(e)
            self.error_signal.emit(f"Error during cropping operation: {e}")

    def run_resize_operation(self):
        input_path = self.params["input_path"]
        output_path = self.params["output_path"]
        w = self.params["width"]
        h = self.params["height"]

        cmd = ["ffmpeg", "-i", input_path, "-vf", f"scale={w}:{h}", "-c:a", "copy", output_path, "-y"]

        try:
            with FfmpegProgress(cmd) as ff:
                for ff_progress in ff.run_command_with_progress():
                    progress = int(ff_progress)
                    self.progress_signal.emit(progress)

            if self.is_running:
                self.finished_signal.emit(output_path)
        except Exception as e:
            print(e)
            self.error_signal.emit(f"Error during resizing operation: {e}")

    def run_draw_operation(self):
        input_path = self.params["input_path"]
        output_path = self.params["output_path"]
        shape = self.params["shape"]
        coords = self.params["coords"]
        color = self.params["color"]
        thickness = self.params["thickness"]

        cap = None
        out = None
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                self.error_signal.emit("Failed to open source video for drawing.")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # The QThread signal may be executed first and then "finally", so you must use "with" to close the writer.
            with imageio.get_writer(output_path, fps=fps) as out:
                for i in range(total_frames):
                    if not self.is_running:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if shape == "Rectangle":
                        pt1 = (coords[0], coords[1])
                        pt2 = (coords[0] + coords[2], coords[1] + coords[3])
                        cv2.rectangle(frame, pt1, pt2, color, thickness)
                    elif shape == "Line":
                        pt1 = (coords[0], coords[1])
                        pt2 = (coords[2], coords[3])
                        cv2.line(frame, pt1, pt2, color, thickness)
                    
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    out.append_data(image)
                    progress = int((i + 1) / total_frames * 100)
                    self.progress_signal.emit(progress)

            if self.is_running:
                self.finished_signal.emit(output_path)
        except Exception as e:
            print(e)
            self.error_signal.emit(f"Error during drawing operation: {e}")
        finally:
            if cap: cap.release()
            if out: out.close()
    
    def poll_progress(self):
        while not self.progress_queue.empty():
            value = self.progress_queue.get()
            self.progress_signal.emit(value)
        
        if not self.process.is_alive():
            if not self.error_queue.empty():
                e = self.error_queue.get()
                print(e)
                self.error_signal.emit(f"Error during Grounded-SAM2-Tracking operation: {e}")
            else:
                self.progress_signal.emit(100)
                if self.is_running:
                    self.finished_signal.emit(self.output_path)
            self.quit()
    
    def run_Grounded_SAM2_operation(self):
        try:
            self.progress_queue = mp.Queue()
            self.error_queue = mp.Queue()
            self.output_path = self.params["output_path"]

            self.process = mp.Process(target=run_Grounded_SAM2_steps, args=(self.params, self.progress_queue, self.error_queue))
            self.process.start()

            self.timer = QTimer()
            self.timer.timeout.connect(self.poll_progress)
            self.timer.start(100)

            self.exec()

        except Exception as e:
            print(e)
            self.error_signal.emit(f"Error during Grounded-SAM2-Tracking operation: {e}")


class VideoDisplayLabel(QLabel):
    mouse_moved_signal = pyqtSignal(QPoint)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.original_video_size = None

    def mouseMoveEvent(self, event):
        self.mouse_moved_signal.emit(event.pos())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.setToolTip("")
        super().leaveEvent(event)

    def set_original_video_size(self, size):
        self.original_video_size = size


class VideoValidatorThread(QThread):
    progress_update = pyqtSignal(int)
    validation_finished = pyqtSignal(bool, int, str) # is_valid, readable_frames, error_message

    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self._is_running = True
    
    def stop(self):
        self._is_running = False

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not self._is_running or not cap.isOpened():
                self.validation_finished.emit(False, 0, "Could not open the video file.")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                self.validation_finished.emit(False, 0, "Video file contains no frames.")
                return

            readable_frames = 0
            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                readable_frames += 1
                progress = int((readable_frames / total_frames) * 100)
                self.progress_update.emit(progress)
            
            if not self._is_running:
                return

            is_valid = (readable_frames == total_frames)
            error_msg = "" if is_valid else f"File may be corrupt. Header reports {total_frames} frames, but only {readable_frames} were readable."
            self.validation_finished.emit(is_valid, readable_frames, error_msg)

        finally:
            if cap:
                cap.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 Video Inference Tool")
        self.setGeometry(100, 100, 1600, 900)

        self.video_path = None
        self.cap = None

        self.edited_video_path = None
        self.edited_cap = None
        self.temp_dir = os.path.join(os.getcwd(), "video_modification_temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.undo_stack = []
        self.redo_stack = []

        self.processed_cap = None

        self.draw_color = QColor(Qt.GlobalColor.red)
        self.gsam2_draw_color = QColor(Qt.GlobalColor.red)
        self.total_frames = 0
        self.video_width = 0
        self.video_height = 0
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.update_frame)
        self.is_playing = False
        self.current_video_source = "original" # "original", "edited", "processed"
        
        self.inference_worker = None
        self.video_worker = None
        self.inference_data = []

        self.gpuinfo_initialized = False
        self.total_vram_gb = 0
        self.handle = None
        self.init_gpuinfo()

        self.initUI()
    
    def init_gpuinfo(self):
        try:
            nvmlInit()
            current_device_index = torch.cuda.current_device()
            self.handle = nvmlDeviceGetHandleByIndex(current_device_index)
            info = nvmlDeviceGetMemoryInfo(self.handle)
            self.total_vram_gb = info.total / (1024**3)
            self.gpuinfo_initialized = True
            print(f"Successfully initialized info for GPU {current_device_index}. Total VRAM: {self.total_vram_gb:.2f} GB")
        except Exception as e:
            print(f"Failed to initialize pynvml: {e}")
            self.nvml_initialized = False

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left UI
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(650)

        file_group = QGroupBox("1. Video Loading")
        file_layout = QVBoxLayout()
        self.load_video_btn = QPushButton(" Choose A Video")
        self.load_video_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.load_video_btn.clicked.connect(self.load_video)
        self.video_path_label = QLabel("No video is loaded yet.")
        self.video_path_label.setWordWrap(True)
        file_layout.addWidget(self.load_video_btn)
        file_layout.addWidget(self.video_path_label)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        edit_group = QGroupBox("2. Video Editing Tools")
        edit_layout = QVBoxLayout()
        self.edit_tabs = QTabWidget()
        self.edit_tabs.setEnabled(False)

        clip_widget = QWidget()
        clip_layout = QFormLayout(clip_widget)
        self.clip_start_spin = QSpinBox()
        self.clip_end_spin = QSpinBox()
        self.clip_fps_spin = QSpinBox()
        self.apply_clip_btn = QPushButton("Apply Clipping")
        self.apply_clip_btn.clicked.connect(self.apply_clipping)
        clip_layout.addRow("Start Frame Number:", self.clip_start_spin)
        clip_layout.addRow("End Frame Number:", self.clip_end_spin)
        clip_layout.addRow("FPS of the edited video:", self.clip_fps_spin)
        clip_layout.addRow(self.apply_clip_btn)
        
        crop_widget = QWidget()
        crop_layout = QFormLayout(crop_widget)
        self.crop_w_spin = QSpinBox()
        self.crop_h_spin = QSpinBox()
        self.crop_x_spin = QSpinBox()
        self.crop_y_spin = QSpinBox()
        self.apply_crop_btn = QPushButton("Apply Cropping")
        self.apply_crop_btn.clicked.connect(self.apply_cropping)
        crop_layout.addRow("Width:", self.crop_w_spin)
        crop_layout.addRow("Height:", self.crop_h_spin)
        crop_layout.addRow("X offset:", self.crop_x_spin)
        crop_layout.addRow("Y offset:", self.crop_y_spin)
        crop_layout.addRow(self.apply_crop_btn)

        resize_widget = QWidget()
        resize_layout = QFormLayout(resize_widget)
        self.resize_w_spin = QSpinBox()
        self.resize_h_spin = QSpinBox()
        self.apply_resize_btn = QPushButton("Apply Resizing")
        self.apply_resize_btn.clicked.connect(self.apply_resizing)
        resize_layout.addRow("New Width:", self.resize_w_spin)
        resize_layout.addRow("New Height:", self.resize_h_spin)
        resize_layout.addRow(self.apply_resize_btn)

        draw_widget = QWidget()
        draw_layout = QFormLayout(draw_widget)
        self.draw_shape_combo = QComboBox()
        self.draw_shape_combo.addItems(["Rectangle", "Line"])
        self.draw_x1_spin = QSpinBox()
        self.draw_y1_spin = QSpinBox()
        self.draw_x2_width_spin = QSpinBox()
        self.draw_y2_height_spin = QSpinBox()
        self.draw_thickness_spin = QSpinBox()
        self.draw_thickness_spin.setValue(3)
        
        draw_color_layout = QHBoxLayout()
        self.draw_pick_color_btn = QPushButton("Pick Color")
        self.draw_pick_color_btn.clicked.connect(self.draw_pick_color)
        self.draw_color_preview_label = QLabel()
        self.draw_color_preview_label.setFixedSize(50, 20)
        self.draw_update_color_preview()
        draw_color_layout.addWidget(self.draw_pick_color_btn)
        draw_color_layout.addWidget(self.draw_color_preview_label)
        
        self.apply_draw_btn = QPushButton("Apply Drawing")
        self.apply_draw_btn.clicked.connect(self.apply_drawing)
        
        draw_layout.addRow("Shape:", self.draw_shape_combo)
        draw_layout.addRow("X1(Top-Left)/X1:", self.draw_x1_spin)
        draw_layout.addRow("Y1(Top-Left)/Y1:", self.draw_y1_spin)
        draw_layout.addRow("Width/X2:", self.draw_x2_width_spin)
        draw_layout.addRow("Height/Y2:", self.draw_y2_height_spin)
        draw_layout.addRow("Thickness:", self.draw_thickness_spin)
        draw_layout.addRow(draw_color_layout)
        draw_layout.addRow(self.apply_draw_btn)

        gsam2_widget = QWidget()
        gsam2_layout = QFormLayout(gsam2_widget)
        self.gsam_option_combo = QComboBox()
        self.gsam_option_combo.addItems(["Only Boxes", "Only Masks", "Only Boxes with Labels", "Boxes and Masks", "Boxes and Masks with Labels"])
        self.gsam2_text_prompt_lineedit = QLineEdit()
        self.gsam2_text_prompt_lineedit.setPlaceholderText("Keyword (e.g., \"person.\" or \"white car.\")")
        self.gsam2_fps_spin = QSpinBox()
        self.gsam_note_label = QLabel("Notes:\n1. This is Grounded SAM 2 Video Object Tracking.\n2. The video shouldn't be too long, otherwise it will cause out of memory.\n3. It is important that text prompt must be lowercased + end with a dot.")
        self.gsam_note_label.setStyleSheet("color: red")
        self.apply_gsam2_btn = QPushButton("Apply Grounded-SAM2-Tracking")
        self.apply_gsam2_btn.clicked.connect(self.apply_Grounded_SAM2)
        self.gsam2_box_thickness_spin = QSpinBox()
        self.gsam2_box_thickness_spin.setValue(3)
        
        gsam2_draw_color_layout = QHBoxLayout()
        self.gsam2_box_pick_color_btn = QPushButton("Pick Draw Color (If Only One Object Is Detected)")
        self.gsam2_box_pick_color_btn.clicked.connect(self.gsam2_box_pick_color)
        self.gsam2_draw_color_preview_label = QLabel()
        self.gsam2_draw_color_preview_label.setFixedSize(50, 20)
        self.gsam2_box_update_color_preview()
        gsam2_draw_color_layout.addWidget(self.gsam2_box_pick_color_btn)
        gsam2_draw_color_layout.addWidget(self.gsam2_draw_color_preview_label)
        
        gsam2_layout.addRow("Generated Type:", self.gsam_option_combo)
        gsam2_layout.addRow("Text Prompt:", self.gsam2_text_prompt_lineedit)
        gsam2_layout.addRow("FPS of the edited video:", self.gsam2_fps_spin)
        gsam2_layout.addRow("Boxes Thickness:", self.gsam2_box_thickness_spin)
        gsam2_layout.addRow(gsam2_draw_color_layout)
        gsam2_layout.addRow(self.gsam_note_label)
        gsam2_layout.addRow(self.apply_gsam2_btn)
        
        self.edit_tabs.addTab(clip_widget, "Clip")
        self.edit_tabs.addTab(crop_widget, "Crop")
        self.edit_tabs.addTab(resize_widget, "Resize")
        self.edit_tabs.addTab(draw_widget, "Draw")
        self.edit_tabs.addTab(gsam2_widget, "Grounded-SAM2")
        
        edit_buttons_layout = QHBoxLayout()
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
        self.undo_btn.clicked.connect(self.undo_edit)
        self.undo_btn.setEnabled(False)
        edit_buttons_layout.addWidget(self.undo_btn)
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward))
        self.redo_btn.clicked.connect(self.redo_edit)
        self.redo_btn.setEnabled(False)
        edit_buttons_layout.addWidget(self.redo_btn)
        self.reset_edits_btn = QPushButton("Reset All Edits")
        self.reset_edits_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.reset_edits_btn.clicked.connect(self.reset_edits)
        self.reset_edits_btn.setEnabled(False)
        edit_buttons_layout.addWidget(self.reset_edits_btn)
        self.save_edited_btn = QPushButton("Save Edited Video")
        self.save_edited_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save_edited_btn.clicked.connect(self.save_edited_video)
        self.save_edited_btn.setEnabled(False)
        edit_buttons_layout.addWidget(self.save_edited_btn)
        
        edit_layout.addWidget(self.edit_tabs)
        edit_layout.addLayout(edit_buttons_layout)
        edit_group.setLayout(edit_layout)
        left_layout.addWidget(edit_group)

        params_group = QGroupBox("3. Inference Parameters Setting")
        params_form_layout = QFormLayout()
        self.inference_source_combo = QComboBox()
        self.inference_source_combo.addItems(["Original Video", "Edited Video"])
        self.inference_source_combo.setEnabled(False)
        self.inference_source_combo.currentTextChanged.connect(self.start_and_end_frame_slider_setting)
        params_form_layout.addRow("Inference Source:", self.inference_source_combo)
        self.start_frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.start_frame_slider.setEnabled(False)
        self.start_frame_slider.valueChanged.connect(self.update_start_frame_label)
        self.start_frame_label = QLabel("0")
        start_frame_layout = QHBoxLayout()
        start_frame_layout.addWidget(self.start_frame_slider)
        start_frame_layout.addWidget(self.start_frame_label)
        self.end_frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.end_frame_slider.setEnabled(False)
        self.end_frame_slider.valueChanged.connect(self.update_end_frame_label)
        self.end_frame_label = QLabel("0")
        end_frame_layout = QHBoxLayout()
        end_frame_layout.addWidget(self.end_frame_slider)
        end_frame_layout.addWidget(self.end_frame_label)
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        self.fps_spinbox.setValue(10)
        self.accumulation_spinbox = QSpinBox()
        self.accumulation_spinbox.setRange(1, 300)
        self.accumulation_spinbox.setValue(10)
        self.stride_spinbox = QSpinBox()
        self.stride_spinbox.setRange(1, 100)
        self.stride_spinbox.setValue(2)
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODEL_MAP.keys()))
        params_form_layout.addRow("Inference start frame:", start_frame_layout)
        params_form_layout.addRow("Inference end frame:", end_frame_layout)
        params_form_layout.addRow("FPS of the generated video:", self.fps_spinbox)
        params_form_layout.addRow("The number of frames used \nfor one inference: (sliding window size)", self.accumulation_spinbox)
        params_form_layout.addRow("How many images to wait \nfor to make an inference: (stride)", self.stride_spinbox)
        params_form_layout.addRow("Choose a model:", self.model_combo)
        params_group.setLayout(params_form_layout)
        left_layout.addWidget(params_group)

        prompt_group = QGroupBox("4. Model Prompt")
        prompt_layout = QVBoxLayout()
        self.system_prompt_edit = QTextEdit()
        self.system_prompt_edit.setPlaceholderText("Please input system prompt...")
        # self.system_prompt_edit.setText("You are a helpful assistant.")
        self.system_prompt_edit.setText("You are a precise visual reasoning assistant. Focus **only** on the person inside the red box. Ignore **all other people, objects, or tools** in the frame, even if they are close or visible. Your task is to identify **only the tool being used by the person inside the red box**. If no tool is visible or if the tool cannot be clearly identified, return 'Unknown'. Do **not** consider tools being used by others, regardless of their proximity. Do **not** guess or provide explanations. Keep your response concise and in the specified format.")
        self.system_prompt_edit.setMaximumHeight(120)
        self.user_prompt_edit = QTextEdit()
        self.user_prompt_edit.setPlaceholderText("Please input user prompt...")
        # self.user_prompt_edit.setText("Describe the video.")
        self.user_prompt_edit.setText("Analyze the given video frame. Focus only on the person inside the red box.\nIdentify the tool being used by this person, if any, from the following options: [Drill, Marker Pen, Brush or Cotton Swab for Oiling, Scanner, Small Manual Wrench, Unknown]. If no tool is being used or the tool cannot be identified, return 'Unknown'.\n\nOutput format:\nTool: <your answer>")
        self.user_prompt_edit.setMaximumHeight(120)
        prompt_layout.addWidget(QLabel("System Prompt:"))
        prompt_layout.addWidget(self.system_prompt_edit)
        prompt_layout.addWidget(QLabel("User Prompt:"))
        prompt_layout.addWidget(self.user_prompt_edit)
        prompt_group.setLayout(prompt_layout)
        left_layout.addWidget(prompt_group)

        action_group = QGroupBox("5. Execute and Results")
        action_layout = QFormLayout()
        action_buttons_layout = QHBoxLayout()
        self.start_inference_btn = QPushButton("Start to Infer")
        self.start_inference_btn.setEnabled(False)
        self.start_inference_btn.clicked.connect(self.start_inference)
        self.stop_inference_or_editing_btn = QPushButton("Stop Inference or Editing")
        self.stop_inference_or_editing_btn.setEnabled(False)
        self.stop_inference_or_editing_btn.setVisible(False)
        self.stop_inference_or_editing_btn.clicked.connect(self.stop_inference_or_editing)
        action_buttons_layout.addWidget(self.start_inference_btn)
        action_buttons_layout.addWidget(self.stop_inference_or_editing_btn)
        action_layout.addRow(action_buttons_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        action_layout.addRow(self.progress_bar)
        self.status_label = QLabel("")
        action_layout.addRow(self.status_label)
        self.inference_text_label = QLabel("The Inference Result of Current Frame:")
        self.inference_text_display = QTextEdit()
        self.inference_text_display.setReadOnly(True)
        self.inference_text_display.setPlaceholderText("When playing the inferred video, the corresponding generated text will be displayed here...")
        self.inference_text_display.setMaximumHeight(120)
        action_layout.addRow(self.inference_text_label)
        action_layout.addRow(self.inference_text_display)
        action_group.setLayout(action_layout)
        left_layout.addWidget(action_group)
        
        left_layout.addStretch()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(left_panel)
        main_layout.addWidget(scroll_area)

        # Right UI
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.video_display_label = VideoDisplayLabel("Please load a video first.")
        self.video_display_label.setObjectName("VideoDisplayLabel")
        self.video_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video_display_label.mouse_moved_signal.connect(self.update_pixel_coords)
        
        slider_spinbox_hbox = QHBoxLayout()
        playback_slider_group = QWidget()
        playback_slider_layout = QVBoxLayout(playback_slider_group)
        playback_slider_layout.setContentsMargins(0,0,0,0)
        self.playback_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_slider.sliderMoved.connect(self.set_position)
        self.playback_slider.setEnabled(False)
        slider_spinbox_hbox.addWidget(self.playback_slider)

        self.frame_label = QLabel("Frame:")
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setEnabled(False)
        self.frame_spinbox.setFixedWidth(80)
        self.frame_spinbox.editingFinished.connect(self.jump_to_frame_from_spinbox)
        slider_spinbox_hbox.addWidget(self.frame_label)
        slider_spinbox_hbox.addWidget(self.frame_spinbox)
        playback_slider_layout.addLayout(slider_spinbox_hbox)

        set_frame_layout = QHBoxLayout()
        self.set_start_btn = QPushButton(" Set Inference Start Frame")
        self.set_start_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogYesButton))
        self.set_start_btn.clicked.connect(self.set_current_frame_as_start)
        self.set_start_btn.setEnabled(False)
        self.set_end_btn = QPushButton(" Set Inference End Frame")
        self.set_end_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogNoButton))
        self.set_end_btn.clicked.connect(self.set_current_frame_as_end)
        self.set_end_btn.setEnabled(False)
        set_frame_layout.addStretch()
        set_frame_layout.addWidget(self.set_start_btn)
        set_frame_layout.addWidget(self.set_end_btn)
        set_frame_layout.addStretch()
        playback_slider_layout.addLayout(set_frame_layout)

        playback_controls = QHBoxLayout()
        self.play_pause_btn = QPushButton(" PLAY")
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.play_pause_btn.setEnabled(False)
        self.video_time_label = QLabel("00:00 / 00:00")
        
        self.video_source_combo = QComboBox()
        self.video_source_combo.addItems(["Original Video", "Edited Video", "Inferred Video"])
        self.video_source_combo.setEnabled(False)
        self.video_source_combo.currentTextChanged.connect(self.switch_video_source)

        playback_controls.addWidget(self.play_pause_btn)
        playback_controls.addWidget(self.video_time_label)
        playback_controls.addStretch(1)
        playback_controls.addWidget(QLabel("Video Type:"))
        playback_controls.addWidget(self.video_source_combo)

        vram_layout = QVBoxLayout()
        vram_info_layout = QHBoxLayout()
        self.vram_label = QLabel("Current GPU VRAM usage of the device (GB):")
        self.vram_label.setStyleSheet("color: #39ff14")
        vram_info_layout.addWidget(self.vram_label)
        self.vram_usage_label = QLabel("N/A")
        self.vram_usage_label.setStyleSheet("color: #39ff14")
        vram_info_layout.addWidget(self.vram_usage_label)
        vram_info_layout.addStretch()
        self.vram_progress_bar = QProgressBar()
        self.vram_progress_bar.setRange(0, 500)
        # self.vram_progress_bar.setTextVisible(False)
        vram_layout.addLayout(vram_info_layout)
        vram_layout.addWidget(self.vram_progress_bar)

        self.vram_timer = QTimer(self)
        self.vram_timer.timeout.connect(self.update_vram_display)
        self.vram_timer.start(500)
        self.update_vram_display()

        right_layout.addWidget(self.video_display_label, 1)
        right_layout.addWidget(playback_slider_group)
        right_layout.addLayout(playback_controls)
        right_layout.addLayout(vram_layout)

        main_layout.addWidget(right_panel, 1)

    def update_vram_display(self):
        if not self.gpuinfo_initialized:
            self.vram_usage_label.setText("N/A (NVIDIA GPU Not Found)")
            self.vram_progress_bar.setEnabled(False)
            return

        try:
            info = nvmlDeviceGetMemoryInfo(self.handle)
            used_gb = info.used / (1024**3)
        except Exception as e:
            self.vram_usage_label.setText("Error reading VRAM")
            print(f"Could not update VRAM info: {e}")
            self.nvml_initialized = False
        
        self.vram_usage_label.setText(f"{used_gb:.2f} / {self.total_vram_gb:.2f} GB")
        percentage = (used_gb / self.total_vram_gb) * 500
        self.vram_progress_bar.setValue(int(percentage))
        self.vram_progress_bar.setToolTip(f"{used_gb:.2f}/{self.total_vram_gb:.2f} GB ({percentage/10:.1f}%)")
    
    def update_pixel_coords(self, pos):
        if not self.video_display_label.pixmap() or self.video_display_label.pixmap().isNull():
            return
        
        label_size = self.video_display_label.size()
        pixmap_size = self.video_display_label.pixmap().size()
        
        if self.video_width == 0 or self.video_height == 0:
            return

        scale_x = label_size.width() / self.video_width
        scale_y = label_size.height() / self.video_height
        scale = min(scale_x, scale_y)
        scaled_w = int(self.video_width * scale)
        scaled_h = int(self.video_height * scale)
        offset_x = (label_size.width() - scaled_w) // 2
        offset_y = (label_size.height() - scaled_h) // 2
        relative_x = pos.x() - offset_x
        relative_y = pos.y() - offset_y

        if 0 <= relative_x <= scaled_w and 0 <= relative_y <= scaled_h:
            px = int(relative_x / scale)
            py = int(relative_y / scale)
            self.video_display_label.setToolTip(f"({px}, {py})")
        else:
            self.video_display_label.setToolTip("")
    
    def cleanup_temp_files(self, keep_history=None):
        if keep_history is None:
            keep_history = []
        history_files = {os.path.basename(p) for p in keep_history if p and self.video_path not in p}
        for filename in os.listdir(self.temp_dir):
            if filename not in history_files:
                try:
                    os.remove(os.path.join(self.temp_dir, filename))
                except OSError as e:
                    print(f"Error removing temp file {os.path.join(self.temp_dir, filename)}: {e}")

    def edit_tabs_range_and_value_setting(self):
        if self.edited_cap:
            cap = self.edited_cap
        else:
            cap = self.cap
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for spin in [self.crop_w_spin, self.crop_x_spin, self.draw_x1_spin, self.draw_x2_width_spin]:
            spin.setRange(0, video_width)
        for spin in [self.crop_h_spin, self.crop_y_spin, self.draw_y1_spin, self.draw_y2_height_spin]:
            spin.setRange(0, video_height)
        self.clip_start_spin.setRange(0, total_frames - 1)
        self.clip_end_spin.setRange(0, total_frames - 1)
        self.clip_fps_spin.setRange(1, 240)
        self.resize_w_spin.setRange(1, 3840)
        self.resize_h_spin.setRange(1, 2160)
        self.gsam2_fps_spin.setRange(1, 240)

        self.clip_start_spin.setValue(0)
        self.clip_end_spin.setValue(total_frames - 1)
        self.clip_fps_spin.setValue(current_fps)
        self.crop_w_spin.setValue(video_width)
        self.crop_h_spin.setValue(video_height)
        self.resize_w_spin.setValue(video_width)
        self.resize_h_spin.setValue(video_height)
        self.gsam2_fps_spin.setValue(current_fps)
    
    def start_and_end_frame_slider_setting(self):
        inference_source = self.inference_source_combo.currentText()
        if inference_source == "Original Video":
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        elif inference_source == "Edited Video":
            total_frames = int(self.edited_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.start_frame_slider.setRange(0, total_frames - 1)
        self.start_frame_slider.setValue(0)
        self.end_frame_slider.setRange(0, total_frames - 1)
        self.end_frame_slider.setValue(total_frames - 1)
    
    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose A Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if not path:
            return
        
        self.temp_path = path
        
        progress_dialog = QProgressDialog("Validating video, please wait...", "Cancel", 0, 100, self)
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setWindowTitle("Video Validation")
        
        self.validator_thread = VideoValidatorThread(self.temp_path)
        self.validator_thread.progress_update.connect(progress_dialog.setValue)
        self.validator_thread.validation_finished.connect(self.on_validation_finished)
        
        self.validation_canceled = False
        progress_dialog.canceled.connect(self.cancel_validation)
        
        self.validator_thread.start()
        progress_dialog.exec()
    
    def cancel_validation(self):
        print("Validation canceled by user.")
        self.validation_canceled = True
        if hasattr(self, 'validator_thread') and self.validator_thread.isRunning():
            self.validator_thread.stop()

    def on_validation_finished(self, is_valid, readable_frames, error_message):
        if self.validation_canceled:
            return
        
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QProgressDialog):
                widget.close()

        if is_valid:
            QMessageBox.information(self, "Validation Successful", "Video file integrity check passed.")
            self.continue_loading_video()
        else:
            QMessageBox.critical(self, "Validation Failed", f"{error_message}\nPlease choose a valid video file.")
    
    def continue_loading_video(self):
        print("Validation successful. Loading video into the application...")
        self.video_path = self.temp_path
        self.video_path_label.setText(os.path.basename(self.video_path))

        self.reset_all_caps()
        self.cleanup_temp_files()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update_undo_redo_buttons()
        self.inference_data = []
        self.inference_text_display.clear()
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "ERROR", "The video file can't be opened.")
            self.video_path = None
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_display_label.set_original_video_size(QPoint(self.video_width, self.video_height))

        self.playback_slider.setRange(0, self.total_frames - 1)
        self.frame_spinbox.setRange(0, self.total_frames - 1)
        self.frame_spinbox.setValue(0)
        
        self.start_frame_slider.setEnabled(True)
        self.end_frame_slider.setEnabled(True)
        self.playback_slider.setEnabled(True)
        self.frame_spinbox.setEnabled(True)
        self.start_inference_btn.setEnabled(True)
        self.play_pause_btn.setEnabled(True)
        self.edit_tabs.setEnabled(True)
        self.reset_edits_btn.setEnabled(False)
        self.save_edited_btn.setEnabled(False)
        
        self.video_source_combo.model().item(1).setEnabled(False) # Disable "Edited Video"
        self.video_source_combo.model().item(2).setEnabled(False) # Disable "Inferred Video"
        self.video_source_combo.setCurrentIndex(0)
        self.video_source_combo.setEnabled(True)

        self.inference_source_combo.model().item(1).setEnabled(False)
        self.inference_source_combo.setCurrentIndex(0)
        self.inference_source_combo.setEnabled(True)
        self.start_and_end_frame_slider_setting()

        self.current_video_source = "original"
        self.switch_video_source("Original Video")
        self.edit_tabs_range_and_value_setting()

    def get_current_editing_source_path(self):
        return self.edited_video_path if self.edited_video_path and os.path.exists(self.edited_video_path) else self.video_path

    def apply_clipping(self):
        source_path = self.get_current_editing_source_path()
        if not source_path: return

        if self.video_worker and self.video_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Another video processing task is already running.")
            return

        output_path = os.path.join(self.temp_dir, f"edited_{str(time.time()).replace('.', '_')}.mp4")
        
        params = {
            "operation": "Clip",
            "input_path": source_path,
            "output_path": output_path,
            "start": self.clip_start_spin.value(),
            "end": self.clip_end_spin.value(),
            "fps": self.clip_fps_spin.value()
        }
        self.video_processing_start(params, "Applying clipping...")

    def apply_cropping(self):
        source_path = self.get_current_editing_source_path()
        if not source_path: return

        if self.video_worker and self.video_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Another video processing task is already running.")
            return

        output_path = os.path.join(self.temp_dir, f"edited_{str(time.time()).replace('.', '_')}.mp4")
        
        params = {
            "operation": "Crop",
            "input_path": source_path,
            "output_path": output_path,
            "width": self.crop_w_spin.value(),
            "height": self.crop_h_spin.value(),
            "coords": (self.crop_x_spin.value(), self.crop_y_spin.value())
        }
        self.video_processing_start(params, "Applying cropping...")

    def apply_resizing(self):
        source_path = self.get_current_editing_source_path()
        if not source_path: return
        
        if self.video_worker and self.video_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Another video processing task is already running.")
            return

        output_path = os.path.join(self.temp_dir, f"edited_{str(time.time()).replace('.', '_')}.mp4")
        
        params = {
            "operation": "Resize",
            "input_path": source_path,
            "output_path": output_path,
            "width": self.resize_w_spin.value(),
            "height": self.resize_h_spin.value()
        }
        self.video_processing_start(params, "Applying resizing...")

    def draw_pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.draw_color = color
            self.draw_update_color_preview()

    def draw_update_color_preview(self):
        pixmap = QPixmap(self.draw_color_preview_label.size())
        pixmap.fill(self.draw_color)
        self.draw_color_preview_label.setPixmap(pixmap)

    def apply_drawing(self):
        source_path = self.get_current_editing_source_path()
        if not source_path: return
        
        if self.video_worker and self.video_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Another video processing task is already running.")
            return

        output_path = os.path.join(self.temp_dir, f"edited_{str(time.time()).replace('.', '_')}.mp4")
        
        params = {
            "operation": "Draw",
            "input_path": source_path,
            "output_path": output_path,
            "shape": self.draw_shape_combo.currentText(),
            "coords": (self.draw_x1_spin.value(), self.draw_y1_spin.value(), 
                       self.draw_x2_width_spin.value(), self.draw_y2_height_spin.value()),
            "color": (self.draw_color.blue(), self.draw_color.green(), self.draw_color.red()), # BGR for OpenCV
            "thickness": self.draw_thickness_spin.value()
        }
        self.video_processing_start(params, "Applying drawing...")
    
    def gsam2_box_pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.gsam2_draw_color = color
            self.gsam2_box_update_color_preview()

    def gsam2_box_update_color_preview(self):
        pixmap = QPixmap(self.gsam2_draw_color_preview_label.size())
        pixmap.fill(self.gsam2_draw_color)
        self.gsam2_draw_color_preview_label.setPixmap(pixmap)

    def apply_Grounded_SAM2(self):
        if not self.gpuinfo_initialized:
            QMessageBox.critical(self, "GPU Error",
                                 "Cannot start Grounded-SAM2 because the NVIDIA GPU could not be detected or initialized.\n\n"
                                 "Please ensure NVIDIA drivers are correctly installed and working.")
            return
        
        source_path = self.get_current_editing_source_path()
        if not source_path: return

        if self.video_worker and self.video_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Another video processing task is already running.")
            return

        output_path = os.path.join(self.temp_dir, f"edited_{str(time.time()).replace('.', '_')}.mp4")
        
        if self.gsam2_text_prompt_lineedit.text() == "":
            QMessageBox.critical(self, "ERROR", "The text prompt can't be empty.")
            return
        text = self.gsam2_text_prompt_lineedit.text().strip()
        if not text.islower() or text.count(".") != len([x for x in text.split(".") if x]):
            QMessageBox.critical(self, "ERROR", "It is important that text prompt must be lowercased + end with a dot.")
            return
        
        generated_type_map = {
            "Only Boxes": {"generate_box": True, "generate_label": False,"generate_mask": False}, 
            "Only Masks": {"generate_box": False, "generate_label": False,"generate_mask": True}, 
            "Only Boxes with Labels": {"generate_box": True, "generate_label": True,"generate_mask": False}, 
            "Boxes and Masks": {"generate_box": True, "generate_label": False,"generate_mask": True}, 
            "Boxes and Masks with Labels": {"generate_box": True, "generate_label": True,"generate_mask": True},
        }
        params = {
            "operation": "Grounded-SAM2-Tracking",
            "input_path": source_path,
            "output_path": output_path,
            "fps": self.gsam2_fps_spin.value(),
            "text": text, 
            "draw_color": (self.gsam2_draw_color.red(), self.gsam2_draw_color.green(), self.gsam2_draw_color.blue()), 
            "box_thickness": self.gsam2_box_thickness_spin.value()
        }
        params |= generated_type_map[self.gsam_option_combo.currentText()]
        self.video_processing_start(params, "Applying Grounded-SAM2-Tracking...")
        
    def update_undo_redo_buttons(self):
        self.undo_btn.setEnabled(bool(self.undo_stack))
        self.redo_btn.setEnabled(bool(self.redo_stack))

    def undo_edit(self):
        current_path = self.get_current_editing_source_path()
        self.redo_stack.append(current_path)
        previous_path = self.undo_stack.pop()
        self.update_after_edit(previous_path, from_undo_redo=True)

    def redo_edit(self):
        current_path = self.get_current_editing_source_path()
        self.undo_stack.append(current_path)
        next_path = self.redo_stack.pop()
        self.update_after_edit(next_path, from_undo_redo=True)
    
    def video_processing_start(self, params, status_text):
        current_source_path = self.get_current_editing_source_path()
        self.undo_stack.append(current_source_path)
        self.redo_stack.clear()
        self.set_ui_enabled(False, status_text)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.video_worker = VideoProcessingWorker(params)
        self.video_worker.progress_signal.connect(self.update_progress)
        self.video_worker.finished_signal.connect(self.video_processing_finished)
        self.video_worker.error_signal.connect(self.video_processing_error)
        self.video_worker.start()

    def video_processing_finished(self, output_path):
        self.cleanup_worker()
        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        self.update_after_edit(output_path)

    def video_processing_error(self, error_message):
        if self.undo_stack:
            self.undo_stack.pop()
        self.update_undo_redo_buttons()
        self.cleanup_worker()
        self.progress_bar.setVisible(False)
        self.set_ui_enabled(True)
        QMessageBox.critical(self, "Processing Error", error_message)

    def update_after_edit(self, new_edited_path, from_undo_redo=False):
        if new_edited_path == self.video_path:
            self.reset_edits(is_undo=True)
            self.update_undo_redo_buttons()
            return
        
        if self.edited_cap and self.edited_cap.isOpened():
            self.edited_cap.release()
        self.edited_video_path = new_edited_path
        self.edited_cap = cv2.VideoCapture(self.edited_video_path)
        if not self.edited_cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open the edited video.")
            self.edited_video_path = None
            return

        self.video_width = int(self.edited_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.edited_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.video_source_combo.model().item(1).setEnabled(True) # "Edited Video"
        self.inference_source_combo.model().item(1).setEnabled(True)
        self.reset_edits_btn.setEnabled(True)
        self.save_edited_btn.setEnabled(True)

        self.video_source_combo.setCurrentIndex(1)
        self.switch_video_source("Edited Video")
        self.edit_tabs_range_and_value_setting()
        self.start_and_end_frame_slider_setting()

        if from_undo_redo:
            self.update_undo_redo_buttons()
    
    def reset_edits(self, is_undo=False):
        if not is_undo:
            self.undo_stack.clear()
            self.redo_stack.clear()
            QMessageBox.information(self, "Success", "All edits have been reset.")
        
        self.cleanup_temp_files(keep_history=self.undo_stack + self.redo_stack)
        self.edited_video_path = None
        if self.edited_cap:
            self.edited_cap.release()
            self.edited_cap = None
        
        self.video_source_combo.model().item(1).setEnabled(False)
        self.inference_source_combo.model().item(1).setEnabled(False)
        self.reset_edits_btn.setEnabled(False)
        self.save_edited_btn.setEnabled(False)
        
        self.video_source_combo.setCurrentIndex(0)
        self.switch_video_source("Original Video")
        self.edit_tabs_range_and_value_setting()
        self.inference_source_combo.setCurrentIndex(0)
        self.start_and_end_frame_slider_setting()
        self.update_undo_redo_buttons()

    def save_edited_video(self):
        if not self.edited_video_path or not os.path.exists(self.edited_video_path):
            QMessageBox.warning(self, "No Video", "There is no edited video to save.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Edited Video", "", "MP4 Video (*.mp4)")
        if save_path:
            try:
                shutil.copy(self.edited_video_path, save_path)
                QMessageBox.information(self, "Success", f"Edited video saved to:\n{save_path}")
            except Exception as e:
                print(e)
                QMessageBox.critical(self, "Save Error", f"Could not save the file: {e}")

    def set_ui_enabled(self, enabled, status_text=""):
        self.edit_tabs.setEnabled(enabled)
        self.load_video_btn.setEnabled(enabled)
        self.start_inference_btn.setEnabled(enabled)
        self.stop_inference_or_editing_btn.setVisible(not enabled)
        self.stop_inference_or_editing_btn.setEnabled(not enabled)
        if enabled:
            self.update_undo_redo_buttons()
        else:
            self.undo_btn.setEnabled(False)
            self.redo_btn.setEnabled(False)
        self.reset_edits_btn.setEnabled(enabled and self.edited_video_path is not None)
        self.save_edited_btn.setEnabled(enabled and self.edited_video_path is not None)
        self.status_label.setText(status_text)
        self.setFocus()
        QApplication.processEvents()

    def reset_all_caps(self):
        if self.cap: self.cap.release()
        if self.edited_cap: self.edited_cap.release()
        if self.processed_cap: self.processed_cap.release()
        self.cap = None
        self.edited_cap = None
        self.processed_cap = None
    
    def update_start_frame_label(self, value):
        self.start_frame_label.setText(str(value))

    def update_end_frame_label(self, value):
        self.end_frame_label.setText(str(value))

    def set_current_frame_as_start(self):
        current_pos = self.playback_slider.value()
        self.start_frame_slider.setValue(current_pos)

    def set_current_frame_as_end(self):
        current_pos = self.playback_slider.value()
        self.end_frame_slider.setValue(current_pos)

    def start_inference(self):
        if not self.gpuinfo_initialized:
            QMessageBox.critical(self, "GPU Error",
                                 "Cannot start inference because the NVIDIA GPU could not be detected or initialized.\n\n"
                                 "Please ensure NVIDIA drivers are correctly installed and working.")
            return
        
        inference_source = self.inference_source_combo.currentText()
        if inference_source == "Original Video":
            inference_video_path = self.video_path
        elif inference_source == "Edited Video":
            inference_video_path = self.edited_video_path
        else:
            QMessageBox.critical(self, "Error", "Invalid inference source.")
            return
            
        if not inference_video_path or not os.path.exists(inference_video_path):
            QMessageBox.critical(self, "Error", f"The selected video for inference does not exist: {inference_source}")
            return
        
        start_frame = self.start_frame_slider.value()
        end_frame = self.end_frame_slider.value()
        if start_frame > end_frame:
            QMessageBox.critical(self, "Value Error", "The inference end frame must be greater than or equal to the inference start frame!")
            return

        frame_accumulation = self.accumulation_spinbox.value()
        stride_size = self.stride_spinbox.value()
        if stride_size > frame_accumulation:
            QMessageBox.critical(self, "Value Error", "The sliding window size must be less than or equal to the number of frames used!")
            return

        params = {
            "video_path": inference_video_path,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "output_fps": self.fps_spinbox.value(),
            "frame_accumulation": frame_accumulation,
            "stride_size": stride_size,
            "system_prompt": self.system_prompt_edit.toPlainText(),
            "user_prompt": self.user_prompt_edit.toPlainText(),
            "model_name": self.model_combo.currentText(),
        }

        self.set_ui_enabled(False, "Inference in progress...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.inference_worker = InferenceWorker(params)
        self.inference_worker.progress_signal.connect(self.update_progress)
        self.inference_worker.finished_signal.connect(self.inference_finished)
        self.inference_worker.error_signal.connect(self.inference_error)
        self.inference_worker.start()

    def stop_inference_or_editing(self):
        self.status_label.setText("Stopping inference or editing...")
        QApplication.processEvents()
        self.cleanup_worker(kill=True)
        print("Inference or editing operation was terminated. Ignore the above error message.")
        self.progress_bar.setVisible(False)
        self.set_ui_enabled(True)
        QMessageBox.information(self, "Successfully Stop", f"Inference or editing operation was terminated.")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def inference_finished(self, output_video_path, generated_texts):
        self.cleanup_worker()
        self.progress_bar.setVisible(False)
        self.set_ui_enabled(True)
        QMessageBox.information(self, "Finish", f"The video inference is completed!\nThe result is saved in:\n{output_video_path}")

        self.inference_data = generated_texts

        if self.processed_cap:
            self.processed_cap.release()
            self.processed_cap = None
        self.processed_cap = cv2.VideoCapture(output_video_path)
        if self.processed_cap.isOpened():
            self.video_source_combo.model().item(2).setEnabled(True)
            self.video_source_combo.setCurrentIndex(2)
            self.switch_video_source("Inferred Video")

    def inference_error(self, error_message):
        self.cleanup_worker()
        self.progress_bar.setVisible(False)
        self.set_ui_enabled(True)
        QMessageBox.critical(self, "Inference Error", error_message)

    def toggle_play_pause(self):
        if not self.get_current_cap(): return
        if self.is_playing:
            self.is_playing = False
            self.playback_timer.stop()
            self.play_pause_btn.setText(" PLAY")
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        else:
            self.is_playing = True
            fps = int(self.get_current_cap().get(cv2.CAP_PROP_FPS))
            self.playback_timer.start(int(1000 / fps))
            self.play_pause_btn.setText(" PAUSE")
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))

    def update_frame(self):
        cap = self.get_current_cap()
        if cap:
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if ret:
                self.display_frame(current_pos, frame)
                self.playback_slider.setValue(current_pos)
            else:
                self.playback_timer.stop()
                self.is_playing = False
                self.play_pause_btn.setText(" PLAY")
                self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind
                self.display_frame(0)
                self.playback_slider.setValue(0)
    
    def jump_to_frame_from_spinbox(self):
        if self.is_playing:
            self.toggle_play_pause()
            
        frame_to_jump = self.frame_spinbox.value()
        self.display_frame(frame_to_jump)

    def display_frame(self, frame_number, frame=None):
        self.playback_slider.setValue(frame_number)

        if frame is None:
            cap = self.get_current_cap()
            if not cap: return
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret: return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_display_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_display_label.setPixmap(scaled_pixmap)

        self.update_time_label(frame_number)
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(frame_number)
        self.frame_spinbox.blockSignals(False)

        if self.current_video_source == "processed" and self.inference_data and frame_number < len(self.inference_data):
            original_video_frame_number = self.inference_data[frame_number]["current_frame"]
            generated_text = self.inference_data[frame_number]["text"]
            self.inference_text_display.setText(f"This frame {frame_number} corresponds to frame {original_video_frame_number} of the {self.inference_source_combo.currentText().lower()}.\nGenerated Text:\n{generated_text}")
        elif self.current_video_source != "processed":
            self.inference_text_display.clear()

    def set_position(self, position):
        if self.is_playing:
            self.toggle_play_pause()

        self.display_frame(position)

    def update_time_label(self, frame_number):
        cap = self.get_current_cap()
        if cap and cap.isOpened():
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_seconds = int(frame_number / fps)
            total_seconds = int(total_frames / fps)
            current_time = time.strftime("%M:%S", time.gmtime(current_seconds))
            total_time = time.strftime("%M:%S", time.gmtime(total_seconds))
            self.video_time_label.setText(f"{current_time} / {total_time}")

    def get_current_cap(self):
        if self.current_video_source == "processed" and self.processed_cap and self.processed_cap.isOpened():
            return self.processed_cap
        elif self.current_video_source == "edited" and self.edited_cap and self.edited_cap.isOpened():
            return self.edited_cap
        return self.cap
    
    def switch_video_source(self, source_text):
        source_map = {"Original Video": "original", "Edited Video": "edited", "Inferred Video": "processed"}
        self.current_video_source = source_map.get(source_text, "original")

        is_original_edited_video = (self.current_video_source == "original" or self.current_video_source == "edited")
        self.set_start_btn.setEnabled(is_original_edited_video)
        self.set_end_btn.setEnabled(is_original_edited_video)
        
        cap = self.get_current_cap()
        if cap and cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.playback_slider.setRange(0, total_frames - 1)
            self.frame_spinbox.setRange(0, total_frames - 1)
            if self.is_playing: self.toggle_play_pause()
            self.display_frame(0)
        else:
            if self.current_video_source != "original":
                QMessageBox.warning(self, "Warning", f"{source_text} is not available. Switching back to Original Video.")
                self.video_source_combo.setCurrentIndex(0)
                self.switch_video_source("Original Video")
                self.edit_tabs_range_and_value_setting()
                self.start_and_end_frame_slider_setting()
            else:
                 self.video_display_label.setText("Video not available.")

    def cleanup_worker(self, kill=False):
        if kill and self.inference_worker and self.inference_worker.process.is_alive():
            os.kill(self.inference_worker.process.pid, signal.SIGKILL)
            self.inference_worker.process.join()
        if self.inference_worker and self.inference_worker.isRunning():
            self.inference_worker.stop()
            self.inference_worker.quit()
            self.inference_worker.wait()
            self.inference_worker.deleteLater()
            self.inference_worker = None
        if kill and self.video_worker and hasattr(self.video_worker, "process") and self.video_worker.process.is_alive():
            os.kill(self.video_worker.process.pid, signal.SIGKILL)
            self.video_worker.process.join()
        if self.video_worker and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker.quit()
            self.video_worker.wait()
            self.video_worker.deleteLater()
            self.video_worker = None
    
    def closeEvent(self, event):
        self.cleanup_worker(kill=True)
        self.reset_all_caps()
        try:
            shutil.rmtree(self.temp_dir)
            print(f"Removed temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error removing temporary directory: {e}")
        event.accept()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())