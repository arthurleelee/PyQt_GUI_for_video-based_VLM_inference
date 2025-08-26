import sys
import os
import cv2
import time
import torch
import numpy as np
import imageio
import gc
import textwrap
import subprocess
import shutil

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QSlider, QSpinBox, QComboBox, QTextEdit, QGroupBox,
    QFormLayout, QMessageBox, QProgressBar, QStyle, QSizePolicy, QTabWidget, QColorDialog, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import QImage, QPixmap, QColor
from collections import deque
from transformers import Qwen2_5_VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration, AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import ImageFont, Image, ImageDraw


STYLE_SHEET = """
/* 全域設定 */
QWidget {
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
    font-size: 16px;
    color: #f0f0f0; /* 亮灰色文字 */
    background-color: #1a1a1a; /* 極暗背景 */
}

/* 主視窗 */
QMainWindow {
    background-color: #121212;
}

/* 群組框 */
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
    color: #39ff14; /* 霓虹綠強調色 */
}

/* 按鈕 */
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
    color: #1a1a1a; /* 按下時反色 */
}
QPushButton:disabled {
    background-color: #2a2a2a;
    color: #555555;
}

/* 文字輸入、SpinBox、下拉選單 */
QTextEdit, QSpinBox, QComboBox {
    background-color: #1a1a1a;
    border: 1px solid #444444;
    border-radius: 4px;
    padding: 5px;
}
QTextEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #39ff14;
}

/* QSpinBox */
QSpinBox::up-button, QSpinBox::down-button {
    subcontrol-origin: border;
    background-color: #333333;
    border-left: 1px solid #242424;
    width: 20px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: #404040;
}
QSpinBox::up-button {
    subcontrol-position: top right;
}
QSpinBox::down-button {
    subcontrol-position: bottom right;
}

/* 滑桿 */
QSlider::groove:horizontal {
    border: 1px solid #333333;
    background: #1a1a1a;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #39ff14;
    border: 2px solid #39ff14;
    width: 14px; /* 稍微調整大小 */
    height: 14px;
    margin: -7px 0;
    border-radius: 8px;
}

/* 進度條 */
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

/* 影片顯示標籤 */
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
}


class InferenceWorker(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str, list) # (output_video_path, generated_texts)
    error_signal = pyqtSignal(str)

    def __init__(self, params, model, processor):
        super().__init__()
        self.params = params
        self.is_running = True
        self.model = model
        self.processor = processor
        self.conversation = None

        print("--- Loading Conversation Template. ---")
        if "Qwen2.5-VL" in self.params['model_name'] or "llava-onevision-qwen2" in self.params['model_name']:
            conversation = [
                {
                    "role":"system",
                    "content":[
                        {
                            "type": "text",
                            "text": self.params['system_prompt']
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
                            "text": self.params['user_prompt']
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
                            "text": self.params['system_prompt']
                        }
                    ]
                }, 
                {
                    "role":"user",
                    "content":[{"type": "image",} for _ in range(self.params['frame_accumulation'])] + [
                        {
                            "type": "text",
                            "text": self.params['user_prompt']
                        }
                    ]
                }
            ]

        self.conversation = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        print("--- Complete Conversation Template. ---")

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
        wrapped_lines = []
        for para in text.split("\n"):
            wrapped = textwrap.fill(para, width=orig_w - 40)
            wrapped_lines.extend(wrapped.split("\n"))
        text_x = padding
        text_y = orig_h + padding
        for line in wrapped_lines:
            bbox = font.getbbox(line)
            line_height = bbox[3] - bbox[1]
            draw.text((text_x, text_y), line, font=font, fill=(255, 255, 255), spacing=line_spacing)
            text_y += line_height + line_spacing
        cv2_image = cv2.cvtColor(np.asarray(new_img), cv2.COLOR_RGB2BGR)
        # OpenCV
        # cv2_image = cv2.cvtColor(np.asarray(PIL_image), cv2.COLOR_RGB2BGR)
        # text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        # text_w, text_h = text_size
        # cv2.rectangle(cv2_image, (10, 30 - text_h - 5), (10 + text_w, 30 + baseline), (34, 139, 34, 0.5), -1)
        # cv2.putText(cv2_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return cv2_image
    
    def stop(self):
        self.is_running = False

    def run_inference_on_frames(self, queue, system_prompt, user_prompt, model_name, curr_frame_number, frame_accumulation):
        """
        Args:
            queue (deque): Accumulated PIL Frames.
            system_prompt (str): System Prompt.
            user_prompt (str): User Prompt.
            model_name (str): Used Model Name.
            curr_frame_number (int): Current Frame Number.

        Returns:
            inference_text (str): Generated Text.
        """
        # print(f"Processing Frame {curr_frame_number - frame_accumulation + 1} ~ Frame {curr_frame_number}")
        # print(f"Used Model Name: {model_name}")
        # print(f"System Prompt: {system_prompt}")
        # print(f"User Prompt: {user_prompt}")

        inputs = self.processor(
            text=[self.conversation], 
            images=None, 
            videos=[list(queue)], 
            padding=True, 
            return_tensors="pt", 
        ).to(self.model.device, self.model.dtype)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=256, eos_token_id=self.processor.tokenizer.eos_token_id, do_sample=False)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        inference_text = output_text[0]

        # print(f"The Results of Frame {curr_frame_number-1} and Frame {curr_frame_number}: {output_text[0]}")
        return inference_text

    def run(self):
        cap = None
        out = None
        try:
            video_path = self.params['video_path']
            start_frame = self.params['start_frame']
            end_frame = self.params['end_frame']
            output_fps = self.params['output_fps']
            frame_accumulation = self.params['frame_accumulation']
            sliding_window_size = self.params['sliding_window_size']
            system_prompt = self.params['system_prompt']
            user_prompt = self.params['user_prompt']
            model_name = self.params['model_name']

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.error_signal.emit("The video file can't be opened.")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            end_frame = min(end_frame, total_frames - 1)
            total_inference_frames = end_frame - start_frame + 1
            
            output_dir = "inference_results"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_filename = f"result_{os.path.splitext(os.path.basename(video_path))[0]}_{timestamp}.mp4"
            output_video_path = os.path.join(output_dir, output_filename)            

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame_index = start_frame
            queue = deque()
            generated_texts = []
            
            # The QThread signal may be executed first and then "finally", so you must use "with" to close the writer.
            with imageio.get_writer(output_video_path, fps=output_fps) as out:
                while self.is_running and current_frame_index <= end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    queue.append(frame)

                    if len(queue) >= frame_accumulation:
                        inference_text = self.run_inference_on_frames(
                            queue, system_prompt, user_prompt, model_name, current_frame_index, frame_accumulation
                        )

                        for index in range(-sliding_window_size, 0):
                            image = self.frame_process(queue[index], inference_text)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            out.append_data(image)

                            generated_texts.append({
                                "current_frame": current_frame_index + index + 1,
                                "text": inference_text
                            })
                        
                        for _ in range(sliding_window_size):
                            queue.popleft()
                    elif len(queue) <= frame_accumulation - sliding_window_size:
                        image = self.frame_process(frame, "None")
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        out.append_data(image)

                        generated_texts.append({
                            "current_frame": current_frame_index,
                            "text": "None"
                        })

                    current_frame_index += 1
                    progress = int((current_frame_index - start_frame) / total_inference_frames * 100)
                    self.progress_signal.emit(progress)

                if len(queue) > frame_accumulation - sliding_window_size:
                    for index in range(-(len(queue) - frame_accumulation + sliding_window_size), 0):
                        image = self.frame_process(queue[index], "None")
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        out.append_data(image)

                        generated_texts.append({
                            "current_frame": current_frame_index - 1 + index + 1,
                            "text": "None"
                        })
            
            if self.is_running:
                self.finished_signal.emit(output_video_path, generated_texts)
        except Exception as e:
            self.error_signal.emit(f"An error occurred during processing: {e}")
        finally:
            if cap: cap.release()
            if out: out.close()
            del self.model
            del self.processor


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
        operation = self.params.get('operation')
        if operation == 'draw':
            self.run_draw_operation()
        else:
            self.error_signal.emit(f"Unknown operation: {operation}")

    def run_draw_operation(self):
        input_path = self.params['input_path']
        output_path = self.params['output_path']
        shape = self.params['shape']
        coords = self.params['coords']
        color = self.params['color']
        thickness = self.params['thickness']

        cap = None
        out = None
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                self.error_signal.emit("Failed to open source video for drawing.")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
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
                    
                    if shape == 'Rectangle':
                        pt1 = (coords[0], coords[1])
                        pt2 = (coords[0] + coords[2], coords[1] + coords[3])
                        cv2.rectangle(frame, pt1, pt2, color, thickness)
                    elif shape == 'Line':
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
            self.error_signal.emit(f"Error during drawing operation: {e}")
        finally:
            if cap: cap.release()
            if out: out.close()


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

        self.processed_cap = None

        self.drawing_color = QColor(Qt.GlobalColor.red)
        self.total_frames = 0
        self.video_width = 0
        self.video_height = 0
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.update_frame)
        self.is_playing = False
        self.current_video_source = 'original' # 'original', 'edited', 'processed'
        
        self.inference_worker = None
        self.video_worker = None
        self.model = None
        self.processor = None
        self.current_model_name = None
        self.inference_data = []

        self.initUI()

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

        crop_widget = QWidget()
        crop_layout = QFormLayout(crop_widget)
        self.crop_w_spin = QSpinBox()
        self.crop_h_spin = QSpinBox()
        self.crop_x_spin = QSpinBox()
        self.crop_y_spin = QSpinBox()
        self.apply_crop_btn = QPushButton("Apply Crop")
        self.apply_crop_btn.clicked.connect(self.apply_crop)
        crop_layout.addRow("Width:", self.crop_w_spin)
        crop_layout.addRow("Height:", self.crop_h_spin)
        crop_layout.addRow("X offset:", self.crop_x_spin)
        crop_layout.addRow("Y offset:", self.crop_y_spin)
        crop_layout.addRow(self.apply_crop_btn)

        resize_widget = QWidget()
        resize_layout = QFormLayout(resize_widget)
        self.resize_w_spin = QSpinBox()
        self.resize_w_spin.setRange(1, 3840)
        self.resize_h_spin = QSpinBox()
        self.resize_h_spin.setRange(1, 2160)
        self.apply_resize_btn = QPushButton("Apply Resize")
        self.apply_resize_btn.clicked.connect(self.apply_resize)
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
        self.draw_thickness_spin.setRange(1, 100)
        self.draw_thickness_spin.setValue(2)
        
        color_layout = QHBoxLayout()
        self.pick_color_btn = QPushButton("Pick Color")
        self.pick_color_btn.clicked.connect(self.pick_draw_color)
        self.color_preview_label = QLabel()
        self.color_preview_label.setFixedSize(50, 20)
        self.update_color_preview()
        color_layout.addWidget(self.pick_color_btn)
        color_layout.addWidget(self.color_preview_label)
        
        self.apply_draw_btn = QPushButton("Apply Drawing")
        self.apply_draw_btn.clicked.connect(self.apply_drawing)
        
        draw_layout.addRow("Shape:", self.draw_shape_combo)
        draw_layout.addRow("X1(Top-Left)/X1:", self.draw_x1_spin)
        draw_layout.addRow("Y1(Top-Left)/Y1:", self.draw_y1_spin)
        draw_layout.addRow("Width/X2:", self.draw_x2_width_spin)
        draw_layout.addRow("Height/Y2:", self.draw_y2_height_spin)
        draw_layout.addRow("Thickness:", self.draw_thickness_spin)
        draw_layout.addRow(color_layout)
        draw_layout.addRow(self.apply_draw_btn)
        
        self.edit_tabs.addTab(crop_widget, "Crop")
        self.edit_tabs.addTab(resize_widget, "Resize")
        self.edit_tabs.addTab(draw_widget, "Draw")
        
        edit_buttons_layout = QHBoxLayout()
        self.save_edited_btn = QPushButton("Save Edited Video As...")
        self.save_edited_btn.clicked.connect(self.save_edited_video)
        self.save_edited_btn.setEnabled(False)
        self.reset_edits_btn = QPushButton("Reset All Edits")
        self.reset_edits_btn.clicked.connect(self.reset_edits)
        self.reset_edits_btn.setEnabled(False)
        edit_buttons_layout.addWidget(self.save_edited_btn)
        edit_buttons_layout.addWidget(self.reset_edits_btn)
        
        edit_layout.addWidget(self.edit_tabs)
        edit_layout.addLayout(edit_buttons_layout)
        edit_group.setLayout(edit_layout)
        left_layout.addWidget(edit_group)

        params_group = QGroupBox("3. Inference Parameters Setting")
        params_form_layout = QFormLayout()
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
        self.sliding_window_spinbox = QSpinBox()
        self.sliding_window_spinbox.setRange(1, 100)
        self.sliding_window_spinbox.setValue(2)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", "llava-hf/llava-onevision-qwen2-7b-ov-hf", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"])
        params_form_layout.addRow("Inference start frame:", start_frame_layout)
        params_form_layout.addRow("Inference end frame:", end_frame_layout)
        params_form_layout.addRow("FPS of the generated video:", self.fps_spinbox)
        params_form_layout.addRow("The number of frames used for one inference:", self.accumulation_spinbox)
        params_form_layout.addRow("One Inference every N frames (sliding window size):", self.sliding_window_spinbox)
        params_form_layout.addRow("Choose a model:", self.model_combo)
        params_group.setLayout(params_form_layout)
        left_layout.addWidget(params_group)

        prompt_group = QGroupBox("4. Model Prompt")
        prompt_layout = QVBoxLayout()
        self.system_prompt_edit = QTextEdit()
        self.system_prompt_edit.setPlaceholderText("Please input system prompt...")
        self.system_prompt_edit.setText("You are a helpful assistant.")
        self.system_prompt_edit.setMaximumHeight(120)
        self.user_prompt_edit = QTextEdit()
        self.user_prompt_edit.setPlaceholderText("Please input user prompt...")
        self.user_prompt_edit.setText("Describe the video.")
        self.user_prompt_edit.setMaximumHeight(120)
        prompt_layout.addWidget(QLabel("System Prompt:"))
        prompt_layout.addWidget(self.system_prompt_edit)
        prompt_layout.addWidget(QLabel("User Prompt:"))
        prompt_layout.addWidget(self.user_prompt_edit)
        prompt_group.setLayout(prompt_layout)
        left_layout.addWidget(prompt_group)

        action_group = QGroupBox("5. Execute and Results")
        action_layout = QFormLayout()
        self.inference_source_combo = QComboBox()
        self.inference_source_combo.addItems(["Original Video", "Edited Video"])
        self.inference_source_combo.setEnabled(False)
        action_layout.addRow("Inference Source:", self.inference_source_combo)
        self.start_inference_btn = QPushButton("Start to Infer")
        self.start_inference_btn.setEnabled(False)
        self.start_inference_btn.clicked.connect(self.start_inference)
        action_layout.addRow(self.start_inference_btn)
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

        right_layout.addWidget(self.video_display_label, 1)
        right_layout.addWidget(playback_slider_group)
        right_layout.addLayout(playback_controls)
        main_layout.addWidget(right_panel, 1)
    
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
    
    def cleanup_temp_files(self):
        if self.edited_video_path and os.path.exists(self.edited_video_path):
            try:
                os.remove(self.edited_video_path)
            except OSError as e:
                print(f"Error removing temp file {self.edited_video_path}: {e}")
        self.edited_video_path = None

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose A Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if not path:
            return
        
        self.video_path = path
        self.video_path_label.setText(os.path.basename(path))

        self.reset_all_caps()
        self.cleanup_temp_files()
        self.inference_data = []
        self.inference_text_display.clear()
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "ERROR", "The video file can't be opened.")
            self.video_path = None
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_display_label.set_original_video_size(QPoint(self.video_width, self.video_height))

        for spin in [self.crop_w_spin, self.crop_h_spin, self.crop_x_spin, self.crop_y_spin, 
                     self.draw_x1_spin, self.draw_x2_width_spin]:
            spin.setRange(0, self.video_width)
        for spin in [self.draw_y1_spin, self.draw_y2_height_spin]:
            spin.setRange(0, self.video_height)
            
        self.crop_w_spin.setValue(self.video_width)
        self.crop_h_spin.setValue(self.video_height)
        self.resize_w_spin.setValue(self.video_width)
        self.resize_h_spin.setValue(self.video_height)

        self.start_frame_slider.setRange(0, self.total_frames - 1)
        self.start_frame_slider.setValue(0)
        self.end_frame_slider.setRange(0, self.total_frames - 1)
        self.end_frame_slider.setValue(self.total_frames - 1)
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
        self.inference_source_combo.model().item(1).setEnabled(False)
        self.video_source_combo.setCurrentIndex(0)
        self.inference_source_combo.setCurrentIndex(0)
        self.video_source_combo.setEnabled(True)
        self.inference_source_combo.setEnabled(True)

        self.current_video_source = 'original'
        self.switch_video_source("Original Video")

    def get_current_editing_source_path(self):
        return self.edited_video_path if self.edited_video_path and os.path.exists(self.edited_video_path) else self.video_path

    def run_ffmpeg_command(self, command):
        self.set_ui_enabled(False, "Processing video...")
        try:
            subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "FFmpeg Error", f"An error occurred: {e.stderr.decode()}")
            return False
        finally:
            self.set_ui_enabled(True)

    def apply_crop(self):
        source_path = self.get_current_editing_source_path()
        if not source_path: return
        
        w = self.crop_w_spin.value()
        h = self.crop_h_spin.value()
        x = self.crop_x_spin.value()
        y = self.crop_y_spin.value()

        output_path = os.path.join(self.temp_dir, f"edited_{str(time.time()).replace('.', '_')}.mp4")
        command = f'ffmpeg -i "{source_path}" -vf "crop={w}:{h}:{x}:{y}" -c:a copy "{output_path}" -y'
        
        if self.run_ffmpeg_command(command):
            self.update_after_edit(output_path)

    def apply_resize(self):
        source_path = self.get_current_editing_source_path()
        if not source_path: return

        w = self.resize_w_spin.value()
        h = self.resize_h_spin.value()
        
        output_path = os.path.join(self.temp_dir, f"edited_{str(time.time()).replace('.', '_')}.mp4")
        command = f'ffmpeg -i "{source_path}" -vf scale={w}:{h} -c:a copy "{output_path}" -y'

        if self.run_ffmpeg_command(command):
            self.update_after_edit(output_path)

    def pick_draw_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.drawing_color = color
            self.update_color_preview()

    def update_color_preview(self):
        pixmap = QPixmap(self.color_preview_label.size())
        pixmap.fill(self.drawing_color)
        self.color_preview_label.setPixmap(pixmap)

    def apply_drawing(self):
        source_path = self.get_current_editing_source_path()
        if not source_path: return
        
        if self.video_worker and self.video_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Another video processing task is already running.")
            return

        output_path = os.path.join(self.temp_dir, f"edited_{str(time.time()).replace('.', '_')}.mp4")
        
        params = {
            'operation': 'draw',
            'input_path': source_path,
            'output_path': output_path,
            'shape': self.draw_shape_combo.currentText(),
            'coords': (self.draw_x1_spin.value(), self.draw_y1_spin.value(), 
                       self.draw_x2_width_spin.value(), self.draw_y2_height_spin.value()),
            'color': (self.drawing_color.blue(), self.drawing_color.green(), self.drawing_color.red()), # BGR for OpenCV
            'thickness': self.draw_thickness_spin.value()
        }
        
        self.set_ui_enabled(False, "Applying drawing...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.video_worker = VideoProcessingWorker(params)
        self.video_worker.progress_signal.connect(self.update_progress)
        self.video_worker.finished_signal.connect(self.video_processing_finished)
        self.video_worker.error_signal.connect(self.video_processing_error)
        self.video_worker.start()

    def video_processing_finished(self, output_path):
        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        self.update_after_edit(output_path)

    def video_processing_error(self, error_message):
        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Processing Error", error_message)

    def update_after_edit(self, new_edited_path):
        self.cleanup_temp_files()
        self.edited_video_path = new_edited_path
        
        if self.edited_cap and self.edited_cap.isOpened():
            self.edited_cap.release()
        
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
    
    def reset_edits(self):
        self.cleanup_temp_files()
        if self.edited_cap:
            self.edited_cap.release()
            self.edited_cap = None
        
        self.video_source_combo.model().item(1).setEnabled(False)
        self.inference_source_combo.model().item(1).setEnabled(False)
        self.reset_edits_btn.setEnabled(False)
        self.save_edited_btn.setEnabled(False)
        
        self.video_source_combo.setCurrentIndex(0)
        QMessageBox.information(self, "Success", "All edits have been reset.")

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
                QMessageBox.critical(self, "Save Error", f"Could not save the file: {e}")

    def set_ui_enabled(self, enabled, status_text=""):
        self.edit_tabs.setEnabled(enabled)
        self.load_video_btn.setEnabled(enabled)
        self.start_inference_btn.setEnabled(enabled)
        self.save_edited_btn.setEnabled(enabled and self.edited_video_path is not None)
        self.reset_edits_btn.setEnabled(enabled and self.edited_video_path is not None)
        self.status_label.setText(status_text)
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
        sliding_window_size = self.sliding_window_spinbox.value()
        if sliding_window_size > frame_accumulation:
            QMessageBox.critical(self, "Value Error", "The sliding window size must be less than or equal to the number of frames used!")
            return
        
        desired_model_name = self.model_combo.currentText()
        if self.current_model_name != desired_model_name:
            print(f"Model switch required. From '{self.current_model_name}' to '{desired_model_name}'.")
            self.set_ui_enabled(False, "Switching model...")
            QApplication.processEvents()

            try:
                if self.model is not None:
                    print(f"Releasing old model: {self.current_model_name}")
                    del self.model
                    del self.processor
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    gc.collect()
                
                print(f"Loading Processor...")
                if "Qwen2.5-VL" in desired_model_name:
                    MIN_PIXELS = 224 * 28 * 28
                    MAX_PIXELS = 840 * 28 * 28
                    if MIN_PIXELS is not None and MAX_PIXELS is not None:
                        self.processor = AutoProcessor.from_pretrained(
                            desired_model_name, 
                            torch_dtype=torch.bfloat16,
                            trust_remote_code=True, 
                            min_pixels=MIN_PIXELS, 
                            max_pixels=MAX_PIXELS, 
                        )
                    else:
                        self.processor = AutoProcessor.from_pretrained(
                            desired_model_name, 
                            torch_dtype=torch.bfloat16,
                            trust_remote_code=True, 
                        )
                else:
                    self.processor = AutoProcessor.from_pretrained(
                        desired_model_name, 
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True, 
                    )
                self.processor.tokenizer.padding_side  = "left"
                
                print(f"Loading new model: {desired_model_name}")
                self.model = MODEL_MAP[desired_model_name].from_pretrained(
                    desired_model_name, 
                    torch_dtype=torch.bfloat16, 
                    attn_implementation="flash_attention_2", 
                    device_map="cuda:0", 
                ).eval()

                self.current_model_name = desired_model_name
                print(f"Successfully loaded {self.current_model_name}.")
            except Exception as e:
                QMessageBox.critical(self, "Model Loading Error", f"Unable to load model: {e}")
                self.set_ui_enabled(True)
                self.model = None
                self.processor = None
                self.current_model_name = None
                return
            finally:
                self.set_ui_enabled(True)

        params = {
            'video_path': inference_video_path,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'output_fps': self.fps_spinbox.value(),
            'frame_accumulation': frame_accumulation,
            'sliding_window_size': sliding_window_size,
            'system_prompt': self.system_prompt_edit.toPlainText(),
            'user_prompt': self.user_prompt_edit.toPlainText(),
            'model_name': self.model_combo.currentText(),
        }

        self.set_ui_enabled(False, "Inference in progress...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.inference_worker = InferenceWorker(params, self.model, self.processor)
        self.inference_worker.progress_signal.connect(self.update_progress)
        self.inference_worker.finished_signal.connect(self.inference_finished)
        self.inference_worker.error_signal.connect(self.inference_error)
        self.inference_worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def inference_finished(self, output_video_path, generated_texts):
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

    def inference_error(self, error_message):
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
            fps = self.get_current_cap().get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = self.original_fps # Fallback
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
        if self.is_playing is False:
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

        if self.current_video_source == 'processed' and self.inference_data and frame_number < len(self.inference_data):
            original_video_frame_number = self.inference_data[frame_number]["current_frame"]
            generated_text = self.inference_data[frame_number]["text"]
            self.inference_text_display.setText(f"This frame {frame_number} corresponds to frame {original_video_frame_number} of the original video.\nGenerated Text:\n{generated_text}")
        elif self.current_video_source != 'processed':
            self.inference_text_display.clear()

    def set_position(self, position):
        if self.is_playing:
            self.toggle_play_pause()

        self.display_frame(position)

    def update_time_label(self, frame_number):
        cap = self.get_current_cap()
        if cap and cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = self.original_fps # Fallback for processed video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_seconds = int(frame_number / fps)
            total_seconds = int(total_frames / fps)
            current_time = time.strftime('%M:%S', time.gmtime(current_seconds))
            total_time = time.strftime('%M:%S', time.gmtime(total_seconds))
            self.video_time_label.setText(f"{current_time} / {total_time}")

    def get_current_cap(self):
        if self.current_video_source == 'processed' and self.processed_cap and self.processed_cap.isOpened():
            return self.processed_cap
        elif self.current_video_source == 'edited' and self.edited_cap and self.edited_cap.isOpened():
            return self.edited_cap
        return self.cap
    
    def switch_video_source(self, source_text):
        source_map = {"Original Video": 'original', "Edited Video": 'edited', "Inferred Video": 'processed'}
        self.current_video_source = source_map.get(source_text, 'original')

        is_original_edited_video = (self.current_video_source == 'original' or self.current_video_source == 'edited')
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
            if self.current_video_source != 'original':
                QMessageBox.warning(self, "Warning", f"{source_text} is not available. Switching back to Original Video.")
                self.video_source_combo.setCurrentIndex(0)
            else:
                 self.video_display_label.setText("Video not available.")

    def closeEvent(self, event):
        if self.inference_worker and self.inference_worker.isRunning():
            self.inference_worker.stop()
            self.inference_worker.wait()
        if self.video_worker and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker.wait()
        if self.model is not None:
            print(f"Releasing model {self.current_model_name} on exit.")
            del self.model
            del self.processor
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
        self.reset_all_caps()
        try:
            shutil.rmtree(self.temp_dir)
            print(f"Removed temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error removing temporary directory: {e}")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())