import sys
import os
import cv2
import time
import torch
import numpy as np
import imageio
import gc
import textwrap

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QSlider, QSpinBox, QComboBox, QTextEdit, QGroupBox,
    QFormLayout, QMessageBox, QProgressBar, QStyle, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont
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
#VideoDisplayLabel {
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
            out = imageio.get_writer(output_video_path, fps=output_fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame_index = start_frame
            queue = deque()
            generated_texts = []

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 Video Inference Tool")
        self.setGeometry(100, 100, 1600, 900)

        self.video_path = None
        self.cap = None
        self.processed_cap = None
        self.total_frames = 0
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.update_frame)
        self.is_playing = False
        self.current_video_source = 'original' # 'original' or 'processed'
        self.worker_thread = None
        self.model = None
        self.processor = None
        self.current_model_name = None
        self.inference_data = []

        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

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

        params_group = QGroupBox("2. Inference Parameters Setting")
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

        prompt_group = QGroupBox("3. Model Prompt")
        prompt_layout = QVBoxLayout()
        self.system_prompt_edit = QTextEdit()
        self.system_prompt_edit.setPlaceholderText("Please input system prompt...")
        self.system_prompt_edit.setText("You are a helpful assistant.")
        self.user_prompt_edit = QTextEdit()
        self.user_prompt_edit.setPlaceholderText("Please input user prompt...")
        self.user_prompt_edit.setText("Describe the video.")
        prompt_layout.addWidget(QLabel("System Prompt:"))
        prompt_layout.addWidget(self.system_prompt_edit)
        prompt_layout.addWidget(QLabel("User Prompt:"))
        prompt_layout.addWidget(self.user_prompt_edit)
        prompt_group.setLayout(prompt_layout)
        left_layout.addWidget(prompt_group)

        action_group = QGroupBox("4. Execute and Results")
        action_layout = QVBoxLayout()
        self.start_inference_btn = QPushButton("Start to Infer")
        self.start_inference_btn.setEnabled(False)
        self.start_inference_btn.clicked.connect(self.start_inference)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.inference_text_label = QLabel("The Inference Result of Current Frame:")
        self.inference_text_display = QTextEdit()
        self.inference_text_display.setReadOnly(True)
        self.inference_text_display.setPlaceholderText("When playing the inferred video, the corresponding generated text will be displayed here...")
        action_layout.addWidget(self.start_inference_btn)
        action_layout.addWidget(self.progress_bar)
        action_layout.addWidget(self.inference_text_label)
        action_layout.addWidget(self.inference_text_display)
        action_group.setLayout(action_layout)
        left_layout.addWidget(action_group)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.video_display_label = QLabel("Please load a video first.")
        self.video_display_label.setObjectName("VideoDisplayLabel")
        self.video_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        # self.video_display_label.setStyleSheet("background-color: black; color: white;")
        
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
        self.video_source_combo.addItems(["Original Video", "Inferred Video"])
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

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose A Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.video_path = path
            self.video_path_label.setText(os.path.basename(path))

            if self.cap: self.cap.release()
            if self.processed_cap: self.processed_cap.release()
            self.processed_cap = None

            self.inference_data = []
            self.inference_text_display.clear()
            
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "ERROR", "The video file can't be opened.")
                return

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.start_frame_slider.setRange(0, self.total_frames - 1)
            self.start_frame_slider.setValue(0)
            self.end_frame_slider.setRange(0, self.total_frames - 1)
            self.end_frame_slider.setValue(self.total_frames - 1)
            self.playback_slider.setRange(0, self.total_frames - 1)
            self.frame_spinbox.setValue(0)
            self.frame_spinbox.setRange(0, self.total_frames - 1)
            
            self.start_frame_slider.setEnabled(True)
            self.end_frame_slider.setEnabled(True)
            self.playback_slider.setEnabled(True)
            self.frame_spinbox.setEnabled(True)
            self.start_inference_btn.setEnabled(True)
            self.play_pause_btn.setEnabled(True)
            self.video_source_combo.setCurrentIndex(0)
            self.video_source_combo.setEnabled(False)

            self.current_video_source = 'original'
            self.switch_video_source("Original Video")
    
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
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "Please load a video first.")
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
            self.start_inference_btn.setEnabled(False)
            self.start_inference_btn.setText("Switching The Model...")
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
                self.start_inference_btn.setText("Start to Infer")
                self.start_inference_btn.setEnabled(True)
                self.model = None
                self.processor = None
                self.current_model_name = None
                return

        params = {
            'video_path': self.video_path,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'output_fps': self.fps_spinbox.value(),
            'frame_accumulation': frame_accumulation,
            'sliding_window_size': sliding_window_size,
            'system_prompt': self.system_prompt_edit.toPlainText(),
            'user_prompt': self.user_prompt_edit.toPlainText(),
            'model_name': self.model_combo.currentText(),
        }

        self.start_inference_btn.setText("Inference In Progress...")
        self.start_inference_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.worker_thread = InferenceWorker(params, self.model, self.processor)
        self.worker_thread.progress_signal.connect(self.update_progress)
        self.worker_thread.finished_signal.connect(self.inference_finished)
        self.worker_thread.error_signal.connect(self.inference_error)
        self.worker_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def inference_finished(self, output_video_path, generated_texts):
        self.progress_bar.setVisible(False)
        self.start_inference_btn.setEnabled(True)
        self.start_inference_btn.setText("Start to Infer")
        QMessageBox.information(self, "Finish", f"The video inference is completed!\nThe result is saved in:\n{output_video_path}")

        self.inference_data = generated_texts

        if self.processed_cap:
            self.processed_cap.release()
            self.processed_cap = None
        self.processed_cap = cv2.VideoCapture(output_video_path)
        if self.processed_cap.isOpened():
            self.video_source_combo.setEnabled(True)
            self.video_source_combo.setCurrentIndex(1)

        self.switch_video_source("Inferred Video")

    def inference_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.start_inference_btn.setEnabled(True)
        self.start_inference_btn.setText("Start to Infer")
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
        self.playback_slider.setValue(frame_number)
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(frame_number)
        self.frame_spinbox.blockSignals(False)

        if self.current_video_source == 'processed' and self.inference_data:
            original_video_frame_number = self.inference_data[frame_number]["current_frame"]
            generated_text = self.inference_data[frame_number]["text"]
            self.inference_text_display.setText(f"This frame {frame_number} corresponds to frame {original_video_frame_number} of the original video.\nGenerated Text:\n{generated_text}")

    def set_position(self, position):
        if self.is_playing:
            self.toggle_play_pause()

        self.display_frame(position)

    def update_time_label(self, frame_number):
        cap = self.get_current_cap()
        if cap:
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
        return self.cap
    
    def switch_video_source(self, source_text):
        self.current_video_source = 'processed' if source_text == 'Inferred Video' else 'original'
        is_original_video = (self.current_video_source == 'original')
        self.set_start_btn.setEnabled(is_original_video)
        self.set_end_btn.setEnabled(is_original_video)
        self.inference_text_display.clear()
        
        cap = self.get_current_cap()
        if cap:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.playback_slider.setRange(0, total_frames - 1)
            self.frame_spinbox.setRange(0, total_frames - 1)
            self.is_playing = False
            self.playback_timer.stop()
            self.play_pause_btn.setText(" PLAY")
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.display_frame(0)

    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait()
        if self.model is not None:
            print(f"Releasing model {self.current_model_name} on exit.")
            del self.model
            del self.processor
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
        if self.cap: self.cap.release()
        if self.processed_cap: self.processed_cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())