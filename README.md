# GUI for Video-Based VLMs Inference and Preprocessing 🚀

This is a powerful and intuitive graphical user interface (GUI) application built with PyQt for comprehensive video processing. The tool combines standard video editing functionalities with cutting-edge AI-powered features, including object tracking with **Grounded-SAM 2** and video description generation using advanced vision-language models.

## ✨ Features

* **Video Loading & Playback**: Load and play various video formats with an interactive timeline slider and playback controls.
* **Comprehensive Video Editing Suite**:
    * **Clip**: Trim videos to specific start and end frames and adjust the FPS.
    * **Crop**: Cut out a specific region of interest from the video.
    * **Resize**: Change the resolution of the video.
    * **Draw**: Overlay shapes like rectangles and lines with custom colors and thickness.
* **🤖 AI-Powered Object Tracking (Grounded-SAM 2)**:
    * Detect, segment, and track objects throughout the video based on simple text prompts (e.g., "a person.", "the red car.").
    * Customize the output to show bounding boxes, masks, or both, with optional labels.
* **🧠 Customizable VLM Inference Pipeline**:
    * Generate detailed descriptions from videos using state-of-the-art Vision-Language Models.
    * Take full control of the inference process with granular settings:
        * **Define Video Range:** Specify the exact start and end frames of the video segment you are interest.
        * **Frame Batching (Accumulation):** Set the number of consecutive frames to be fed into the model for each single inference.
        * **Sliding Window Control:** Adjust the step of the sliding window to determine the frequency of inference (e.g., perform one inference every N frames).
* **Interactive Workflow**: Seamlessly switch between viewing the **original**, **edited**, and **VLMs-inferred** video outputs within the same interface.

---

## 🔧 Prerequisites

Before you begin, ensure you have the following installed and configured on your system.

* **GPU**: An NVIDIA GPU with at least **24 GB of VRAM** is highly recommended for smooth operation, especially for the Grounded-SAM 2 and video understanding features.
* **NVIDIA CUDA Toolkit**: You must have **NVIDIA CUDA Toolkit 12.4** or newer installed.
* **GUI Environment**: A desktop environment is required to run the application. Ensure the `DISPLAY` environment variable is set.

---

## 🛠️ Setup and Installation

### 1. Configure Environment Variables

You must set up the environment variables for CUDA to work correctly. Add the following lines to your shell configuration file (e.g., `~/.bashrc`, `~/.zshrc`), and remember to adjust the `CUDA_HOME` path to match your CUDA installation directory.

```bash
export CUDA_HOME=/usr/local/cuda-12.4 # Or your specific version like /usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

After editing the file, reload your shell configuration by running `source ~/.bashrc` or opening a new terminal.

On Linux, also ensure your `DISPLAY` variable is set, for example:
```bash
export DISPLAY=:0
```

### 2. Clone the Repository

```bash
git clone --recurse-submodules <repository-url>
cd <repository-directory>
```

### 3. Install Dependencies

You can install the required packages using either `pip` or `uv`.

**Option A: Using `pip`**

```bash
# Install dependencies from requirements.txt
pip install -r requirements.txt

# Install the local Grounded-SAM-2 package
pip install ./Grounded-SAM-2
```

**Option B: Using `uv`**

```bash
# Synchronize the virtual environment with pyproject.toml
uv sync

# Install the local Grounded-SAM-2 package
uv pip install ./Grounded-SAM-2
```

---

## ▶️ How to Run

Once the installation is complete, you can start the application by running the Python script:

**Option A: Noraml Run**
```bash
python video_inference_PyQt.py
```

**Option B: Using `uv`**
```bash
uv run video_inference_PyQt.py
```

### Workflow Guide
1.  **Load Video**: Click on **"Choose A Video"** to load your video file. The video player on the right will become active.
2.  **Edit Video (Optional)**:
    * Navigate through the tabs in the **"2. Video Editing Tools"** section (Clip, Crop, Resize, Draw, Grounded-SAM2).
    * Enter your desired parameters and click the corresponding **"Apply"** button for each edit.
    * After applying edits, you can save the result using the **"Save Edited Video"** button or view it by selecting "Edited Video" in the player's dropdown menu.
    * **Note for Grounded-SAM2**: The text prompt must be **lowercased and end with a period** (e.g., `a white car.`).
3.  **Set Inference Parameters**: In the **"3. Inference Parameters Setting"** section, choose your source (Original or Edited video), select the frame range for using, and configure other settings like FPS and the vision-language model.
4.  **Define Prompts**: In the **"4. Model Prompt"** section, provide the system and user prompts to guide the video description model.
5.  **Execute and View Results**: Click **"Start to Infer"** to begin the AI response. The progress will be shown, and once completed, you can watch the final video by selecting "Inferred Video" from the player's dropdown. The corresponding generated text for each frame will appear in the text box below.