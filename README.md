# Real-Time VLM Visual Analysis Web App

### —— A web application for real-time visual analysis using Vision Language Models

<a href="#cn">中文介绍</a>

---

<!-- Navigation -->
<ul>
  <li><a href="#introduction">Introduction</a>
    <ul>
      <li><a href="#features">Features</a></li>
    </ul>
  </li>
  <li><a href="#repository-structure">Repository Structure</a></li>
  <li><a href="#usage-guide">Usage Guide</a>
    <ul>
      <li><a href="#environment-setup">Environment Setup</a></li>
      <li><a href="#installation">Installation</a></li>
      <li><a href="#configuration">Configuration</a></li>
      <li><a href="#running-the-application">Running the Application</a></li>
    </ul>
  </li>
  <li><a href="#demo">Demo</a></li>
</ul>

---

## Introduction
<a id="introduction"></a>

This project provides a complete web application that leverages a Vision Language Model (VLM) to perform real-time analysis of visual content. The application can capture video from a webcam, a video file, or use a static image as input. It continuously processes the visual feed, generates textual descriptions using the VLM, and displays the video stream, the model's output, and the processing FPS on a responsive web interface.

### Features
<a id="features"></a>
- **Multi-Source Input**: Supports real-time video capture from a webcam, looping playback from a video file (e.g., `.mp4`), or analysis of a static image.
- **Real-Time VLM Inference**: Continuously sends frames to the VLM for analysis and description generation.
- **Web Interface**: A clean, responsive web UI that displays the visual source, real-time FPS, and the VLM's textual output side-by-side.
- **Asynchronous Processing**: Utilizes multithreading to handle frame capture and VLM inference in parallel, maximizing performance.
- **Configurable**: Easily configurable through variables at the top of the main script (`realtime_vlm_app.py`).

---

## Repository Structure
<a id="repository-structure"></a>

```bash
.
├── realtime_vlm_app.py     # Main Flask application script
├── templates/
│   └── index.html          # HTML template for the web interface
├── requirements.txt        # Python dependencies
├── pic.jpg                 # Example image file
└── README.md               # This file
```

---

## Usage Guide
<a id="usage-guide"></a>

### Environment Setup
<a id="environment-setup"></a>
It is recommended to use a virtual environment (e.g., conda or venv).

1.  **Create a virtual environment** (example with conda):
    ```bash
    conda create -n vlm_webapp python=3.10
    ```
2.  **Activate the environment**:
    ```bash
    conda activate vlm_webapp
    ```

### Installation
<a id="installation"></a>
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/stlin256/VLM_Live.git
    cd VLM_Live
    ```
2.  **Install PyTorch**: Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) and install a version compatible with your CUDA setup.
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the VLM model**: Ensure you have the `SmolVLM-256M-Instruct` model files in a directory named `SmolVLM-256M-Instruct` at the project root.

### Configuration
<a id="configuration"></a>
Open `realtime_vlm_app.py` and modify the variables in the "Configuration" section as needed:

```python
# --- Configuration ---
USE_WEBCAM = False  # Set to True for webcam, False for file input
INPUT_SOURCE = "a.mp4" # Path to your video or image file
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "./SmolVLM-256M-Instruct"
PROMPT = "what you see? Answer in a word."
# --- End Configuration ---
```

### Running the Application
<a id="running-the-application"></a>
1.  **Start the server**:
    ```bash
    python realtime_vlm_app.py
    ```
2.  **Open your browser**: Navigate to `http://127.0.0.1:5000` to view the application.

---

## Demo
<a id="demo"></a>
The web interface displays the video feed on the left and the analysis results (FPS and VLM output) on the right. The layout is responsive and maintains a 2/3 to 1/3 ratio between the video and text sections.

![webpage](./webpage.png)

---
---
<div id="cn"></div>

# 实时 VLM 视觉分析 Web 应用

### —— 一个使用视觉语言模型进行实时视觉分析的Web应用

---

<!-- 目录导航 -->
<ul>
  <li><a href="#中文介绍">介绍</a>
    <ul>
      <li><a href="#功能特性">功能特性</a></li>
    </ul>
  </li>
  <li><a href="#仓库结构">仓库结构</a></li>
  <li><a href="#使用指南">使用指南</a>
    <ul>
      <li><a href="#环境设置">环境设置</a></li>
      <li><a href="#安装">安装</a></li>
      <li><a href="#配置">配置</a></li>
      <li><a href="#运行应用">运行应用</a></li>
    </ul>
  </li>
  <li><a href="#演示">演示</a></li>
</ul>

---

## 介绍
<a id="中文介绍"></a>

本项目提供了一个完整的Web应用程序，它利用视觉语言模型（VLM）对视觉内容进行实时分析。该应用可以从网络摄像头、视频文件捕获视频，或使用静态图像作为输入。它会持续处理视觉输入，使用VLM生成文本描述，并在一个响应式的Web界面上并排显示视频流、模型的输出以及处理的FPS。

### 功能特性
<a id="功能特性"></a>
- **多源输入**: 支持从网络摄像头进行实时视频捕获、循环播放视频文件（如 `.mp4`），或分析静态图像。
- **实时VLM推理**: 持续将视频帧发送给VLM进行分析并生成描述。
- **Web界面**: 一个简洁、响应式的Web UI，可并排显示视觉源、实时FPS和VLM的文本输出。
- **异步处理**: 利用多线程并行处理帧捕获和VLM推理，以最大化性能。
- **可配置**: 可通过主脚本 (`realtime_vlm_app.py`) 顶部的变量轻松进行配置。

---

## 仓库结构
<a id="仓库结构"></a>

```bash
.
├── realtime_vlm_app.py     # 主Flask应用脚本
├── templates/
│   └── index.html          # Web界面的HTML模板
├── requirements.txt        # Python依赖项
├── pic.jpg                 # 示例图片
└── README.md               # 本文件
```

---

## 使用指南
<a id="使用指南"></a>

### 环境设置
<a id="环境设置"></a>
建议使用虚拟环境（例如 conda 或 venv）。

1.  **创建虚拟环境** (使用 conda 的示例):
    ```bash
    conda create -n vlm_webapp python=3.10
    ```
2.  **激活环境**:
    ```bash
    conda activate vlm_webapp
    ```

### 安装
<a id="安装"></a>
1.  **克隆仓库**:
    ```bash
    git clone https://github.com/stlin256/VLM_Live.git
    cd VLM_Live
    ```
2.  **安装 PyTorch**: 访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 并安装与您的 CUDA 环境兼容的版本。
3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **下载VLM模型**: 确保您已将 `SmolVLM-256M-Instruct` 模型文件放置在项目根目录下名为 `SmolVLM-256M-Instruct` 的文件夹中。

### 配置
<a id="配置"></a>
打开 `realtime_vlm_app.py` 文件，并根据需要修改“Configuration”部分中的变量：

```python
# --- Configuration ---
USE_WEBCAM = False  # 设置为 True 使用摄像头, False 使用文件输入
INPUT_SOURCE = "a.mp4" # 你的视频或图片文件路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "./SmolVLM-256M-Instruct"
PROMPT = "what you see? Answer in a word."
# --- End Configuration ---
```

### 运行应用
<a id="运行应用"></a>
1.  **启动服务器**:
    ```bash
    python realtime_vlm_app.py
    ```
2.  **打开浏览器**: 访问 `http://127.0.0.1:5000` 查看应用。

---

## 演示
<a id="演示"></a>
Web界面在左侧显示视频，右侧显示分析结果（FPS和VLM输出）。该布局是响应式的，并保持视频和文本部分之间 2/3 到 1/3 的宽度比例。

![webpage](./webpage.png)