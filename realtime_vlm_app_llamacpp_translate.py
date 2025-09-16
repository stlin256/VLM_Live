import os
import cv2
import time
import threading
import base64
from io import BytesIO
import contextlib
import sys
from flask import Flask, render_template, Response, jsonify, request
from PIL import Image
import numpy as np
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

# --- Global Configuration Store ---
config = {
    "USE_WEBCAM": False,
    "INPUT_SOURCE": "pic.jpg",
    "MODEL_NAME": "./SmolVLM-256M-Instruct-GGUF/SmolVLM2-256M-Video-Instruct-f16.gguf",
    "MMPROJ_MODEL_PATH": "./SmolVLM-256M-Instruct-GGUF/mmproj-SmolVLM2-256M-Video-Instruct-f16.gguf",
    "TRANSLATOR_MODEL_NAME": "./Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
    "PROMPT": "A one-sentence description of the image.",
    "TRANSLATE_PROMPT": "/no_think翻译为中文：",
    "MAX_NEW_TOKENS": 32,
    "N_GPU_LAYERS": 20,
    "TRANSLATOR_N_GPU_LAYERS": 10,
    "TRANSLATE_ENABLED": False
}
config_lock = threading.Lock()

app = Flask(__name__)

# --- Llama.cpp VLM Initialization ---
print(f"Loading VLM GGUF model from {config['MODEL_NAME']}...")
chat_handler = Llava15ChatHandler(clip_model_path=config["MMPROJ_MODEL_PATH"], verbose=False)
llm_vlm = Llama(
    model_path=config["MODEL_NAME"],
    chat_handler=chat_handler,
    n_ctx=2048,
    n_gpu_layers=config["N_GPU_LAYERS"],
    verbose=False
)
print("VLM GGUF model loaded successfully.")

# --- Llama.cpp Translator Initialization ---
print(f"Loading Translator GGUF model from {config['TRANSLATOR_MODEL_NAME']}...")
llm_translator = Llama(
    model_path=config["TRANSLATOR_MODEL_NAME"],
    n_ctx=512,
    n_gpu_layers=config["TRANSLATOR_N_GPU_LAYERS"],
    verbose=False
)
print("Translator GGUF model loaded successfully.")

# --- Thread-safe data storage ---
last_frame = None
last_description = "Initializing..."
last_latency = 0.0
frame_lock = threading.Lock()
vlm_lock = threading.Lock()

# --- Thread Management ---
capture_thread = None
stop_capture_event = threading.Event()

# ... [create_error_image, pil_to_base64, suppress_stdout_stderr functions remain the same] ...
def create_error_image(message, width=640, height=480):
    img_np = np.full((height, width, 3), 0, np.uint8)
    cv2.putText(img_np, message, (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img_np

def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        save_stdout, save_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = save_stdout, save_stderr

def capture_frames(stop_event):
    """Function to run in a separate thread for capturing frames."""
    global last_frame
    # ... [capture_frames logic remains the same] ...
    with config_lock:
        use_webcam = config["USE_WEBCAM"]
        input_source = config["INPUT_SOURCE"]
    is_stream = not use_webcam and (input_source.lower().startswith(('rtsp://', 'http://')))
    is_local_video = not use_webcam and input_source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    is_video = is_stream or is_local_video
    frame_delay = 0.1
    video_capture = None
    static_image = None
    if use_webcam: video_capture = cv2.VideoCapture(0)
    elif is_video: video_capture = cv2.VideoCapture(input_source)
    else:
        if os.path.exists(input_source): static_image = cv2.imread(input_source)
        else:
            with frame_lock: last_frame = create_error_image("Image not found")
            return
    if video_capture and not video_capture.isOpened():
        with frame_lock: last_frame = create_error_image("Error opening source")
        return
    if is_local_video:
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        if fps > 0: frame_delay = 1 / fps
    print("Capture thread started.")
    while not stop_event.is_set():
        frame = None
        if use_webcam or is_video:
            success, frame = video_capture.read()
            if not success:
                if is_local_video: video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else: time.sleep(0.1)
                continue
        else: frame = static_image.copy()
        with frame_lock: last_frame = frame
        # For streams, we should not sleep, to clear the buffer and reduce latency
        if not is_stream:
            time.sleep(frame_delay)
    if video_capture: video_capture.release()
    print("Capture thread stopped.")

def vlm_inference():
    """Runs VLM inference and then translates if enabled (serial execution)."""
    global last_description, last_latency
    
    print("VLM inference thread started.")
    while True:
        frame_copy = None
        with frame_lock:
            if last_frame is not None:
                frame_copy = last_frame.copy()

        if frame_copy is None:
            time.sleep(0.1)
            continue

        with config_lock:
            prompt = config["PROMPT"]
            max_tokens = config["MAX_NEW_TOKENS"]
            translate_enabled = config["TRANSLATE_ENABLED"]
            translate_prompt = config["TRANSLATE_PROMPT"]

        pil_image = Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
        
        start_time = time.time()
        
        data_url = f"data:image/jpeg;base64,{pil_to_base64(pil_image)}"
        
        with suppress_stdout_stderr():
            response = llm_vlm.create_chat_completion(
                messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}, {"type": "text", "text": prompt}]}],
                max_tokens=max_tokens
            )
        
        description = response['choices'][0]['message']['content']

        if translate_enabled:
            full_translation_prompt = f"{translate_prompt}\n{description}"
            with suppress_stdout_stderr():
                trans_response = llm_translator.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a helpful translation assistant."},
                        {"role": "user", "content": full_translation_prompt}
                    ],
                    max_tokens=max_tokens * 2 # Allow more tokens for translation
                )
            translated_text = trans_response['choices'][0]['message']['content'].strip()
            # Post-process to remove <think> tags
            final_text = translated_text.replace("<think>", "").replace("</think>", "").strip()
        else:
            final_text = description

        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        with vlm_lock:
            last_description = final_text
            last_latency = processing_time

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', engine='llamacpp')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    # ... [gen_frames logic remains the same] ...
    while True:
        frame_to_show = None
        with frame_lock:
            if last_frame is None: frame_to_show = create_error_image("Waiting for stream...")
            else: frame_to_show = last_frame.copy()
        ret, buffer = cv2.imencode('.jpg', frame_to_show)
        if not ret: continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/vlm_data')
def vlm_data():
    with vlm_lock:
        return jsonify({'description': last_description, 'latency': f"{last_latency:.2f}"})

@app.route('/get_settings')
def get_settings():
    with config_lock:
        return jsonify(config)

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global capture_thread, stop_capture_event
    data = request.get_json()
    with config_lock:
        config["USE_WEBCAM"] = data.get('use_webcam', config["USE_WEBCAM"])
        config["INPUT_SOURCE"] = data.get('input_source', config["INPUT_SOURCE"])
        config["MAX_NEW_TOKENS"] = data.get('max_new_tokens', config["MAX_NEW_TOKENS"])
        config["PROMPT"] = data.get('prompt', config["PROMPT"])
        config["TRANSLATE_ENABLED"] = data.get('translate_enabled', config["TRANSLATE_ENABLED"])

    if capture_thread and capture_thread.is_alive():
        stop_capture_event.set()
        capture_thread.join()

    stop_capture_event.clear()
    capture_thread = threading.Thread(target=capture_frames, args=(stop_capture_event,), daemon=True)
    capture_thread.start()
    
    return jsonify({"status": "success", "message": "Settings updated."})

if __name__ == '__main__':
    from waitress import serve
    
    capture_thread = threading.Thread(target=capture_frames, args=(stop_capture_event,), daemon=True)
    capture_thread.start()

    vlm_thread = threading.Thread(target=vlm_inference, daemon=True)
    vlm_thread.start()
    
    host = '0.0.0.0'
    port = 5000
    print("Starting server with waitress...")
    print(f"--> Access the application at http://127.0.0.1:{port}")
    serve(app, host=host, port=port)