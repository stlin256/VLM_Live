import os
import cv2
import torch
import time
import threading
from flask import Flask, render_template, Response, jsonify, request
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import numpy as np

# --- Global Configuration Store ---
# These will be the default values, but they can be changed dynamically via the API
config = {
    "USE_WEBCAM": False,
    "INPUT_SOURCE": "pic.jpg",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "MODEL_NAME": "./SmolVLM-256M-Instruct",
    "PROMPT": "A one-sentence description.",
    "MAX_NEW_TOKENS": 32
}
config_lock = threading.Lock()

app = Flask(__name__)

# --- VLM Initialization ---
print(f"Loading model from {config['MODEL_NAME']}...")
processor = AutoProcessor.from_pretrained(config['MODEL_NAME'])
model = AutoModelForVision2Seq.from_pretrained(
    config['MODEL_NAME'],
    torch_dtype=torch.bfloat16,
).to(config['DEVICE']).eval()
print("Model loaded successfully.")
# --- End VLM Initialization ---

# --- Thread-safe data storage ---
last_frame = None
last_description = "Initializing..."
last_latency = 0.0
frame_lock = threading.Lock()
vlm_lock = threading.Lock()

# --- Thread Management ---
capture_thread = None
stop_capture_event = threading.Event()

# --- Thread Management ---
capture_thread = None
stop_capture_event = threading.Event()

def create_error_image(message, width=640, height=480):
    """Creates a black image with an error message."""
    img_np = np.full((height, width, 3), 0, np.uint8)
    cv2.putText(img_np, message, (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img_np

def capture_frames(stop_event):
    """Function to run in a separate thread for capturing frames."""
    global last_frame
    
    with config_lock:
        use_webcam = config["USE_WEBCAM"]
        input_source = config["INPUT_SOURCE"]

    is_stream = not use_webcam and (input_source.lower().startswith('rtsp://') or input_source.lower().startswith('http://'))
    is_local_video = not use_webcam and input_source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    is_video = is_stream or is_local_video
    frame_delay = 0.1

    video_capture = None
    static_image = None

    if use_webcam:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Error: Could not open webcam.")
            with frame_lock:
                last_frame = create_error_image("Error: Could not open webcam.")
            return
    elif is_video:
        if not is_stream and not os.path.exists(input_source):
            print(f"Error: Video file not found at {input_source}")
            with frame_lock:
                last_frame = create_error_image(f"Video not found: {input_source}")
            return
        video_capture = cv2.VideoCapture(input_source)
        # CRITICAL CHECK: Verify if the video source was opened successfully
        if not video_capture.isOpened():
            print(f"CRITICAL: OpenCV could not open the video source: {input_source}")
            print("This might be due to missing codecs (FFmpeg) or network issues.")
            with frame_lock:
                last_frame = create_error_image(f"Error opening source")
            return
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            frame_delay = 1 / fps
    else: # Image
        if not os.path.exists(input_source):
            print(f"Error: Image file not found at {input_source}")
            with frame_lock:
                last_frame = create_error_image(f"Image not found: {input_source}")
            return
        static_image = cv2.imread(input_source)
        if static_image is None:
            print(f"Error: Could not read image file at {input_source}")
            with frame_lock:
                last_frame = create_error_image(f"Error reading image: {input_source}")
            return

    print("Capture thread started.")
    while not stop_event.is_set():
        frame = None
        if use_webcam or is_video:
            success, frame = video_capture.read()
            if not success:
                if is_video:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.1)
                    continue
        else:
            frame = static_image.copy()
        
        with frame_lock:
            last_frame = frame
        
        time.sleep(frame_delay)
    
    if video_capture:
        video_capture.release()
    print("Capture thread stopped.")


def vlm_inference():
    """Function to run in a separate thread for VLM inference."""
    global last_description, last_fps
    
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

        pil_image = Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
        
        start_time = time.time()
        
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        prompt_template = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt_template, images=[pil_image], return_tensors="pt").to(config["DEVICE"])

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.6)
        
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        raw_response = generated_texts[0]
        assistant_marker = "Assistant:"
        if assistant_marker in raw_response:
            response = raw_response.split(assistant_marker, 1)[-1].strip()
        else:
            response = raw_response.split("ASSISTANT:")[-1].split("assistant:")[-1].strip()
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000 # Convert to milliseconds
        
        with vlm_lock:
            global last_description, last_latency
            last_description = response
            last_latency = processing_time

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', engine='torch')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    while True:
        frame_to_show = None
        with frame_lock:
            if last_frame is None:
                frame_to_show = create_error_image("Waiting for video stream...")
            else:
                frame_to_show = last_frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame_to_show)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
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

    # Stop the old capture thread
    if capture_thread and capture_thread.is_alive():
        stop_capture_event.set()
        capture_thread.join() # Wait for the thread to finish

    # Start a new capture thread with the updated settings
    stop_capture_event.clear()
    capture_thread = threading.Thread(target=capture_frames, args=(stop_capture_event,), daemon=True)
    capture_thread.start()
    
    return jsonify({"status": "success", "message": "Settings updated."})

if __name__ == '__main__':
    from waitress import serve
    
    # Start the initial capture thread
    capture_thread = threading.Thread(target=capture_frames, args=(stop_capture_event,), daemon=True)
    capture_thread.start()

    # Start the VLM inference thread (it will run continuously)
    vlm_thread = threading.Thread(target=vlm_inference, daemon=True)
    vlm_thread.start()
    
    host = '0.0.0.0'
    port = 5000
    print("Starting server with waitress...")
    print(f"--> Access the application at http://127.0.0.1:{port}")
    serve(app, host=host, port=port)
