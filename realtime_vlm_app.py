import os
import cv2
import torch
import time
import threading
from flask import Flask, render_template, Response, jsonify
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import numpy as np

# --- Configuration ---
USE_WEBCAM = False  # Set to False to use a video file or placeholder image
INPUT_SOURCE = "pic.jpg" # Can be a path to a video file or an image file
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "./SmolVLM-256M-Instruct"
PROMPT = "what you see? "
# --- End Configuration ---

app = Flask(__name__)

# --- VLM Initialization ---
print(f"Loading model from {MODEL_NAME}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
).to(DEVICE).eval() # Set to evaluation mode
print("Model loaded successfully.")
# --- End VLM Initialization ---

# --- Thread-safe data storage ---
last_frame = None
last_description = "Initializing..."
last_fps = 0.0
frame_lock = threading.Lock()
vlm_lock = threading.Lock()
# ---

def create_error_image(message, width=640, height=480):
    """Creates a black image with an error message."""
    img_np = np.full((height, width, 3), 0, np.uint8)
    cv2.putText(img_np, message, (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img_np

def capture_frames():
    """Function to run in a separate thread for capturing frames."""
    global last_frame
    
    is_video = INPUT_SOURCE.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    frame_delay = 0.1 # Default delay for images and webcam

    if USE_WEBCAM:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Error: Could not open webcam.")
            with frame_lock:
                last_frame = create_error_image("Error: Could not open webcam.")
            return
    elif is_video:
        if not os.path.exists(INPUT_SOURCE):
            print(f"Error: Video file not found at {INPUT_SOURCE}")
            with frame_lock:
                last_frame = create_error_image(f"Video not found: {INPUT_SOURCE}")
            return
        video_capture = cv2.VideoCapture(INPUT_SOURCE)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            frame_delay = 1 / fps
        else:
            print("Warning: Could not get video FPS. Using default delay.")
    else: # It's an image
        if not os.path.exists(INPUT_SOURCE):
            print(f"Error: Image file not found at {INPUT_SOURCE}")
            with frame_lock:
                last_frame = create_error_image(f"Image not found: {INPUT_SOURCE}")
            return
        static_image = cv2.imread(INPUT_SOURCE)
        if static_image is None:
            print(f"Error: Could not read image file at {INPUT_SOURCE}")
            with frame_lock:
                last_frame = create_error_image(f"Error reading image: {INPUT_SOURCE}")
            return

    print("Capture thread started.")
    while True:
        frame = None
        if USE_WEBCAM or is_video:
            success, frame = video_capture.read()
            if not success:
                if is_video: # Loop the video
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else: # Webcam error
                    print("Error: Failed to capture frame from webcam.")
                    time.sleep(0.1)
                    continue
        else: # Static image
            frame = static_image.copy()
        
        with frame_lock:
            last_frame = frame
        
        time.sleep(frame_delay) # Sleep OUTSIDE the lock

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

        pil_image = Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
        
        start_time = time.time()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT}
                ]
            },
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[pil_image], return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=True,
                temperature=0.6
            )
        
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Reliably get only the text after "Assistant:"
        raw_response = generated_texts[0]
        assistant_marker = "Assistant:"
        if assistant_marker in raw_response:
            response = raw_response.split(assistant_marker, 1)[-1].strip()
        else:
            # Fallback for cases where the marker might be different or absent
            response = raw_response.split("ASSISTANT:")[-1].split("assistant:")[-1].strip()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        with vlm_lock:
            last_description = response
            if processing_time > 0:
                last_fps = 1.0 / processing_time


# --- Flask Routes ---
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen_frames():
    """Generator function for video streaming."""
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
        time.sleep(0.03) # A small delay to prevent overwhelming the browser

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vlm_data')
def vlm_data():
    """Provides VLM data as JSON."""
    with vlm_lock:
        return jsonify({'description': last_description, 'fps': f"{last_fps:.2f}"})


if __name__ == '__main__':
    from waitress import serve
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    vlm_thread = threading.Thread(target=vlm_inference, daemon=True)
    capture_thread.start()
    vlm_thread.start()
    
    print("Starting server with waitress...")
    serve(app, host='0.0.0.0', port=5000)
