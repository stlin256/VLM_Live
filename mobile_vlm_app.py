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

# --- Global Configuration Store (Simplified for Mobile) ---
config = {
    "MODEL_NAME": "./SmolVLM-256M-Instruct-GGUF/SmolVLM2-256M-Video-Instruct-f16.gguf",
    "MMPROJ_MODEL_PATH": "./SmolVLM-256M-Instruct-GGUF/mmproj-SmolVLM2-256M-Video-Instruct-f16.gguf",
    "TRANSLATOR_MODEL_NAME": "./Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
    "PROMPT": "A one-sentence description of the image.",
    "TRANSLATE_PROMPT": "/no_think翻译为中文：",
    "MAX_NEW_TOKENS": 32,
    "N_GPU_LAYERS": 20,
    "TRANSLATOR_N_GPU_LAYERS": 10,
    "TRANSLATE_ENABLED": True # Enabled by default for this version
}
config_lock = threading.Lock()

app = Flask(__name__)

# --- Llama.cpp VLM Initialization ---
print(f"Loading VLM GGUF model from {config['MODEL_NAME']}...")
print(f"--> VLM: Attempting to offload {config.get('N_GPU_LAYERS', 0)} layers to GPU.")
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
print(f"--> Translator: Attempting to offload {config.get('TRANSLATOR_N_GPU_LAYERS', 0)} layers to GPU.")
llm_translator = Llama(
    model_path=config["TRANSLATOR_MODEL_NAME"],
    n_ctx=512,
    n_gpu_layers=config["TRANSLATOR_N_GPU_LAYERS"],
    verbose=False
)
print("Translator GGUF model loaded successfully.")

# --- Thread-safe data storage ---
sessions = {}
sessions_lock = threading.Lock()

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        save_stdout, save_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = save_stdout, save_stderr

def cleanup_inactive_sessions():
    """Periodically removes sessions that have been inactive for too long."""
    while True:
        inactive_threshold = 30  # seconds
        with sessions_lock:
            current_time = time.time()
            inactive_users = [
                user_id for user_id, session in sessions.items()
                if current_time - session.get('last_active', 0) > inactive_threshold
            ]
            for user_id in inactive_users:
                print(f"Cleaning up inactive session for user: {user_id}")
                del sessions[user_id]
        time.sleep(10)

def vlm_inference():
    """Runs VLM inference for all active user sessions."""
    print("VLM inference thread started.")
    while True:
        with sessions_lock:
            user_ids = list(sessions.keys())

        if not user_ids:
            time.sleep(0.5)
            continue

        for user_id in user_ids:
            frame_copy = None
            session_config = {}
            with sessions_lock:
                session = sessions.get(user_id)
                if not session or 'frame' not in session or session['frame'] is None:
                    continue
                frame_copy = session['frame']
                session['frame'] = None  # Consume the frame
                session_config = {
                    "PROMPT": config["PROMPT"],
                    "MAX_NEW_TOKENS": config["MAX_NEW_TOKENS"],
                    "TRANSLATE_PROMPT": config["TRANSLATE_PROMPT"],
                    "TRANSLATE_ENABLED": session.get("translate_enabled", True)
                }

            if frame_copy is None:
                continue

            pil_image = Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
            start_time = time.time()
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{img_str}"

            with suppress_stdout_stderr():
                response = llm_vlm.create_chat_completion(
                    messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}, {"type": "text", "text": session_config["PROMPT"]}]}],
                    max_tokens=session_config["MAX_NEW_TOKENS"]
                )
            description = response['choices'][0]['message']['content']

            if session_config["TRANSLATE_ENABLED"]:
                full_translation_prompt = f"{session_config['TRANSLATE_PROMPT']}\n{description}"
                with suppress_stdout_stderr():
                    trans_response = llm_translator.create_chat_completion(
                        messages=[
                            {"role": "system", "content": "You are a helpful translation assistant."},
                            {"role": "user", "content": full_translation_prompt}
                        ],
                        max_tokens=session_config["MAX_NEW_TOKENS"] * 2
                    )
                translated_text = trans_response['choices'][0]['message']['content'].strip()
                final_text = translated_text.replace("<think>", "").replace("</think>", "").strip()
            else:
                final_text = description

            end_time = time.time()
            processing_time = (end_time - start_time) * 1000

            with sessions_lock:
                if user_id in sessions:
                    sessions[user_id]['description'] = final_text
                    sessions[user_id]['latency'] = processing_time
        time.sleep(0.01)

# --- Flask Routes ---
@app.route('/')
def index():
    """Render the mobile-optimized webpage."""
    return render_template('mobile_index.html')

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """Receive a frame from the mobile client and associate with a user session."""
    data = request.get_json()
    if not data or 'image' not in data or 'user_id' not in data:
        return jsonify({"status": "error", "message": "Incomplete data"}), 400
    
    user_id = data['user_id']

    try:
        header, encoded = data['image'].split(',', 1)
        image_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        with sessions_lock:
            if user_id not in sessions:
                sessions[user_id] = {
                    'frame': None,
                    'description': 'New session started...',
                    'latency': 0.0,
                    'translate_enabled': True
                }
            sessions[user_id]['frame'] = img
            sessions[user_id]['last_active'] = time.time()
            
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error decoding image for user {user_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/vlm_data')
def vlm_data():
    """Provide the latest VLM inference result for a specific user."""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"status": "error", "message": "user_id is required"}), 400
        
    with sessions_lock:
        session = sessions.get(user_id)
        if not session:
            return jsonify({'description': 'Session not found. Please refresh.', 'latency': '0.00'})
        return jsonify({'description': session.get('description', '...'), 'latency': f"{session.get('latency', 0.0):.2f}"})

@app.route('/toggle_translation', methods=['POST'])
def toggle_translation():
    """Enable or disable translation for a specific user session."""
    data = request.get_json()
    if not data or 'user_id' not in data or 'translate' not in data:
        return jsonify({"status": "error", "message": "Incomplete data"}), 400

    user_id = data['user_id']
    translate_enabled = bool(data['translate'])

    with sessions_lock:
        if user_id in sessions:
            sessions[user_id]['translate_enabled'] = translate_enabled
            print(f"Translation for user {user_id} set to: {translate_enabled}")
            return jsonify({"status": "success", "translate_enabled": translate_enabled})
        else:
            # It's possible for this request to arrive before the first frame, so create a session.
            sessions[user_id] = {
                'frame': None, 'description': 'New session started...', 'latency': 0.0,
                'translate_enabled': translate_enabled, 'last_active': time.time()
            }
            print(f"New session for {user_id} started via language toggle. Translation set to: {translate_enabled}")
            return jsonify({"status": "success", "translate_enabled": translate_enabled})

if __name__ == '__main__':
    # To enable camera access from mobile, the server MUST run over HTTPS.
    # You need to generate a self-signed certificate for this to work.
    # Run the following command in your terminal in the project directory:
    # openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
    
    # Start the VLM inference and session cleanup threads
    vlm_thread = threading.Thread(target=vlm_inference, daemon=True)
    vlm_thread.start()
    cleanup_thread = threading.Thread(target=cleanup_inactive_sessions, daemon=True)
    cleanup_thread.start()
    
    host = '0.0.0.0'
    port = 5000
    ssl_context = ('cert.pem', 'key.pem')

    print("Starting mobile-optimized server with HTTPS...")
    print(f"--> Generate SSL certs by running: openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365")
    print(f"--> Access the application on your network at https://<YOUR_COMPUTER_IP>:{port}")
    
    try:
        app.run(host=host, port=port, ssl_context=ssl_context)
    except FileNotFoundError:
        print("\nERROR: SSL certificate files (cert.pem, key.pem) not found.")
        print("Please generate them by running the openssl command shown above and restart the server.")