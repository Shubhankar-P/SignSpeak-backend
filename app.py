# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# 

#import eventlet
#eventlet.monkey_patch()

from flask import Flask, render_template
from flask_socketio import SocketIO
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
import base64
from collections import deque

# --------------------------------------------------
# Warnings
# --------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead."
)

# --------------------------------------------------
# Flask + Socket.IO setup
# --------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")


# --------------------------------------------------
# Load ML model ONCE
# --------------------------------------------------
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading the model:", e)
    model = None

# --------------------------------------------------
# Initialize MediaPipe ONCE  (CRITICAL FIX)
# --------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --------------------------------------------------
# Labels dictionary ONCE  (CRITICAL FIX)
# --------------------------------------------------
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry',
    31: 'Please', 32: 'You are welcome.'
}

# --------------------------------------------------
# Prediction smoothing buffer
# --------------------------------------------------
pred_buffer = deque(maxlen=5)

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("🔗 Client connected")

# --------------------------------------------------
# Receive frame from frontend
# --------------------------------------------------
@socketio.on('frame')
def handle_frame(data):
    print("FRAME RECEIVED")
    try:
        # Decode base64 image
        encoded = data.split(',')[1]
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        # Mirror view
       # frame = cv2.flip(frame, 1)

        process_frame(frame)

    except Exception as e:
        print("Frame handling error:", e)

# --------------------------------------------------
# Process ONE frame only (NO loops, NO camera)
# --------------------------------------------------
def process_frame(frame):
    if model is None:
        return

    data_aux = []
    x_ = []
    y_ = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return

    for hand_landmarks in results.multi_hand_landmarks:

        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        try:
            prediction = model.predict([np.asarray(data_aux)])
            prediction_proba = model.predict_proba([np.asarray(data_aux)])
            confidence = float(max(prediction_proba[0]))

            predicted_character = labels_dict[int(prediction[0])]

            # Add prediction to buffer
            pred_buffer.append(predicted_character)

            # Get most common prediction in last 5 frames
            final_prediction = max(set(pred_buffer), key=pred_buffer.count)

            socketio.emit(
                'prediction',
                {
                    'text': final_prediction,
                    'confidence': confidence
                }
            )

        except Exception as e:
            print("Prediction error:", e)

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

