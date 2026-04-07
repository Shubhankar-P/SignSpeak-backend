# SignSpeak — ML Backend

> Real-time ASL sign language classification server using MediaPipe and Random Forest.

This is the Python backend for the [SignSpeak Android app](https://github.com/Shubhankar-P/SignSpeak-frontend). It receives live camera frames from the Android client via WebSocket, runs hand landmark detection using MediaPipe, and classifies the gesture using a trained Random Forest model. Predictions are returned to the client in real time.

---

## Model Performance

| Metric | Value |
|---|---|
| Accuracy | 96.57% |
| MCC Score | 0.9646 |
| Macro F1-Score | 0.97 |
| Weighted F1-Score | 0.97 |
| Cross-validation F1 (5-fold) | 0.9604 ± 0.0054 |
| Avg. Inference Time | 5.414 ms |
| Max Inference Time | 13.200 ms |
| Classes | 33 ASL signs |
| Dataset Size | 5,823 images (self-collected) |

---

## Supported Signs

A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, Hello, Done, Thank You, I Love You, Sorry, Please, You Are Welcome

> Static gestures only. Dynamic signs requiring motion (J, Z) have limited support.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Server | Flask + Flask-SocketIO |
| Hand Detection | Google MediaPipe Hands |
| ML Model | Random Forest (scikit-learn) |
| Image Processing | OpenCV |
| Numerical | NumPy |
| Serialisation | Python pickle |

---

## Project Structure

```
SignSpeak-backend/
├── app.py                  # Flask + SocketIO server — main inference endpoint
├── create_dataset.py       # Extract landmarks from images → data.pickle
├── train_classifier.py     # Train Random Forest → model.p
├── evaluate_model.py       # Run full model evaluation → evaluation_results/
├── check_data.py           # Dataset cleaning utility
├── model.p                 # Trained model (generated — not committed)
├── data.pickle             # Processed dataset (generated — not committed)
├── data/                   # Raw image dataset organised by class folder
│   ├── 0/                  # Sign A
│   ├── 1/                  # Sign B
│   └── ...
├── data_problematic/       # Images moved here by check_data.py
├── evaluation_results/     # Evaluation outputs (charts + CSV)
│   ├── confusion_matrix.png
│   ├── learning_curve.png
│   ├── feature_importance.png
│   └── classification_report.csv
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SignSpeak-backend.git
cd SignSpeak-backend
```

### 2. Create and activate virtual environment

```bash
py -3.10 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run these three scripts in order:

```bash
# Step 1 — Remove bad images from the dataset
python check_data.py

# Step 2 — Extract landmarks and build data.pickle
python create_dataset.py

# Step 3 — Train the classifier and save model.p
python train_classifier.py
```

---

## Evaluating the Model

```bash
python evaluate_model.py
```

This runs 5 tests and saves all outputs to `evaluation_results/`:

| Test | Output |
|---|---|
| Accuracy, Precision, Recall, F1 | Printed to console + `classification_report.csv` |
| Confusion Matrix | `confusion_matrix.png` |
| 5-Fold Cross-Validation | Printed to console |
| Learning Curve | `learning_curve.png` |
| Feature Importance | `feature_importance.png` |

---

## Running the Server

```bash
python app.py
```

Server starts on `http://0.0.0.0:5000`. Expose it to the Android app using a tunneling service:

```bash
ngrok http 5000
```

Copy the generated URL into your Android app's `local.properties` as `BACKEND_URL`.

---

## Dataset

- **Standard:** American Sign Language (ASL)
- **Size:** 5,823 images across 33 classes (~200 per class)
- **Type:** Static hand gesture images (single-frame)
- **Origin:** Self-collected by the project team

### Dataset Cleaning

`check_data.py` scans the dataset and moves problematic images to `data_problematic/`. An image is flagged if:

1. OpenCV cannot read the file
2. MediaPipe detects no hand
3. MediaPipe detects more than one hand
4. The extracted feature vector is not exactly 63 values

Only images producing exactly 63 features (21 landmarks × 3 coordinates: x, y, z) are used for training.

---

## How It Works

```
Camera frame (Android)
        │
        ▼
Base64 decode + BGR→RGB (OpenCV)
        │
        ▼
MediaPipe Hands — 21 landmarks detected
        │
        ▼
Feature extraction — 63 values (x−min_x, y−min_y, z)
        │
        ▼
Random Forest — 200 trees vote
        │
        ▼
Prediction + confidence → SocketIO → Android
```

**Normalisation:** x and y coordinates are shifted by the per-frame minimum, making predictions invariant to hand position in the frame. z (depth) is used as-is.

**Prediction smoothing (Android-side):** The last 3 predictions are buffered; the most frequent result is displayed, reducing flicker.

---

## Standalone Inference (No Mobile Required)

To test inference using your webcam directly:

```bash
python inference_classifier.py
```

Uses OpenCV for camera capture with 20-frame majority voting. Press `q` to quit.

---

## Team

Developed as a Final Year Project (BE, Semester VIII) at **Vidyalankar Institute of Technology**.

| Name |
|---|
| Shubhankar Potnis |
| Riddhi Welekar |
| Aayush Samanta |
| Aishwarya Pawar |
