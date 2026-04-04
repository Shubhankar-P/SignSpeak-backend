import os
import shutil
import mediapipe as mp
import cv2

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR       = './data'
PROBLEM_DIR    = './data_problematic'   # where bad images will be moved
EXPECTED_LEN   = 63                     # 21 landmarks × 3 (x, y, z)
# ────────────────────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

moved_count   = 0
checked_count = 0

for class_dir in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_dir)

    # Skip non-directories (e.g. stray files)
    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        checked_count += 1

        # ── Read & convert image ────────────────────────────────────────────
        img = cv2.imread(img_path)
        if img is None:                         # unreadable file
            reason = "unreadable"
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                reason = "no_hand_detected"
            else:
                # Replicate your exact data_aux build logic
                data_aux = []
                x_ , y_  = [], []

                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x)
                        y_.append(lm.y)

                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_))
                        data_aux.append(lm.y - min(y_))
                        data_aux.append(lm.z)

                if len(data_aux) == EXPECTED_LEN:
                    continue                     # ✅ good image — skip it
                else:
                    reason = f"wrong_length_{len(data_aux)}"

        # ── Move the bad image ──────────────────────────────────────────────
        dest_dir = os.path.join(PROBLEM_DIR, reason, class_dir)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(img_path, os.path.join(dest_dir, img_name))

        moved_count += 1
        print(f"[MOVED] {class_dir}/{img_name}  →  {reason}/")

print(f"\nDone. Checked: {checked_count} | Moved: {moved_count} | OK: {checked_count - moved_count}")