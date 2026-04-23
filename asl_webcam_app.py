import cv2
import numpy as np
import json
import os
import sys
import collections
import time

# CONFIG
MODEL_PATH      = "asl_model.h5"
CLASS_NAMES_PATH= "class_names.json"
IMG_SIZE        = 128
ROI_SIZE        = 300
CONFIDENCE_MIN  = 0.60
SMOOTH_FRAMES   = 10
FLIP_CAMERA     = True

# UI Colors (BGR)
COLOR_BOX       = (0, 220, 80)
COLOR_TEXT_BG   = (15, 15, 15)
COLOR_ACCENT    = (0, 200, 255)
COLOR_WARN      = (0, 100, 255)
COLOR_WHITE     = (255, 255, 255)

# LOAD MODEL
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        print(f"  Model not found: {MODEL_PATH}")
        print("    Train the model first using asl_train.ipynb")
        sys.exit(1)
    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"  class_names.json not found.")
        sys.exit(1)

    import tensorflow as tf
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH) as f:
        raw = json.load(f)
    # Keys are string indices
    class_names = {int(k): v for k, v in raw.items()}
    print(f" Model loaded — {len(class_names)} classes")
    return model, class_names


# PREPROCESSING
def preprocess_roi(roi):
    """Resize and normalize the hand ROI for model input."""
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


# DRAWING HELPERS
def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=12):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_label_panel(frame, letter, confidence, top_preds, class_names, fps):
    h, w = frame.shape[:2]
    panel_w = 220

    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    x = w - panel_w + 12
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, "ASL DETECTOR", (x, 35), font, 0.55, COLOR_ACCENT, 1, cv2.LINE_AA)
    cv2.line(frame, (w - panel_w, 48), (w, 48), (60, 60, 60), 1)

    cv2.putText(frame, "Detected:", (x, 80), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    if letter and confidence >= CONFIDENCE_MIN:
        cv2.putText(frame, letter, (x, 150), font, 3.5, COLOR_BOX, 4, cv2.LINE_AA)
        pct = f"{confidence * 100:.1f}%"
        cv2.putText(frame, pct, (x, 175), font, 0.55, COLOR_BOX, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "?", (x, 150), font, 3.5, (80, 80, 80), 3, cv2.LINE_AA)
        cv2.putText(frame, "Low confidence", (x, 175), font, 0.42, COLOR_WARN, 1, cv2.LINE_AA)

    cv2.line(frame, (w - panel_w, 195), (w, 195), (60, 60, 60), 1)
    cv2.putText(frame, "Top predictions:", (x, 218), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    for i, (idx, conf) in enumerate(top_preds[:3]):
        name = class_names.get(idx, "?")
        bar_w = int((panel_w - 24) * conf)
        bar_y = 230 + i * 38
        # Background bar
        cv2.rectangle(frame, (x, bar_y), (x + panel_w - 24, bar_y + 18), (50, 50, 50), -1)
        # Filled bar
        bar_color = COLOR_BOX if i == 0 else (100, 150, 100)
        cv2.rectangle(frame, (x, bar_y), (x + bar_w, bar_y + 18), bar_color, -1)
        # Label
        label = f"{name}  {conf * 100:.0f}%"
        cv2.putText(frame, label, (x + 4, bar_y + 13), font, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)

    cv2.line(frame, (w - panel_w, h - 40), (w, h - 40), (60, 60, 60), 1)
    cv2.putText(frame, f"FPS: {fps:.0f}", (x, h - 18), font, 0.45, (130, 130, 130), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q: quit  R: reset", (x, h - 5), font, 0.38, (100, 100, 100), 1, cv2.LINE_AA)


def draw_roi_box(frame, cx, cy, size, color):
    x1, y1 = cx - size // 2, cy - size // 2
    x2, y2 = cx + size // 2, cy + size // 2
    draw_rounded_rect(frame, (x1, y1), (x2, y2), color, thickness=2)
    # Corner accents
    L = 20
    t = 3
    for px, py, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (px, py), (px + dx * L, py), color, t)
        cv2.line(frame, (px, py), (px, py + dy * L), color, t)
    label = "Place hand here"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)
    return x1, y1, x2, y2


def run():
    model, class_names = load_model_and_classes()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check camera permissions.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 30)

    prediction_buffer = collections.deque(maxlen=SMOOTH_FRAMES)
    fps_timer = time.time()
    fps = 0.0
    frame_count = 0

    print("\n Webcam started!")
    print("   → Position your hand in the green box")
    print("   → Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        if FLIP_CAMERA:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        panel_w = 220
        # ROI center — slightly left of center to leave panel space
        cx = (w - panel_w) // 2
        cy = h // 2

        x1, y1, x2, y2 = draw_roi_box(frame, cx, cy, ROI_SIZE, COLOR_BOX)

        rx1, ry1 = max(0, x1), max(0, y1)
        rx2, ry2 = min(w, x2), min(h, y2)
        roi = frame[ry1:ry2, rx1:rx2]

        letter, confidence, top_preds = None, 0.0, []

        if roi.size > 0:
            inp = preprocess_roi(roi)
            preds = model.predict(inp, verbose=0)[0]

            prediction_buffer.append(preds)
            avg_preds = np.mean(prediction_buffer, axis=0)

            top_idx = np.argsort(avg_preds)[::-1][:5]
            top_preds = [(i, float(avg_preds[i])) for i in top_idx]

            best_idx = top_idx[0]
            confidence = float(avg_preds[best_idx])
            letter = class_names.get(best_idx, "?")

        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        draw_label_panel(frame, letter, confidence, top_preds, class_names, fps)

        cv2.imshow("ASL Alphabet Detector — Press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            prediction_buffer.clear()
            print("Buffer reset.")

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    run()