from ultralytics import YOLO
import cv2
import torch

VIDEO_PATH = "data/raw_videos/test.mp4"
OUTPUT_PATH = "outputs/videos/count_demo.mp4"

PERSON_CLASS_ID = 0

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps if fps > 0 else 20,
    (width, height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, device=device, verbose=False)[0]

    count = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == PERSON_CLASS_ID and conf > 0.35:
            count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.putText(
        frame,
        f"Count: {count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    writer.write(frame)

cap.release()
writer.release()

print("Saved:", OUTPUT_PATH)
