# SJSU AI & ML Club — Nuvoton Team 2
**Edge AI People Counting on the Nuvoton M55M1 EVB**

Deploy a real-time headcount system using a top-down camera and a model accelerated by the Ethos-U55 NPU. All inference runs locally on-device — no cloud or external storage.

## Goals
- ≥95% headcount accuracy
- ≥15 FPS inference
- Zero-Cloud: all processing on-device, no data transmitted externally

---

## Resources
- [ML_YOLO Repo](https://github.com/OpenNuvoton/ML_YOLO) — training, export, and quantization pipeline
- [ML_M55M1_SampleCode](https://github.com/OpenNuvoton/ML_M55M1_SampleCode) — sample M55M1 ML application reference
- [M55M1 BSP](https://github.com/OpenNuvoton/M55M1BSP) — board support package for lower-level firmware integration
- [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise)

---

## Environment Setup

### 1. Create and activate virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio ultralytics opencv-python numpy matplotlib pyyaml
```

### 3. Clone required repositories

```bash
mkdir -p repos
cd repos
git clone https://github.com/OpenNuvoton/ML_YOLO.git
git clone https://github.com/OpenNuvoton/ML_M55M1_SampleCode.git
cd ..
```

### 4. Download baseline YOLOv8 weights

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## ML Training Setup (YOLOv8)

### Dataset Format
The training dataset must follow YOLO object detection format:

```text
dataset/
├── dataset.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

- One class: `person`
- Each image has a matching `.txt` label file
- YOLO label format:
```text
class x_center y_center width height
```

### Example `dataset.yaml`

```yaml
path: .

train: dataset/images/train
val: dataset/images/val

names:
  0: person
```

### Training Command
Run from the project root:

```bash
python repos/ML_YOLO/yolov8_ultralytics/dg_train.py \
  --model-cfg repos/ML_YOLO/yolov8_ultralytics/ultralytics/cfg/models/v8/relu6-yolov8.yaml \
  --weights yolov8n.pt \
  --data dataset/dataset.yaml \
  --imgsz 192 \
  --epochs 50 \
  --batch 8 \
  --device cpu \
  --project runs/train \
  --name elevator_v1
```

---

## Notes
- Using YOLOv8 ReLU6 for better Ethos-U55 compatibility
- Counting is done by the number of detected person bounding boxes
- The model config file `relu6-yolov8.yaml` should be edited to use `nc: 1`
- A `test/` split can be added later for final evaluation
- `people_count_demo.py` is a local baseline inference script for testing on a laptop; it is not the board deployment pipeline