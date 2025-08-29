# Tooth Numbering Detection (FDI) 

End-to-end pipeline to detect and label teeth using the **FDI system (32 classes)**.  
This README covers **environment setup, dataset layout, training, evaluation on TEST, predictions, metrics (Precision/Recall/mAP), confusion matrix, comparison, and troubleshooting**.


## Table of Contents
- [Overview](#overview)
- [Dataset Layout & `data.yaml`](#dataset-layout--datayaml)
- [Environment Setup](#environment-setup)
  - [Google Colab (recommended)](#google-colab-recommended)
  - [Local (Conda/venv)](#local-condavenv)
- [Training & Evaluation](#training--evaluation)
  - [YOLOv11 (Ultralytics)](#yolov11-ultralytics)
  - [YOLOv8 (Ultralytics)](#yolov8-ultralytics)
  - [YOLOv5 (Ultralytics repo)](#yolov5-ultralytics-repo)
  - [YOLOv7](#yolov7)
- [Predictions (rendered images + labels)](#predictions-rendered-images--labels)
- [Key Metrics & Confusion Matrix](#key-metrics--confusion-matrix)
- [Model Comparison (one table)](#model-comparison-one-table)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Repo Structure](#suggested-repo-structure)
- [Acknowledgements](#acknowledgements)


## Overview
- **Task:** Object detection of **32 FDI tooth classes** on intra-oral images.
- **Labels:** YOLO format per line → `class cx cy w h` (all values normalized to `[0,1]`).
- **Split:** `80/10/10` → **train/val/test** (by filename stem).
- **Image size:** 640×640.
- **Models:** YOLOv5s, YOLOv7-tiny (or full), YOLOv8s, YOLOv11s.
- **Primary selection metric:** **mAP@50-95** on the **test** split (tie-breakers: Recall → Precision).


## Dataset Layout & `data.yaml`

Required directory layout:
```

dataset/
images/
train/  \*.jpg|png|jpeg
val/    \*.jpg|png|jpeg
test/   \*.jpg|png|jpeg
labels/
train/  \*.txt
val/    \*.txt
test/   \*.txt

````

Create a `data.yaml` at project root (or `/content/data.yaml` in Colab):

```yaml
path: /content/dataset
train: images/train
val: images/val
test: images/test
names:
  - Canine (13)              # 0
  - Canine (23)              # 1
  - Canine (33)              # 2
  - Canine (43)              # 3
  - Central Incisor (21)     # 4
  - Central Incisor (41)     # 5
  - Central Incisor (31)     # 6
  - Central Incisor (11)     # 7
  - First Molar (16)         # 8
  - First Molar (26)         # 9
  - First Molar (36)         # 10
  - First Molar (46)         # 11
  - First Premolar (14)      # 12
  - First Premolar (34)      # 13
  - First Premolar (44)      # 14
  - First Premolar (24)      # 15
  - Lateral Incisor (22)     # 16
  - Lateral Incisor (32)     # 17
  - Lateral Incisor (42)     # 18
  - Lateral Incisor (12)     # 19
  - Second Molar (17)        # 20
  - Second Molar (27)        # 21
  - Second Molar (37)        # 22
  - Second Molar (47)        # 23
  - Second Premolar (15)     # 24
  - Second Premolar (25)     # 25
  - Second Premolar (35)     # 26
  - Second Premolar (45)     # 27
  - Third Molar (18)         # 28
  - Third Molar (28)         # 29
  - Third Molar (38)         # 30
  - Third Molar (48)         # 31
````

**Sanity tips**

* Each label line must have **5 tokens**.
* Class IDs ∈ **\[0,31]**.
* `cx,cy,w,h ∈ [0,1]` (normalized), else fix or quarantine affected files before training.


## Environment Setup

### Google Colab (recommended)

* **Python:** 3.12 (Colab default)
* **GPU:** T4/A100
* **Packages used here:**

  * Ultralytics `8.3.189`, NumPy `>=2.0,<2.4`, OpenCV-headless `4.10.0.84`, Matplotlib `3.9.0`
  * YOLOv5 and YOLOv7 are separate repos with their own `requirements.txt`.

> You’ll see exact `pip`/`git clone` lines in the training sections.

### Local (Conda/venv)

```bash
conda create -n toothdet python=3.10 -y
conda activate toothdet
# For Ultralytics (v8/v11):
pip install "ultralytics==8.3.189" "numpy>=2.0,<2.4" opencv-python-headless==4.10.0.84 matplotlib==3.9.0
# For YOLOv5:
git clone -b v7.0 https://github.com/ultralytics/yolov5 && cd yolov5
pip install -r requirements.txt "numpy>=2.0,<2.4" matplotlib==3.9.0
# For YOLOv7:
git clone https://github.com/WongKinYiu/yolov7 && cd yolov7
pip install -r requirements.txt "numpy>=2.0,<2.4" pycocotools matplotlib==3.9.0
```


## Training & Evaluation

> Common hyperparams: `imgsz=640`, `epochs=100`, `batch=-1 (auto)`, `seed=42`, patience/early-stop where available.

### YOLOv11 (Ultralytics)

```python
# Install
!pip -q install -U "ultralytics==8.3.189" "numpy>=2.0,<2.4" opencv-python-headless==4.10.0.84
from ultralytics import YOLO

# Train
model = YOLO("yolo11s.pt")
results = model.train(
    data="/content/data.yaml", imgsz=640, epochs=100, batch=-1, device=0,
    project="/content/runs_v11", name="tooth_yolo11s", exist_ok=True,
    seed=42, patience=20, verbose=True, plots=True
)

# Evaluate on TEST
best = YOLO("/content/runs_v11/tooth_yolo11s/weights/best.pt")
metrics = best.val(
    data="/content/data.yaml", split="test", imgsz=640,
    conf=0.001, iou=0.6, plots=True, save_json=True,
    project="/content/runs_v11", name="tooth_yolo11s_test", exist_ok=True
)
print(metrics.results_dict)  # precision, recall, mAP@50, mAP@50-95
```

### YOLOv8 (Ultralytics)

```python
# Install
!pip -q install -U "ultralytics==8.3.189" "numpy>=2.0,<2.4" opencv-python-headless==4.10.0.84
from ultralytics import YOLO

# Train
model = YOLO("yolov8s.pt")
model.train(
    data="/content/data.yaml", imgsz=640, epochs=100, batch=-1, device=0,
    project="runs", name="tooth_yolov8s", exist_ok=True,
    seed=42, patience=20, verbose=True, plots=True
)

# Evaluate on TEST
YOLO("runs/detect/tooth_yolov8s/weights/best.pt").val(
    data="/content/data.yaml", split="test", imgsz=640,
    conf=0.001, iou=0.6, plots=True, save_json=True,
    project="runs", name="tooth_yolov8s_test", exist_ok=True
)
```

### YOLOv5 (Ultralytics repo)

```python
# Install
%cd /content
!rm -rf yolov5
!git clone -q --depth 1 -b v7.0 https://github.com/ultralytics/yolov5
%cd yolov5
!pip -q install -r requirements.txt "numpy>=2.0,<2.4" matplotlib==3.9.0

# Train
!python train.py \
  --img 640 --epochs 100 --batch -1 \
  --data /content/data.yaml \
  --weights yolov5s.pt \
  --project /content/runs_v5 --name tooth_yolov5 --exist-ok

# Evaluate on TEST
!python val.py \
  --img 640 \
  --data /content/data.yaml \
  --weights /content/runs_v5/tooth_yolov5/weights/best.pt \
  --task test \
  --conf 0.001 --iou 0.6 \
  --project /content/runs_v5 --name tooth_v5_test --exist-ok --plots --save-json
```

### YOLOv7

```python
# Install
%cd /content
!rm -rf yolov7
!git clone -q https://github.com/WongKinYiu/yolov7
%cd yolov7
!pip -q install -r requirements.txt "numpy>=2.0,<2.4" pycocotools matplotlib==3.9.0

# Train (tiny for speed; swap to full cfg/weights for higher accuracy)
!python train.py \
  --workers 2 --device 0 --batch-size -1 \
  --data /content/data.yaml \
  --img 640 640 \
  --cfg cfg/training/yolov7-tiny.yaml \
  --weights yolov7-tiny.pt \
  --name tooth_yolov7 \
  --epochs 100 \
  --project /content/runs_v7 --exist-ok

# Evaluate on TEST
!python test.py \
  --data /content/data.yaml \
  --img-size 640 \
  --batch-size 16 \
  --conf-thres 0.001 --iou-thres 0.6 \
  --device 0 \
  --weights /content/runs_v7/tooth_yolov7/weights/best.pt \
  --task test \
  --project /content/runs_v7 --name tooth_v7_test --exist-ok --save-json
```


## Predictions (rendered images + labels)

Ultralytics (v8/v11):

```python
YOLO("/path/to/best.pt").predict(
  source="/content/dataset/images/test", imgsz=640,
  conf=0.25, iou=0.6, save=True, save_txt=True,
  project="/content/runs_preds", name="pred_test", exist_ok=True
)
```

YOLOv5:

```python
%cd /content/yolov5
!python detect.py \
  --weights /content/runs_v5/tooth_yolov5/weights/best.pt \
  --img 640 --conf 0.25 --iou 0.6 \
  --source /content/dataset/images/test \
  --project /content/runs_v5 --name pred_test --exist-ok --save-txt
```

YOLOv7:

```python
%cd /content/yolov7
!python detect.py \
  --weights /content/runs_v7/tooth_yolov7/weights/best.pt \
  --img-size 640 --conf 0.25 --iou 0.6 \
  --source /content/dataset/images/test \
  --project /content/runs_v7 --name pred_test --exist-ok --save-txt
```


## Key Metrics & Confusion Matrix

**Ultralytics (v8/v11):**

* `val(..., split="test", plots=True)` saves:

  * `confusion_matrix.png`, `PR_curve.png`, `results.csv`
  * `metrics.results_dict` contains **precision**, **recall**, **mAP\@50**, **mAP\@50-95**

**YOLOv5/YOLOv7:**

* `val.py`/`test.py` writes:

  * `results.txt` with a line beginning `all` → `P, R, mAP@.5, mAP@.5:.95`
  * `confusion_matrix.png`

**One-shot table for all models (auto-find + print + save CSV):**

```python
# Produces /content/key_metrics_all_models.csv after evaluations are run.
import os, glob, re
from pathlib import Path
import pandas as pd

def latest(p): 
    c = sorted(glob.glob(p), key=os.path.getmtime, reverse=True)
    return c[0] if c else None

def parse_results_txt(run_glob, name):
    files = sorted(glob.glob(f"{run_glob}/*test*/results.txt"), key=os.path.getmtime, reverse=True) \
         or sorted(glob.glob(f"{run_glob}/*/results.txt"), key=os.path.getmtime, reverse=True)
    if not files: return None
    txt = Path(files[0]).read_text().splitlines()
    line = next((ln for ln in txt if ln.strip().startswith("all ") or "all" in ln), None)
    if not line: return None
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
    if len(nums) < 4: return None
    return {"model": Path(files[0]).parent.name, "framework": name,
            "precision": nums[-4], "recall": nums[-3],
            "mAP@50": nums[-2], "mAP@50-95": nums[-1],
            "ckpt_or_run": str(Path(files[0]).parent)}

rows = []

# Ultralytics v11/v8 (need prior .val() runs)
try:
    from ultralytics import YOLO
    for label, pat, proj, nm in [
        ("YOLOv11", "/content/runs_v11/*/weights/best.pt", "/content/runs_v11", "tooth_yolo11s_test"),
        ("YOLOv8", "runs/detect/*/weights/best.pt", "runs", "tooth_yolov8s_test")
    ]:
        ckpt = latest(pat) or latest(pat.replace("best","last"))
        if ckpt and Path(ckpt).exists():
            res = YOLO(ckpt).val(data="/content/data.yaml", split="test", imgsz=640,
                                  conf=0.001, iou=0.6, plots=False, save_json=False,
                                  project=proj, name=nm, exist_ok=True, verbose=False)
            rd = getattr(res, "results_dict", {}) or {}
            def g(d,k): 
                return float(d.get(k, d.get(k.replace("(B)",""), float("nan"))))
            rows.append({"model": Path(ckpt).name, "framework": label,
                         "precision": g(rd,"metrics/precision(B)"),
                         "recall":    g(rd,"metrics/recall(B)"),
                         "mAP@50":    g(rd,"metrics/mAP50(B)"),
                         "mAP@50-95": g(rd,"metrics/mAP50-95(B)"),
                         "ckpt_or_run": ckpt})
except Exception as e:
    print("Ultralytics not available or eval failed:", e)

# YOLOv5 / YOLOv7 (parse)
v5 = parse_results_txt("/content/runs_v5", "YOLOv5")
if v5: rows.append(v5)
v7 = parse_results_txt("/content/runs_v7", "YOLOv7")
if v7: rows.append(v7)

if rows:
    df = pd.DataFrame(rows).sort_values("mAP@50-95", ascending=False)
    print(df.to_string(index=False))
    df.to_csv("/content/key_metrics_all_models.csv", index=False)
    print("\nSaved → /content/key_metrics_all_models.csv")
else:
    print("No metrics found. Run evaluations first.")
```


## Model Comparison (one table)

* Choose the **best** by **mAP\@50-95** on the **test** split.
* Use **Recall** then **Precision** as tie-breakers.
* Also inspect a few **rendered predictions** per model for qualitative sanity.



## Reproducibility

* Fixed splits and seeds (`seed=42`) for all trainings.
* Same image size (`640`) and evaluation thresholds (`conf=0.001`, `IoU=0.6`) on the **test** set across models.
* Keep artifacts: `best.pt`, `results.png` (training curves), `confusion_matrix.png`, PR curves, sample predictions, and key-metrics CSVs.


## Troubleshooting

* **NumPy/ABI error (e.g., `numpy.dtype size changed`)**
  Reinstall compatible wheels **then restart runtime**:

  ```python
  !pip install -U --force-reinstall --no-cache-dir "numpy>=2.0,<2.4" "ultralytics==8.3.189"
  import os; os.kill(os.getpid(), 9)
  ```
* **Checkpoint not found (`best.pt`)**
  The run folder name may differ (e.g., `..._2`). Use a glob or the printed `results.save_dir` to locate it.
* **No rendered images**
  Detect likely failed earlier due to wrong weights path; fix the path and rerun detect.


##  Repo Structure

```
.
├── data.yaml
├── README.md
├── notebooks/
│   └── tooth_detection_colab.ipynb
├── scripts/
│   ├── split_dataset.py
│   ├── eval_key_metrics.py
│   └── make_report_docx.py
├── runs_v5/        # (gitignored or saved to Drive)
├── runs_v7/
├── runs/           # YOLOv8 default
└── runs_v11/
```





## Acknowledgements

* **Ultralytics YOLO** (v8/v11)
* **YOLOv5** by Ultralytics
* **YOLOv7** by WongKinYiu et al.

