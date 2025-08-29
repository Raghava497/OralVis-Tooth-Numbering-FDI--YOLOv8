# OralVis-Tooth-Numbering-FDI--YOLOv8

📌 Environment Setup

Python 3.12

torch 2.8.0 + CUDA 12.8

ultralytics==8.3.0

opencv-python-headless==4.10.0.84

scikit-learn==1.6.0

pyyaml==6.0.2

Install dependencies:

pip install ultralytics==8.3.0 opencv-python-headless==4.10.0.84 scikit-learn==1.6.0 pyyaml==6.0.2

📂 Dataset

~500 panoramic dental X-ray images with YOLO-format annotations.

Tooth labels follow FDI numbering system (32 classes, fixed order).

Dataset split: 80% Train / 10% Validation / 10% Test.

Dataset configuration is provided in data.yaml (included in repo).

🚀 Training

To train YOLOv8s with 640×640 input size for 50 epochs:

yolo detect train data=data.yaml model=yolov8s.pt imgsz=640 epochs=50 batch=16 project=oralvis name=tooth_yolo

📊 Evaluation

To evaluate on the test set:

yolo val model=/content/oralvis/tooth_yolo2/weights/best.pt data=data.yaml imgsz=640 split=test

Key Test Metrics (all classes)

Precision: 0.585

Recall: 0.846

mAP@50: 0.746

mAP@50–95: 0.530

📑 Outputs (included in repo)

results.png → training/validation curves

confusion_matrix.png → per-class confusion matrix

PR_curve.png, F1_curve.png → Precision–Recall and F1 curves

best.pt → trained YOLOv8 model weights

Sample predictions → at least 3 annotated images

🧠 Approach Summary

Dataset (~500 panoramic dental X-rays) was prepared in YOLO format with 32 FDI classes.

Data split: 80% train, 10% validation, 10% test.

Used YOLOv8s pretrained weights with input resolution 640×640.

Trained for 50 epochs, batch size 16.

Evaluated using Precision, Recall, mAP@50, and mAP@50–95.

Generated training curves, confusion matrix, PR/F1 curves, and sample predictions.

Results show good detection accuracy across most tooth classes.
