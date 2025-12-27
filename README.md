# Brain Tumor Segmentation using YOLO11

This project implements a system for detecting and segmenting brain tumors from MRI scans using the **YOLO11**. By leveraging instance segmentation, the model not only identifies the presence of a tumor but also delineates its exact boundaries, providing critical spatial information for medical analysis.

## 🚀 Project Overview

Early and accurate detection of brain tumors is vital for treatment planning. This repository utilizes the latest YOLO11 segmentation model to achieve high-speed, high-accuracy results on MRI datasets.

* **Model Architecture:** YOLO11n-seg (Nano Segmentation)
* **Task:** Instance Segmentation (Classifying and masking tumor regions)
* **Framework:** Ultralytics

---

## 📊 Dataset Information

The model was trained using a specialized MRI dataset sourced from Roboflow.

* **Source:** [Roboflow Universe - Brain Tumor Segmentation](https://www.google.com/search?q=https://universe.roboflow.com/mri-brain-tumor/brain-tumor-segmentation-8aek5/dataset/1)
* **Total Images:** 257 images
* **Classes:** 1. `Tumor`: Areas identified with abnormal growths.
2. `no_tumor`: Healthy brain scans (used for negative control).
* **Pre-processing:** Auto-orientation and resizing to  pixels.

---

## 📈 Performance Results

The model was evaluated on a test set using a Tesla T4 GPU. Below are the key metrics achieved:

### Summary Metrics

| Metric | Value |
| --- | --- |
| **Box mAP50** | 0.868 |
| **Mask mAP50** | 0.867 |
| **Mask mAP50-95** | 0.580 |
| **Inference Speed** | ~22.4ms per image |

### Class-wise Performance

| Class | Images | Instances | Box (P) | Box (R) | Mask (P) | Mask (R) |
| --- | --- | --- | --- | --- | --- | --- |
| **Tumor** | 31 | 31 | 0.822 | 0.871 | 0.822 | 0.871 |
| **No Tumor** | 4 | 4 | 0.574 | 1.000 | 0.574 | 1.000 |
| **All** | 35 | 35 | 0.698 | 0.935 | 0.698 | 0.935 |

---

## 🛠️ Installation & Usage

### Prerequisites

* Python 3.12+
* CUDA-enabled GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/brain-tumor-segmentation.git
cd brain-tumor-segmentation

```


2. Install dependencies:
```bash
pip install ultralytics

```



### Training/Validation

To run the validation using the provided notebook logic:

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('path/to/your/best.pt')

# Validate the model
results = model.val(data='path/to/data.yaml')
print(f"Mask mAP50: {results.seg.map50}")

```

---

## 🔬 Key Technical Highlights

* **Efficient Segmentation:** With only **2.8 million parameters**, the YOLO11n-seg model provides a lightweight yet powerful solution suitable for near real-time medical imaging applications.
* **High Recall:** The model achieved a recall of **0.935**, indicating a very low rate of false negatives—crucial in a medical diagnostic context.
* **Inference Speed:** Total processing time (preprocess to postprocess) is approximately **45.2ms**, enabling rapid screening of large MRI batches.

---

## 📄 License

This project is intended for educational and research purposes. Please refer to the Roboflow dataset link for specific data usage licenses.

Would you like me to add a section on how to visualize the segmentation masks on new images?This README is designed to be professional, clear, and highlights the impressive performance metrics you achieved with the YOLO11n-seg model.

---

# Brain Tumor Segmentation using YOLO11

This project implements an automated system for detecting and segmenting brain tumors from MRI scans using the **YOLO11 (Ultralytics)** architecture. By leveraging instance segmentation, the model not only identifies the presence of a tumor but also delineates its exact boundaries, providing critical spatial information for medical analysis.

## 🚀 Project Overview

Early and accurate detection of brain tumors is vital for treatment planning. This repository utilizes the latest YOLO11 segmentation model to achieve high-speed, high-accuracy results on MRI datasets.

* **Model Architecture:** YOLO11n-seg (Nano Segmentation)
* **Task:** Instance Segmentation (Classifying and masking tumor regions)
* **Framework:** Ultralytics / PyTorch

---

## 📊 Dataset Information

The model was trained using a specialized MRI dataset sourced from Roboflow.

* **Source:** [Roboflow Universe - Brain Tumor Segmentation](https://www.google.com/search?q=https://universe.roboflow.com/mri-brain-tumor/brain-tumor-segmentation-8aek5/dataset/1)
* **Total Images:** 257 images
* **Classes:** 1. `Tumor`: Areas identified with abnormal growths.
2. `no_tumor`: Healthy brain scans (used for negative control).
* **Pre-processing:** Auto-orientation and resizing to  pixels.

---

## 📈 Performance Results

The model was evaluated on a test set using a Tesla T4 GPU. Below are the key metrics achieved:

### Summary Metrics

| Metric | Value |
| --- | --- |
| **Box mAP50** | 0.868 |
| **Mask mAP50** | 0.867 |
| **Mask mAP50-95** | 0.580 |
| **Inference Speed** | ~22.4ms per image |

### Class-wise Performance

| Class | Images | Instances | Box (P) | Box (R) | Mask (P) | Mask (R) |
| --- | --- | --- | --- | --- | --- | --- |
| **Tumor** | 31 | 31 | 0.822 | 0.871 | 0.822 | 0.871 |
| **No Tumor** | 4 | 4 | 0.574 | 1.000 | 0.574 | 1.000 |
| **All** | 35 | 35 | 0.698 | 0.935 | 0.698 | 0.935 |

---

## 🛠️ Installation & Usage

### Prerequisites

* Python 3.12+
* CUDA-enabled GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/brain-tumor-segmentation.git
cd brain-tumor-segmentation

```


2. Install dependencies:
```bash
pip install ultralytics torch

```



### Training/Validation

To run the validation using the provided notebook logic:

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('path/to/your/best.pt')

# Validate the model
results = model.val(data='path/to/data.yaml')
print(f"Mask mAP50: {results.seg.map50}")

```

---

## 🔬 Key Technical Highlights

* **Efficient Segmentation:** With only **2.8 million parameters**, the YOLO11n-seg model provides a lightweight yet powerful solution suitable for near real-time medical imaging applications.
* **High Recall:** The model achieved a recall of **0.935**, indicating a very low rate of false negatives—crucial in a medical diagnostic context.
* **Inference Speed:** Total processing time (preprocess to postprocess) is approximately **45.2ms**, enabling rapid screening of large MRI batches.

---
