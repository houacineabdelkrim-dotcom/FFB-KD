Here is the raw Markdown code for your `README.md` file. You can copy and paste this directly into a new file on GitHub.

```markdown
# FFB-KD -IDS: Edge-Optimized Intrusion Detection for IoT

This repository provides a lightweight Intrusion Detection System (IDS) framework developed using **Fixed-Point Feature-Based Knowledge Distillation FFB-KD **. The project focuses on bridging the gap between sophisticated machine intelligence and the extreme resource constraints of 8-bit IoT hardware.

## 🎯 Project Objective
The goal is to deploy an effective security layer on the **MSP430-based Z1 Mote**, which is limited by a strict **8 KB RAM** envelope. Our methodology compresses the detection model to a **5.28 KB** footprint, ensuring it can operate alongside the **Contiki-NG/RPL** network stack.

## 🚀 Key Performance
* **Model Size:** 5.28 KB (INT8 Quantized)
* **Target Hardware:** Z1 Mote / MSP430 Architecture
* **RAM Headroom:** ~2.7 KB remaining for System/Network tasks

---

## 📁 Repository Structure

The implementation is divided into three functional Python scripts:

| File | Purpose |
| :--- | :--- |
| `dataset_clean.py` | Initial data parsing, feature engineering, and normalization of IoT traffic. |
| `dataset_clean_smote.py` | Implements SMOTE to balance attack classes, ensuring robust detection of rare threats. |
| `train_qakd.py` | The core engine that performs Knowledge Distillation (Teacher-to-Student) and INT8 Quantization. |

---

## 📊 Methodology Overview

### 1. Memory Profiling
The framework is designed to fit within the primary memory of the Z1 Mote. 


### 2. QAKD Workflow
We transfer high-tier intelligence from an FP32 Teacher to an optimized INT8 Student model.


---

## 🛠️ Getting Started

### Prerequisites
* Python 3.8+
* Scikit-learn
* Imbalanced-learn (for SMOTE)
* PyTorch or TensorFlow (depending on your `train_qakd.py` implementation)

### Execution Flow
1. **Pre-process the data:**
   ```bash
   python dataset_clean.py

```

2. **Balance the classes:**
```bash
python dataset_clean_smote.py

```


3. **Train and Distill:**
```bash
python train_FFB-KD.py

```



---

## 📈 Evaluation Results

The following table summarizes the feasibility of our approach compared to a standard uncompressed model:

| Metric | Baseline (Teacher) | QAKD Model (Ours) |
| --- | --- | --- |
| Precision | 32-bit Floating Point | 8-bit Integer |
| RAM Usage | ~120 KB | **5.26 KB** |


---

## 📜 Citation

If you use this work in your research, please cite our study:

> *Deep Feature Alignment and Quantized Distillation for Resilient Edge Intrusion Detection in IoT Networks.2026*

