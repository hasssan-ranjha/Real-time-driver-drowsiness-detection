# Real-Time AI-Based Driver Drowsiness Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg?style=flat&logo=keras)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org)
[![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-lightgrey.svg?style=flat&logo=google)](https://arxiv.org/abs/1801.04381)
[![Pygame](https://img.shields.io/badge/UI/Audio-Pygame-yellowgreen.svg?style=flat)](https://www.pygame.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)

A non-intrusive, real-time Computer Vision and Deep Learning pipeline engineered to monitor a driver’s physiological state and instantly trigger an auditory alarm upon identifying signs of fatigue or micro-sleep. 

This project bridges the gap between complex Deep Learning theory and real-world edge deployment by optimizing a heavy neural network pipeline to run efficiently on standard, consumer-grade laptop hardware.

---

##  Key Features
* **Hybrid Vision Pipeline:** Combined classical object localization with modern deep learning state classification.
* **Lag-Free Engine:** Multi-stage performance optimization maintaining a smooth **28-30 FPS** on CPU without requiring an external GPU.
* **Noise Filtering:** Custom algorithmic "hacks" designed to prevent false positives from facial geometry (nostril tracking avoidance, eyebrow exclusion).
* **Native Local Execution:** Fully compiled standalone Windows executable (`.exe`) included directly in the release files.

---

##  System Pipeline Architecture
The application runs as an efficient, deterministic pipeline operating on a frame-by-frame loop:

[Frame Capture] ➡️ [Downscale 640x480] ➡️ [Face Tracking] ➡️ [Vertical ROI Slicing]
⬇️
[Audio Alert] ⬅️ [Temporal State Machine] ⬅️ [CNN Binary Inference] ⬅️ [Tight-Crop Eye ROI]

1. **Frame Capture & Downsampling:** Captures standard video stream and downsamples it to `640x480` to drastically reduce CPU cycle overhead.
2. **Face Tracking (Localization):** Employs Haar Cascade Classifiers (`haarcascade_frontalface_default.xml`) to establish regional facial bounds.
3. **Vertical ROI Slicing:** Discards the bottom 50% of the facial frame to optimize downstream compute cycles and eliminate mouth/nostril noise.
4. **Eye Tracking:** Isolates ocular bounds using `haarcascade_eye.xml` inside the validated upper face region.
5. **Tight-Crop Filtering:** Crops out the top 20% and bottom 20% of the bounding eye box to slice away eyebrow and cheek artifact noise.
6. **CNN Binary Inference:** Resizes the tight crop to `224x224x3` (RGB space) and feeds it to **MobileNetV2** to calculate a closed-eye probability score between `0.0` (Awake) and `1.0` (Sleepy).
7. **Temporal State Machine:** Evaluates a threshold (`Score > 0.5`). If consecutive sleepy frames breach `10 frames (~1/3 of a sec)`, a thread-safe background buzzer triggers.

---

##  Model Training & Evaluation

The classification engine relies on a Transfer Learning approach built over a pre-trained **MobileNetV2** architecture. The final classification dense layers were custom-trained over a curated dataset of open and closed eyes using binary cross-entropy loss.

###  Training History (Loss & Accuracy Plots)
The convergence profiles below demonstrate stable training across 5 targeted fine-tuning epochs, capturing optimal performance before hitting diminishing returns or overfitting thresholds:

| Training & Validation Accuracy | Training & Validation Loss |
| :---: | :---: |
| ![Accuracy Plot](images/accuracy_plot.png) | ![Loss Plot](images/loss_plot.png) |


<p align="center">
  <img src="images/training_metrics.png" alt="Training Performance History Matrix" width="80%">
</p>

###  Training Validation Proof (5-Epoch Execution Log)
Below is the verified stdout execution trace confirming successful network convergence, loss minimization, and target accuracy verification through the 5 training cycles:

<p align="center">
  <img src="images/training_proof_screenshot.png" alt="Proof of Model Training Logs over 5 Epochs" width="90%">
</p>

---

##  Technical Hurdles & Engineering Troubleshooting

### 1. The Processor Bottleneck (Severe Video Lag)
* **The Failure:** Running continuous 1080p frame inferences through the CNN choked the CPU, introducing an unacceptable 1-2 second delay behind real-time.
* **The Fix:** Implemented an optimized **"Thinking vs. Drawing"** loop. Video downsampling reduced standard pixel volume, and the underlying thread structure was re-designed so that heavy AI inference calculations execute exactly every **4th frame**, while the visual UI layer continues rendering bounding boxes seamlessly at 30 FPS.

### 2. The Nostril Glitch (False Detections)
* **The Failure:** In variable or low-light situations, the Haar Cascade eye classifier would repeatedly mistake the dark cavities of the nostrils for eyes, corrupting the drowsiness metrics.
* **The Fix:** Engineered a strict **Vertical Region of Interest (ROI) Slice**. The software slices the bounding face box horizontally and ignores anything below the geometric midpoint, completely blinding the eye classifier to nose or mouth regions.

### 3. Eyebrow/Cheek Inference Noise
* **The Failure:** The model struggled to spot closed eyes because standard bounding eye boxes included eyebrows. The AI learned to associate eyebrows with an "awake" state, completely skewing the decision boundaries when eyelids closed.
* **The Fix:** Programmed a dynamic **"Tight-Crop" scaling filter**. The system mathematically slices off the top 20% (eyebrows) and bottom 20% (cheeks) of the localized eye coordinates before formatting the image for tensor translation, forcing the network to exclusively evaluate eyelid states.

---

##  Deployment Journey: Local over Cloud
Originally, deployment was attempted as an interactive cloud web application via **Hugging Face Spaces / Streamlit Cloud and WebRTC**. However, cloud-telemetry testing exposed structural bottlenecks:
* **Latency:** Routing raw, compressed frame packets up to a remote cloud server and awaiting predictions created a 2+ second roundtrip delay, which is entirely unviable for a life-safety mechanism.
* **Infrastructure Overhead:** Managing reliable STUN/TURN server signaling over inconsistent consumer network conditions proved highly inefficient for real-time edge environments.

### The Application Executable Solution:
To enforce instant, zero-lag edge execution, the project shifted to a **Local Deployment strategy**. Using PowerShell and the PyInstaller utility, the complete Python code, environment dependencies, Haar cascades, and weights (`drowsiness_model.h5`) were compiled into a completely portable, standalone Windows Executable (`.exe`) included directly in this repository.

---

##  Installation & How to Run Local Application (.exe)

You do **not** need Python, TensorFlow, or OpenCV installed on your system to run the app. 

1. Head over to the root folder or the **Releases** section of this GitHub repository.
2. Download the packaged application folder containing `DrowsinessDetector.exe`.
3. Launch your command prompt or PowerShell, navigate to the folder, and execute:
   ```bash
   ./DrowsinessDetector.exe
## References

- [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset)
- [Forked Dataset on Kaggle](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset)

