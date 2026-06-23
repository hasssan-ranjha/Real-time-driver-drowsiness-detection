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

## 🚀 Key Features
* **Hybrid Vision Pipeline:** Combined classical object localization with modern deep learning state classification.
* **Lag-Free Engine:** Multi-stage performance optimization maintaining a smooth **28-30 FPS** on CPU without requiring an external GPU.
* **Noise Filtering:** Custom algorithmic "hacks" designed to prevent false positives from facial geometry (nostrils, eyebrows).
* **Native Local Execution:** Fully compiled standalone Windows executable (`.exe`) requiring zero installation dependencies.

---

## 🛠️ Tech Stack & Toolkit
* **Core Language:** Python 3.x
* **Deep Learning Framework:** TensorFlow / Keras (MobileNetV2 backbone)
* **Computer Vision:** OpenCV (`cv2`)
* **Audio Alerts:** Pygame Mixer (for asynchronous thread-safe sound rendering)
* **Packaging & Tooling:** PyInstaller, PowerShell

---

## 🧠 System Pipeline Architecture
The application runs as an efficient, deterministic pipeline operating on a frame-by-frame loop:



1. **Frame Capture & Downsampling:** Captures standard video stream and downsamples it to `640x480` to drastically reduce CPU cycle overhead.
2. **Face Tracking (Localization):** Employs Haar Cascade Classifiers (`haarcascade_frontalface_default.xml`) to establish regional facial bounds.
3. **Vertical ROI Slicing:** Discards the bottom 50% of the facial frame to optimize downstream compute cycles.
4. **Eye Tracking:** Isolates ocular bounds using `haarcascade_eye.xml` inside the validated upper face region.
5. **Tight-Crop Filtering:** Crops out the top 20% and bottom 20% of the bounding eye box to slice away eyebrow and cheek artifact noise.
6. **CNN Binary Inference:** Resizes the tight crop to `224x224x3` (RGB space) and feeds it to **MobileNetV2** to calculate a closed-eye probability score between `0.0` (Awake) and `1.0` (Sleepy).
7. **Temporal State Machine:** Evaluates a threshold (`Score > 0.5`). If consecutive sleepy frames breach `10 frames (~1/3 of a sec)`, a thread-safe background buzzer triggers.

---

## 🛑 Technical Hurdles & Engineering Troubleshooting

Developing an embedded vision system for real-time safety scenarios exposed several constraints. Below are the key failures faced during development and how they were resolved:

### 1. The Processor Bottleneck (Severe Video Lag)
* **The Failure:** Initially, running continuous 1080p frame inferences through the CNN entirely choked the CPU, introducing an unacceptable 1-2 second lag behind real-time.
* **The Fix:** Implemented an optimized **"Thinking vs. Drawing"** loop. Video downsampling reduced standard pixel volume, and the underlying thread structure was re-designed so that heavy AI inference calculations execute exactly every **4th frame**, while the visual UI layer continues rendering bounding boxes seamlessly at 30 FPS.

### 2. The Nostril Glitch (False Detections)
* **The Failure:** In variable or low-light situations, the Haar Cascade eye classifier would repeatedly mistake the dark cavities of the nostrils for eyes, corrupting the drowsiness metrics.
* **The Fix:** Engineered a strict **Vertical Region of Interest (ROI) Slice**. The software slices the bounding face box horizontally and ignores anything below the geometric midpoint, completely blinding the eye classifier to nose or mouth regions.

### 3. Eyebrow/Cheek Inference Noise
* **The Failure:** The model often struggled to spot closed eyes because standard bounding boxes included eyebrows. The AI learned to associate eyebrows with an "awake" state, completely skewing the decision boundaries when eyelids closed.
* **The Fix:** Programmed a dynamic **"Tight-Crop" scaling filter**. The system mathematically slices off the top 20% (eyebrows) and bottom 20% (cheeks) of the localized eye coordinates before formatting the image for tensor translation, forcing the network to exclusively evaluate eyelid states.

---

## 📦 Deployment Journey: local over cloud
Originally, deployment was attempted as an interactive cloud web application via **Streamlit Cloud and WebRTC**. However, network telemetry testing exposed critical problems:
* **Latency:** Routing raw, compressed frame packets up to a cloud server and awaiting predictions created a 2+ second roundtrip delay, which is entirely unviable for a life-safety mechanism.
* **Infrastructure Overhead:** Dealing with STUN/TURN server signaling over inconsistent network conditions proved inefficient for edge environments.

### The Solution:
To enforce instant, zero-lag edge execution, the project shifted to a **Local Deployment strategy**. Using PowerShell and PyInstaller, the complete Python code, environmental dependencies, Haar cascades, and weights (`drowsiness_model.h5`) were compiled into a completely portable, standalone Windows Executable (`.exe`). This delivers immediate processing directly on the driver's laptop with zero cloud requirements or runtime configuration.

---

## 📊 Performance & Accuracy
* **Validation Accuracy:** 92% accuracy achieved using custom transfer-learning adjustments on MobileNetV2.
* **Inference Speed:** ~28 - 30 FPS steady on commercial, non-GPU laptop architectures.
* **Reaction Window:** Less than **350ms** from continuous eye-closure to immediate audio buzz trigger.

## Example Images

The dataset includes both open and closed eye images. Below are examples:

![Eye Image Example](http://mrl.cs.vsb.cz/images/eyedataset/eyedataset01.png)

## References

- [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset)
- [Forked Dataset on Kaggle](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset)

