

# 🚦 Lane Flow Prediction using Vision-Based Spatio-Temporal Modeling

## 📌 Overview

This project implements an **AI-powered Lane Flow Prediction & Dynamic Traffic Signal Control System** that analyzes real-time traffic video feeds to estimate lane-wise vehicle density and predict near-future traffic flow. The system dynamically allocates green signal timings to optimize intersection throughput and reduce congestion.

The solution combines **Computer Vision**, **Spatio-Temporal Learning**, and **Traffic Optimization** techniques, making it suitable for **smart cities**, **adaptive traffic signals**, and **urban traffic analytics**.

---

## 🎯 Key Objectives

* Detect vehicles and pedestrians from traffic camera feeds
* Estimate **lane-wise traffic density** in real time
* Predict **short-term traffic flow** using temporal patterns
* Dynamically allocate green signal time per lane
* Provide a **real-time dashboard** for monitoring and analysis

---

## 🧠 System Architecture

```
Video Feeds (CCTV)
        ↓
Object Detection (YOLOv8 / Detectron2)
        ↓
Lane-wise Tracking & Counting (DeepSORT)
        ↓
Density Estimation (SVD / Vehicle Count / Occupancy)
        ↓
Spatio-Temporal Transformer
        ↓
Lane Flow Prediction
        ↓
Signal Time Optimization
        ↓
Real-time Dashboard (Streamlit / Dash)
```

---

## 🛠️ Tech Stack

### Computer Vision

* YOLOv8 (Vehicle & Pedestrian Detection)
* Detectron2 (Optional alternative)
* OpenCV
* DeepSORT (Multi-object Tracking)

### Machine Learning / Deep Learning

* PyTorch
* Spatio-Temporal Transformer
* Temporal Buffers & Sliding Windows
* SVD-based Density Estimation

### Backend & Visualization

* Python 3.10
* Streamlit / Dash
* NumPy, Pandas
* Matplotlib / Plotly




## 🔍 Core Modules Explained

### 1. Vehicle Detection

* Uses YOLOv8 to detect cars, bikes, buses, trucks, and pedestrians
* Outputs bounding boxes with confidence scores

### 2. Lane-wise Tracking

* DeepSORT assigns persistent IDs
* Vehicles are mapped to predefined lane regions

### 3. Density Estimation

* Vehicle count per lane
* Occupancy-based metrics
* **SVD-based density smoothing** to reduce noise

### 4. Spatio-Temporal Prediction

* Temporal window captures historical lane states
* Transformer learns inter-lane and temporal dependencies
* Predicts short-term lane congestion

### 5. Signal Optimization

* Green time allocated proportional to predicted density
* Minimum and maximum constraints applied
* Prevents starvation of low-density lanes

### 6. Dashboard

* Live vehicle count per lane
* Density heatmaps
* Signal timer visualization
* Prediction graphs

---

## 📊 Example Use Cases

* Smart traffic signal control
* Urban traffic congestion analysis
* Peak-hour traffic optimization
* Emergency vehicle prioritization (extensible)
* Smart city traffic dashboards

---

## 🚀 How to Run

```bash
# Clone repository
git clone https://github.com/your-username/lane-flow-prediction.git
cd lane-flow-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run main orchestrator
python main/main.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## 📈 Results & Impact

* Reduced average waiting time per lane
* Improved intersection throughput
* Real-time adaptability to traffic surges
* Scalable to multi-intersection setups

---

## 🔮 Future Enhancements

* Reinforcement Learning for signal control
* Multi-intersection coordination
* Weather-aware traffic modeling
* Edge deployment on NVIDIA Jetson
* Privacy-preserving analytics

---

## 👨‍💻 Author

**Kovid Bidada**
B.E. Artificial Intelligence & Machine Learning
Osmania University

---

## 📜 License

This project is licensed under the MIT License.

---

> *“Turning traffic data into intelligent flow.”* 🚦
