# SmartProtectAI 🛡️  
### Real-Time Fall Detection & Scream Detection using Machine Learning  

A real-time emergency detection system combining motion-based fall detection and audio-based scream detection using Machine Learning.

SmartProtectAI is a dual-mode safety monitoring system that detects:

- **Falls** using motion sensor data  
- **Screams** using audio pattern recognition  

The system is built to support elderly care, personal safety, and emergency response scenarios.  
It can run in **real-time**, trigger alerts, and be integrated with IoT, phones, or wearable devices.

---

## 🚀 Features

### 🔹 **1. Fall Detection**
- Uses accelerometer patterns  
- Extracts dynamic motion features  
- Trained on the **SisFall dataset**
- Multi-stage model tuning and scaling  
- Outputs real-time fall alerts

### 🔹 **2. Scream Detection**
- Audio-based emergency detection  
- Uses MFCC features  
- Machine learning classifier trained to detect distress sounds  
- Runs live through default microphone  

### 🔹 **3. Real-Time Monitoring**
- Continuously listens for scream events via microphone  
- Continuously monitors sensor activity for falls  
- Lightweight & efficient

---

## 📁 Project Structure

SmartProtectAI/
│
├── fall_dataset_build.py                        # Preprocess SisFall dataset
├── fall_model.py                                # Train + evaluate fall detection model
├── fall_model_tune.py                           # Hyperparameter tuning for fall model
├── final_model.py                               # Combined fall + scream detection logic
├── scream_detector.py                           # Train + test scream detection model
├── realtime_test.py                             # Real-time scream detection script
├── train_model.py                               # Central training script (fall detection)
├── requirements.txt                             # Project dependencies
└── .gitattributes                               # Git attributes


---

## 🧠 How It Works

### 🟩 **Fall Detection Workflow**
1. Raw accelerometer signals are loaded  
2. Features extracted (variance, magnitude, energy…)  
3. Data labeled as *fall* or *non-fall*  
4. Model trained using Random Forest / ML classifier  
5. Final tuned model exported  

### 🟦 **Scream Detection Workflow**
1. Audio captured using microphone  
2. MFCC features computed  
3. Classifier predicts *scream* or *normal audio*  
4. Alerts triggered on detection  

---

## 📦 Installation

```bash
git clone https://github.com/Dev-Aloo/SmartProtectAI.git
cd SmartProtectAI
pip install -r requirements.txt
```

---

## ▶️ Run Real-Time Detection

### 🔊 Scream Detection
```bash
python realtime_test.py
```

### 🛡️ Full System (Fall + Scream)
```bash
python final_model.py
```

---

## ⚠️ Dataset Note

The **SisFall dataset** is not included in this repository due to size limitations.

Download it from:  
https://sites.google.com/site/sisfalldataset/

Place it in:

```
SmartProtectAI/SisFall_dataset/
```

---

## 👤 Author

**Devraj Singh**  
Computer Science • Data Science • Machine Learning  
GitHub: https://github.com/Dev-Aloo

---


