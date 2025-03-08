# 🎵 Gesture-Based Volume Control

This project allows users to **control system volume using hand gestures**. By detecting the distance between the **thumb and index finger**, the system dynamically adjusts the volume. It uses **OpenCV, MediaPipe, and Pycaw** for real-time hand tracking and audio control.

## 🚀 Features

- ✋ **Real-time Hand Tracking** using MediaPipe  
- 🔊 **Volume Control** based on hand distance  
- 📊 **Logarithmic Scaling** for smoother control  
- 🎚️ **Visual Volume Bar** for feedback  
- 🎥 **Works with Any Webcam**  
- 🖥️ **Windows Audio System Integration**  

---

## 📌 How It Works

1. **Hand Detection**: Uses MediaPipe to detect the hand and identify key landmarks.  
2. **Distance Measurement**: Calculates the distance between **thumb (landmark 4)** and **index finger (landmark 8)**.  
3. **Volume Mapping**: Converts the measured distance to a volume level using Pycaw.  
4. **Display Feedback**: A **volume bar** visually represents the current volume level.  

---

## 🛠️ Installation  

### 1️⃣ Clone the Repository  

```bash
git clone https://github.com/bharathreddy18/Computer-Vision-Projects.git
cd "Gesture Volume Control"
```

### 2️⃣ Install Dependencies  

Ensure you have **Python 3.8+** installed, then run:

```bash
pip install -r requirements.txt
```

---

## 📄 Requirements (`requirements.txt`)

```
opencv-python
mediapipe
numpy
pycaw
comtypes
```

---

## ▶️ Usage

1️⃣ **Run the script**  

```bash
python GestureVolumeControl.py
```

2️⃣ **Control volume** by moving your **thumb and index finger** closer or farther apart.  
3️⃣ **Press `q` to exit** the program.  

---

## 🚗 Use in Cars  

Gesture-based volume control is now integrated into **modern car infotainment systems**. Instead of using physical buttons or touchscreens, drivers can **adjust volume with simple hand gestures**, improving **safety and convenience** while driving.

- **🚘 BMW Gesture Control**: Featured in **BMW iDrive**, allowing volume adjustments with a circular hand motion.  
- **🔊 Mercedes MBUX**: Uses **gesture recognition cameras** to detect driver commands.  
- **🎵 Tesla**: Exploring gesture-based UI for infotainment controls.  

This project demonstrates a **basic version** of this concept using a webcam!

---

## 📌 To-Do / Improvements  

- ✅ Improve hand tracking accuracy  
- ✅ Implement gesture-based **play/pause** control  
- 🔲 Integrate with **custom audio applications**  
- 🔲 Add **voice feedback** for volume changes  

---

## 📜 License  

This project is **open-source** under the **MIT License**.

---

## 🏷️ Hashtags  

`#ComputerVision #OpenCV #HandTracking #AI #MachineLearning #GestureControl #Python #DeepLearning #Automation #Innovation #HCI #Mediapipe #Pycaw #GestureRecognition`

