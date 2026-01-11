# AI Gesture Control System

An AI-powered real-time hand gesture control system that allows users to control their computer using only hand movements captured by a webcam.

This project uses **OpenCV**, **MediaPipe**, and **PyAutoGUI** to track hand landmarks and convert gestures into system actions.

---

##  Features

- 🖱️ Mouse movement using index finger  
- 👆 Left click (index + thumb)  
- 👉 Right click (index + middle finger)  
- 🔊 Volume control (two fingers swipe left/right)  
- 💡 Brightness control (index finger swipe up/down)  
- 📜 Scroll (three fingers up)  
- ⏸️ Pause & Resume gestures  
- ⚡ Real-time performance with FPS display  

---

##  Technologies Used

- Python 3.11  
- OpenCV  
- MediaPipe  
- PyAutoGUI  
- NumPy  
- Screen Brightness Control  

---

##  How to Run

```bash
pip install opencv-python pyautogui numpy screen-brightness-control mediapipe
python main.py

 Controls
Gesture	Action
Index finger	Move mouse
Index + Thumb	Left click
Index + Middle	Right click
Two fingers swipe	Volume
Index swipe	Brightness
Three fingers	Scroll
p key	Pause
q key	Quit
 
