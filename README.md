# Finger Counter Project 👋
A high-accuracy real-time finger counting system using computer vision with OpenCV and Python.
### Features ✨
- 🖐️ Real-time finger detection from webcam feed

- 🎯 Advanced contour analysis for precise counting

- 🔍 Sophisticated convexity defect detection

- 📊 Multiple accuracy enhancement techniques

- 🖼️ Visual feedback with contours and convex hull

- ⚙️ Configurable parameters for different environments

### Requirements 🛠️
- Python 3.6+
- OpenCV (opencv-python)
- NumPy
### Installation ⚙️
Clone this repository:
```
bash
git clone https://github.com/your-username/finger-counter.git
cd finger-counter
```
Install dependencies:
```
bash
pip install -r requirements.txt
```
### Usage 🚀
Run the application:
```
bash
python finger_counter.py
```
- Calibration Phase (first 120 frames):
- Keep your hand out of the red rectangle
- Wait until calibration reaches 100%
- Detection Phase:
- Place your hand inside the red rectangle
- Keep fingers slightly spread
- View the finger count in the top-left corner
