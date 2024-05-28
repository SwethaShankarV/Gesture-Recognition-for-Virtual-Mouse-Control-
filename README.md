**Gesture Recognition System for Virtual Mouse Control**

Welcome to the Gesture Recognition System for Virtual Mouse Control project repository. This project aims to develop a graphical-based AI-controlled mouse pointer using hand gesture recognition with the help of image processing and computer vision.
Table of Contents
1. Introduction
2. Project Structure
3. Setup and Installation
4. Usage
5. Contributing
6. License
7. Acknowledgements

**Introduction**

The objective of this project is to create a vision-based system that controls various mouse activities using hand gestures, making the interaction more efficient and reliable. The system uses a web camera to capture hand gestures and processes these gestures using MediaPipe and OpenCV frameworks.

**Project Structure**

The repository contains the following files and directories:
* HandTracker.py # Script for hand tracking and gesture detection  
* Main.py # Main application script  
* Project_Report.pdf # Detailed project report  
* README.md # This readme file 

**Setup and Installation**

To run this project, ensure you have the following dependencies installed:
* Python 3.7+
* OpenCV
* MediaPipe

You can install the required Python packages using pip:
pip install opencv-python mediapipe numpy 

**Usage**

1. HandTracker.py: This script contains the implementation for detecting and tracking hand landmarks using MediaPipe.
2. Main.py: This script runs the main application, which captures video from the webcam and performs mouse control based on detected hand gestures.
To run the main application, use the following command:
python Main.py
Ensure your webcam is connected and properly set up.

