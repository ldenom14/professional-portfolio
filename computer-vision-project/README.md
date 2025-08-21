# Object Detection using Haar Cascade Classifiers

# Introduction
This project demonstrates the use of a Haar Cascade classifier for real-time object detection. The model, trained on images of Russian license plates, 
is a practical application of a foundational computer vision algorithm. This project showcases my proficiency in image processing, algorithm application, 
and building simple yet effective computer vision pipelines.

# Key Features
+ **Classic Computer Vision Method:** This project uses a pre-trained Haar Cascade classifier, a classic and efficient algorithm for object detection.
It demonstrates a solid understanding of fundamental computer vision techniques.
+ **Image Preprocessing:** The script includes essential preprocessing steps to prepare the input images for the classifier, such as grayscale conversion,
histogram equalization, and Gaussian blurring, which are critical for improving detection accuracy.
+ **Object Detection Pipeline:** The project showcases a complete object detection workflow, from loading an image and preprocessing it to detecting the
target object and highlighting it with a bounding box.

**Technologies Used:** Python, OpenCV

# How It Works
+ **1. Load Image:** The script first loads an input image.
+ **2. Preprocessing:** The image is preprocessed to make it easier for the classifier to detect the target object. This involves converting the image to grayscale, equalizing the histogram for better contrast, and applying a Gaussian blur to reduce noise.
+ **3. Classifier Application:** The preprocessed image is passed to the haarcascade_russian_plate_number.xml classifier, which scans the image for patterns that match the features it was trained to detect.
+ **4. Output - Detection & Bounding Box:** When a license plate is detected, the script draws a green bounding box around it and saves the output image.

# How To Run
+ Clone this repository or download the project files.
+ Ensure you have Python and OpenCV installed. If not, you can install OpenCV via pip:
  pip install opencv-python
+ Place the haarcascade_russian_plate_number.xml file and your input image (e.g., russianplates_1.png) in the same folder as the script.
+ Run the script from your terminal: python russianplatedetector.py

