#Script to detect if license plates are Russian from input images comparing against a cascade classifier
#Created by L. Denomme for Foundations of Comp. Vision (CSC515), graduate class at Colorado State Univ. - Global Campus
#Last Update: 10/7/24


import numpy as np

import cv2

#from sys import argv

#Load the image
img = cv2.imread('russianplates_1.png')

#Step 1: Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('Step 1_Grayscale.png', gray)

#Step 2: Equalize histogram for better contrast
gray_img = cv2.equalizeHist(gray)
cv2.imwrite('Step 2_Equalize Histogram.png', gray_img)

#Step 3: preprocess with a Gaussian Blur, 3x3 kernel
blur = cv2.GaussianBlur(gray_img, (3,3), 2)
cv2.imwrite('Step 3_Gaussian Blur.png', blur)

#Load the pretrained classifier
pretrained = 'haarcascade_russian_plate_number.xml'
licenseplate_cascade = cv2.CascadeClassifier(pretrained)

#Check if the classifier was loaded correctly:
if licenseplate_cascade.empty():
    print('Error loading cascade classifier')

#detect the license plate using the preprocessed image
license_plate = licenseplate_cascade.detectMultiScale(blur, scaleFactor = 1.1, minNeighbors = 5, minSize = (30,30), maxSize = (300, 300))

#draw rectangles around the plates
for (x, y, w, h) in license_plate:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('License Plate Detected', img)

cv2.waitKey(0)

cv2.destroyAllWindows()
