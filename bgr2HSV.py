import os
import cv2
import matplotlib.pyplot as plt

preprocessingPath="preprocessingData"
secondStepPreprocessing = "secondStep_preprocessingData"
classes = os.listdir(preprocessingPath)

## CHANGING RGB IMAGE TO HSV AND SAVING
for c in classes:
    classPath = os.path.join(preprocessingPath,c)
    for image in os.listdir(classPath):
        imagePath = os.path.join(classPath,image)
        image = cv2.imread(imagePath)
        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        secondImagePath = imagePath.replace(preprocessingPath, secondStepPreprocessing)
        cv2.imwrite(secondImagePath,hsvImage)
