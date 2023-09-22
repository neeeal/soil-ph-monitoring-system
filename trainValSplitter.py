import os
import shutil
import numpy as np

preprocessingPath="preprocessingData"
path = "data"
classes = os.listdir(preprocessingPath)
trainPath = os.path.join(path,"train")
valPath = os.path.join(path,"val")
train = 0.80 # ;val = 1. - train
np.random.seed(26)

## SEPARATING TO TRAIN AND VALIDATION SETS
for c in classes:
    classPath = os.path.join(preprocessingPath,c)
    images = np.array(os.listdir(classPath))
    images = np.random.shuffle(images)
    numImages = len(images)
    setClassPath = os.path.join(trainPath,c)
    for n,image in enumerate(images):
        if n == round(numImages*train): 
            setClassPath = os.path.join(valPath,c)
        imagePath = os.path.join(classPath,image)
        newImagePath = os.path.join(setClassPath,image)
        shutil.copy(imagePath, newImagePath)