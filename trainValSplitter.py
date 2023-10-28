import os
import shutil
import numpy as np

preprocessingPath="stitch-hsv"
path = "data"
classes = os.listdir(preprocessingPath)
trainPath = os.path.join(path,"train")
valPath = os.path.join(path,"val")
train = 0.80 # ;val = 1. - train
np.random.seed(26)

if os.path.exists(path): shutil.rmtree(path); os.mkdir(path)
else: os.mkdir(path)

for s in [trainPath, valPath]:
    if os.path.exists(s): shutil.rmtree(s); os.mkdir(s)
    else: os.mkdir(s)
    for c in classes:
        classPath = os.path.join(s,c)
        if os.path.exists(classPath): shutil.rmtree(classPath); os.mkdir(classPath)
        else: os.mkdir(classPath)

## SEPARATING TO TRAIN AND VALIDATION SETS
for c in classes:
    classPath = os.path.join(preprocessingPath,c)
    images = np.array(os.listdir(classPath))
    np.random.shuffle(images)
    numImages = len(images)
    setClassPath = os.path.join(trainPath,c)
    for n,image in enumerate(images):
        if n == round(numImages*train): 
            setClassPath = os.path.join(valPath,c)
        imagePath = os.path.join(classPath,image)
        newImagePath = os.path.join(setClassPath,image)
        shutil.copy(imagePath, newImagePath)