import os
import numpy as np
import cv2
import shutil
from PIL import Image
from rembg import remove

## Variable Declerations
dataPath = "pngData"
noBgDataPath="noBgData"
classes = os.listdir(dataPath)
path = "data"
input_shape = (224,224)
train = 0.80
val = 1. - train
trainPath = os.path.join(path,"train")
valPath = os.path.join(path,"val")
PATHS = [trainPath, valPath]



## REMOVING BACKGROUND AND RETAINING SOIL
## RESIZING IMAGES FROM 416x416 to 224x224
## src: https://www.geeksforgeeks.org/how-to-remove-the-background-from-an-image-using-python/
  

# CHECKS IF DATA DIRECTORIES EXISTS
# DELETE IF SO, CREATE OTHERWISE
for c in classes:
    classPath = os.path.join(noBgDataPath,c)
    if os.path.exists(classPath): shutil.rmtree(classPath); os.mkdir(classPath)
    else: os.mkdir(classPath)

for c in classes:
    classPath = os.path.join(dataPath,c)
    images = os.listdir(classPath)
    for image in images:
        imagePath = os.path.join(classPath,image)
  
        ## RESIZING IMAGE
        origImage = cv2.imread(imagePath)
        resizedImage = cv2.resize(origImage, input_shape, interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(imagePath,resizedImage)

        # Processing the image
        input = Image.open(imagePath)
        
        # Removing the background from the given Image
        output = remove(input)
        
        #Saving the image in the given path
        output_path = imagePath.replace(dataPath,noBgDataPath)
        output.save(output_path)


## CHECKING IF TRAIN AND VAL PATHS EXISTS
## IF SO, DELETE, IF NOT THEN CREATE
for path in PATHS:
    pass
    if os.path.exists(path): 
        shutil.rmtree(path)
        os.mkdir(path)
        for c in classes:
            classPath = os.path.join(path,c)
            os.mkdir(classPath)
    else: 
        os.mkdir(path)
        for c in classes:
            classPath = os.path.join(path,c)
            os.mkdir(classPath)

## SEPARATING TO TRAIN AND VALIDATION SETS
for c in classes:
    classPath = os.path.join(noBgDataPath,c)
    images = os.listdir(classPath)
    numImages = len(images)
    setClassPath = os.path.join(trainPath,c)
    for n,image in enumerate(images):
        if n == round(numImages*train): 
            setClassPath = os.path.join(valPath,c)
        imagePath = os.path.join(classPath,image)
        newImagePath = os.path.join(setClassPath,image)
        shutil.copy(imagePath, newImagePath)