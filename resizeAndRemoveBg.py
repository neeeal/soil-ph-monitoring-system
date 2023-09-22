import os
import numpy as np
import cv2
import shutil
from PIL import Image
from rembg import remove

## Variable Declerations
preprocessingPath="preprocessingData"
classes = os.listdir(preprocessingPath)
path = "data"
input_shape = (224,224) # changed from 224x224
train = 0.80
val = 1. - train
trainPath = os.path.join(path,"train")
valPath = os.path.join(path,"val")
paths = [trainPath, valPath]



## REMOVING BACKGROUND AND RETAINING SOIL
## RESIZING IMAGES FROM 416x416 to 224x224
## src: https://www.geeksforgeeks.org/how-to-remove-the-background-from-an-image-using-python/
  

# # CHECKS IF DATA DIRECTORIES EXISTS
# # DELETE IF SO, CREATE OTHERWISE
# for c in classes:
#     classPath = os.path.join(preprocessingPath,c)
#     if os.path.exists(classPath): shutil.rmtree(classPath); os.mkdir(classPath)
#     else: os.mkdir(classPath)

for c in classes:
    classPath = os.path.join(preprocessingPath,c)
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
        output_path = imagePath.replace(preprocessingPath,preprocessingPath)
        output.save(output_path)