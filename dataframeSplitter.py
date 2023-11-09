import os
import numpy as np
import csv

preprocessingPath="close_seg"
path = "data"
classes = os.listdir(preprocessingPath)
trainPath = "df//train_df.csv"
valPath = "df//val_df.csv"
train = 0.80 # ;val = 1. - train

np.random.seed(26)

## Setting file headers
header = ["path","pH"]
trainFile= open (trainPath, "w")
trainWriter = csv.writer(trainFile)
trainWriter.writerow(header)
valFile=open (valPath, "w")
valWriter = csv.writer(valFile)
valWriter.writerow(header)

## SEPARATING TO TRAIN AND VALIDATION SETS
for c in classes:
    classPath = os.path.join(preprocessingPath,c)
    images = np.array(os.listdir(classPath))
    np.random.shuffle(images)
    numImages = len(images)
    for n,image in enumerate(images):
        imagePath = os.path.join(classPath,image)
        if n < round(160*train): 
            trainWriter.writerow([imagePath,int(c)])
        else:
            valWriter.writerow([imagePath,int(c)])
        if n >= 160: break
trainFile.close()
valFile.close()