import os
import shutil

## REPLACE FILE EXTENTION AND MOVE DATA
path = "jpgData"
dirs = [os.path.join(path,dir) for dir in os.listdir(path)]
classes = os.listdir(path)

for c in classes:
    classPath = os.path.join("pngData",c)
    if os.path.exists(classPath): shutil.rmtree(classPath); os.mkdir(classPath)
    else: os.mkdir(classPath)

oldImages = []
for dir in dirs:
    for image in os.listdir(dir):
        oldImages.append(os.path.join(dir,image))

for image in oldImages:
    shutil.copy(image, image.replace(path,'pngData',1).replace("jpg","png"))