import os
import shutil
import cv2
import numpy as np

## apply watershed and convert to hsv
path = "jpgData"
dirs = [os.path.join(path,dir) for dir in os.listdir(path)]
classes = os.listdir(path)
newPath = 'step-step'

for c in classes:
    classPath = os.path.join(newPath,c)
    if os.path.exists(classPath): shutil.rmtree(classPath); os.mkdir(classPath)
    else: os.mkdir(classPath)

oldImages = []
for dir in dirs:
    for image in os.listdir(dir):
        oldImages.append(os.path.join(dir,image))

for image in oldImages:
    image2 = cv2.imread(image)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    cropped_image_2 = image2.copy()[140:268,140:268,:]
    rgb_planes = cv2.split(cropped_image_2)
    result_planes,result_norm_planes = [], []
    # create a CLAHE object (Arguments are optional).
    # clahe = cv2.createCLAHE(tileGridSize = (16,16), clipLimit = 3.0)
    for plane in rgb_planes:
        processed_image = cv2.dilate(plane, (7, 7), 20) 
        # processed_image = clahe.apply(processed_image)
        processed_image= cv2.medianBlur(processed_image,5)
        norm_processed_image = cv2.normalize(processed_image,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(processed_image)
        result_norm_planes.append(norm_processed_image)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    # fig, ax = plt.subplots(1,3,figsize=(15,4))
    # ax[0].imshow(cropped_image_2)
    # ax[0].set_title("Original Cropped Image")
    # ax[1].imshow(result)
    # ax[1].set_title("Processed Image")
    # ax[2].set_title("Normalized Image")
    # ax[2].imshow(result_norm)
    # fig.suptitle("RGB Image")
    # plt.show()

    result_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    # fig, ax = plt.subplots(1,3, figsize=(15,4))
    # ax[0].imshow(result_hsv[:,:,0]) #hue
    # ax[0].set_title("HUE")
    # ax[1].imshow(result_hsv[:,:,1]) #saturation
    # ax[1].set_title("SATURATION")
    # ax[2].imshow(result_hsv[:,:,2]) #value
    # ax[2].set_title("VALUE")
    # fig.suptitle("HSV Image")
    # plt.show()

    # fig, ax = plt.subplots(1,3, figsize=(15,4))
    # result_planes=[]
    # for plane in rgb_planes:
    #     # for n,row in enumerate(plane):
    #     #     for m,col in enumerate(row):
    #     #         if plane[n,m] <= result_hsv[n,m,1]: plane[n,m] = 0
    #     #         else: plane[n,m] = plane[n,m] - result_hsv[n,m,1]
    #     plane = plane * (result_hsv[:,:,1] > 64)
    #     result_planes.append(plane)

    # saturation_clean = result_hsv[:,:,1] * (result_hsv[:,:,1] > 64) 
    # ax[0].imshow(saturation_clean)
    # ax[0].set_title("Saturation with < 64 = 0")
    # fig.suptitle("Final Image")

    # ax[1].imshow(result_hsv[:,:,1])
    # ax[1].imshow(cropped_image_2, alpha=0.5)
    # ax[1].set_title("overlay saturation with with original image")

    # minus = cv2.merge(result_planes)  
    # # ax[2].imshow(minus)
    # # ax[2].set_title("overlay saturation with with original image")

    # plt.show()
    cv2.imwrite(image.replace(path,newPath,1),result_hsv[:,:,1])

