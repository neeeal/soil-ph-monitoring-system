import os
import shutil
import cv2
import numpy as np

## apply watershed and convert to hsv
path = "jpgData"
dirs = [os.path.join(path,dir) for dir in os.listdir(path)]
classes = os.listdir(path)
newPath = 'waterhsv'

for c in classes:
    classPath = os.path.join(newPath,c)
    if os.path.exists(classPath): shutil.rmtree(classPath); os.mkdir(classPath)
    else: os.mkdir(classPath)

oldImages = []
for dir in dirs:
    for image in os.listdir(dir):
        oldImages.append(os.path.join(dir,image))

for image in oldImages:
    
    #Image loading 
    img = cv2.imread(image)[100:300,100:300,:]
    rgb_planes = cv2.split(img)
    result_planes,result_norm_planes = [], []
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE()
    for plane in rgb_planes:
        # processed_image = cv2.dilate(plane, (15, 15), 10) 
        processed_image = clahe.apply(plane)
        # processed_image= cv2.medianBlur(processed_image,5)
        # norm_processed_image = cv2.normalize(processed_image,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(processed_image)
        # result_norm_planes.append(norm_processed_image)

    result = cv2.merge(result_planes)
    # result_norm = cv2.merge(result_norm_planes)
    
    #image grayscale conversion 
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) 

    #Threshold Processing 
    ret, bin_img = cv2.threshold(gray, 
                                0, 255,  
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 
    
    # noise removal 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
    bin_img = cv2.morphologyEx(bin_img,  
                            cv2.MORPH_OPEN, 
                            kernel, 
                            iterations=2) 

    # Create subplots with 1 row and 2 columns 
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8)) 
    # sure background area 
    sure_bg = cv2.dilate(bin_img, kernel, iterations=3) 
    # imshow(sure_bg, axes[0,0]) 
    # axes[0, 0].set_title('Sure Background') 
    
    # Distance transform 
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5) 
    # imshow(dist, axes[0,1]) 
    # axes[0, 1].set_title('Distance Transform') 
    
    #foreground area 
    ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY) 
    sure_fg = sure_fg.astype(np.uint8)   
    # imshow(sure_fg, axes[1,0]) 
    # axes[1, 0].set_title('Sure Foreground') 
    
    # unknown area 
    unknown = cv2.subtract(sure_bg, sure_fg) 
    # imshow(unknown, axes[1,1]) 
    # axes[1, 1].set_title('Unknown') 
    
    # plt.show()

    # Marker labelling 
    # sure foreground  
    ret, markers = cv2.connectedComponents(sure_fg) 
    
    # Add one to all labels so that background is not 0, but 1 
    markers += 1
    # mark the region of unknown with zero 
    markers[unknown == 255] = 0
    
    # fig, ax = plt.subplots(figsize=(6, 6)) 
    # ax.imshow(markers, cmap="tab20b") 
    # ax.axis('off') 
    # plt.show()

    # watershed Algorithm 
    markers = cv2.watershed(img, markers) 

    # fig, ax = plt.subplots(figsize=(5, 5)) 
    # ax.imshow(markers, cmap="tab20b") 
    # ax.axis('off') 
    # plt.show() 


    labels = np.unique(markers) 

    coins = [] 
    # Iterate through the labels and process each one
    for label in labels[2:]:
        # Create a binary image in which only the area of the label is in the foreground
        target = np.where(markers == label, 255, 0).astype(np.uint8)

        # Perform contour extraction on the created binary image
        _, contours, _ = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Check if there are any contours found
        if contours:
            # Draw the outline for each contour
            for contour in contours:
                img = cv2.drawContours(img, [contour], -1, color=(0, 23, 223), thickness=2)

    # imshow(img)

    img = cv2.imread("jpgData\9\9F--46-_jpg.rf.a10da8c992871ddc643e74e0bbe30c94.jpg")[100:300,100:300,:]
    rgb_planes = cv2.split(img)
    result_planes=[]
    for plane in rgb_planes:
        plane = plane * (markers > 1)
        result_planes.append(plane)
    result_image = cv2.merge(result_planes)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    # plt.imshow(result_image)

    result_hsv = cv2.cvtColor(result_image, cv2.COLOR_RGB2HSV)


    # fig, ax = plt.subplots(1,3, figsize=(15,4))
    # ax[0].imshow(result_hsv[:,:,0]) #hue
    # ax[0].set_title("HUE")
    # ax[1].imshow(result_hsv[:,:,1]) #saturation
    # ax[1].set_title("SATURATION")
    # ax[2].imshow(result_hsv[:,:,2]) #value
    # ax[2].set_title("VALUE")
    # fig.suptitle("HSV Image")
    # plt.show()
    cv2.imwrite(image.replace(path,newPath,1),result_hsv)

