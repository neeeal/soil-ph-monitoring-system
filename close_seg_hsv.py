import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from random import shuffle
import shutil

## apply watershed and convert to hsv
path = "jpgData"
dirs = [os.path.join(path,dir) for dir in os.listdir(path)]
classes = os.listdir(path)
newPath = 'close_seg'

for c in classes:
    classPath = os.path.join(newPath,c)
    if os.path.exists(classPath): shutil.rmtree(classPath); os.mkdir(classPath)
    else: os.mkdir(classPath)

oldImages = []
for dir in dirs:
    for image in os.listdir(dir):
        oldImages.append(os.path.join(dir,image))

for image in oldImages:
    # Load image
    img = cv2.imread(image)#[100:300, 100:300, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_img = img.copy()

    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    # Threshold Processing
    ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=7)
    bin_img = cv2.dilate(bin_img, kernel, iterations=5)

    # Sure background area
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    sure_bg = cv2.dilate(bin_img, kernel, iterations=10)

    # Distance transform
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)

    # Foreground area
    ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # Unknown area
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    # Watershed Algorithm
    markers = cv2.watershed(img, markers)

    # Create a directory to save the extracted region images
    if not os.path.exists("segmented_regions"):
        os.mkdir("segmented_regions")

    # Get unique labels/markers (excluding background and unknown)
    unique_labels = np.unique(markers)[2:]
    shuffle(unique_labels)
    # Iterate through the labels and process each one
    for label in unique_labels:
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

    # Define the size of the new image and initialize it
    new_image_size = 128

    # Iterate through the labels and process each one
    for label in unique_labels:
        # Create a binary mask for the specific label
        region_mask = np.where(markers == label, 255, 0).astype(np.uint8)

        # Apply the mask to the original image to extract the region
        region = cv2.bitwise_and(orig_img, orig_img, mask=region_mask)

        # Find the bounding box coordinates (non-zero pixels)
        non_zero_coords = np.argwhere(region > 0)
        min_y, min_x, _ = non_zero_coords.min(axis=0)
        max_y, max_x, _ = non_zero_coords.max(axis=0)

        # Crop the region to include only non-zero pixels
        cropped_region = region[min_y:max_y + 1, min_x:max_x + 1]
        
        new_image = cv2.resize(cropped_region, (new_image_size, new_image_size))

    rgb_planes = cv2.split(new_image)
    result_planes = []
    # Create a CLAHE object.
    clahe = cv2.createCLAHE(tileGridSize=(5,5),clipLimit=5)
    for plane in rgb_planes:
        processed_image = cv2.medianBlur(plane, 7)
        processed_image = clahe.apply(processed_image) 
        result_planes.append(processed_image)

    result = cv2.merge(result_planes)

    HSV = cv2.cvtColor(result,cv2.COLOR_RGB2HSV)
    # H,S,V = cv2.split(HSV)
    # V *= 0
    # HSV = cv2.merge([H,S,V])

    cv2.imwrite(image.replace(path,newPath,1).split('.')[0]+f"-region_{label}.png",HSV)