import os
import shutil
import cv2
import numpy as np
from PIL import Image
from math import floor, sqrt
from random import shuffle

## apply watershed and convert to hsv
path = "jpgData"
dirs = [os.path.join(path,dir) for dir in os.listdir(path)]
classes = os.listdir(path)
newPath = 'stitch-hsv'

for c in classes:
    classPath = os.path.join(newPath,c)
    if os.path.exists(classPath): shutil.rmtree(classPath); os.mkdir(classPath)
    else: os.mkdir(classPath)

oldImages = []
for dir in dirs:
    for image in os.listdir(dir):
        oldImages.append(os.path.join(dir,image))

for image in oldImages:

    # Load your image
    img = cv2.imread(image)[100:300, 100:300, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_planes = cv2.split(img)
    result_planes, result_norm_planes = [], []

    # Create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE()

    for plane in rgb_planes:
        processed_image = clahe.apply(plane)
        processed_image = cv2.medianBlur(processed_image, 5)
        result_planes.append(processed_image)

    result = cv2.merge(result_planes)
    
    orig_img = result.copy()
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2HSV)
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Threshold Processing
    ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)

    # # Create subplots with 1 row and 2 columns
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    # Sure background area
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

    try:
        # Define the size of the new image and initialize it
        new_image_size = 224
        new_image = Image.new("RGB", (new_image_size, new_image_size), "black")
        contents_num = floor(sqrt(len(unique_labels)))**2
        sub_image_size = (new_image_size*new_image_size // contents_num)
        sub_image_size = floor(np.sqrt(sub_image_size))

        current_x, current_y = 0, 0
        shuffle(unique_labels)
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
            
            cropped_region = cv2.resize(cropped_region, (sub_image_size, sub_image_size))

            # Paste the extracted region into the new image
            new_image.paste(Image.fromarray(cropped_region), (current_x, current_y))

            # Update the current position for pasting
            current_y += cropped_region.shape[1]

            # Check if the current row is full, and if so, move to the next row
            if current_y + cropped_region.shape[1] > new_image_size:
                current_x += cropped_region.shape[0]
                current_y = 0

        # Display the new image
        # plt.imshow(new_image)
        # plt.show()

        # cv2.imwrite(image.replace(path,newPath,1),new_image)
        hue, sat, val = new_image.split()
        val = val.point(lambda i : i * 0)
        new_image = Image.merge(new_image.mode,(hue,sat,val)) ## set value to zero, only hue and saturation
        new_image.save(image.replace(path,newPath,1))        

    except Exception as e:
        print(e, image)
