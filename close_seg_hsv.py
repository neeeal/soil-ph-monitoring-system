import cv2
import numpy as np
import os
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
    # Distance transform
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
    # Foreground area
    ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    new_image_size=128
    # Apply the mask to the original image to extract the region
    region = cv2.bitwise_and(orig_img, orig_img, mask=sure_fg)
    # Find the bounding box coordinates (non-zero pixels)
    non_zero_coords = np.argwhere(region > 0)
    min_y, min_x, _ = non_zero_coords.min(axis=0)
    max_y, max_x, _ = non_zero_coords.max(axis=0)
    # Crop the region to include only non-zero pixels
    cropped_region = region[min_y:max_y + 1, min_x:max_x + 1]
    cropped_region = cv2.resize(cropped_region, (new_image_size, new_image_size))

    rgb_planes = cv2.split(cropped_region)
    result_planes = []
    # Create a CLAHE object.
    clahe = cv2.createCLAHE(tileGridSize=(3,3),clipLimit=10)
    for plane in rgb_planes:
        processed_image = cv2.medianBlur(plane, 7)
        processed_image = clahe.apply(processed_image) 
        result_planes.append(processed_image)
    result = cv2.merge(result_planes)

    HSV = cv2.cvtColor(result,cv2.COLOR_RGB2HSV)
    H,S,V = cv2.split(HSV)
    V *= 0
    HS = cv2.merge([H,S,V])

    cv2.imwrite(image.replace(path,newPath,1),HS)