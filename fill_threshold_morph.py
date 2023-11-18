import cv2
import numpy as np
import os
import shutil
from random import randint

## apply watershed and convert to hsv
path = "raw_data\\train"
dirs = [os.path.join(path,dir) for dir in os.listdir(path)[:-1]]
classes = os.listdir('jpgData')
newPath = 'fill-thresh'

for c in classes:
    classPath = os.path.join(newPath,c)
    if os.path.exists(classPath): shutil.rmtree(classPath); os.mkdir(classPath)
    else: os.mkdir(classPath)

oldImages = []
for dir in dirs:
    for image in os.listdir(dir):
        oldImages.append(os.path.join(dir,image))

def center_crop(img, shape, key=0):
    """Returns center cropped image

    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped from center
    """
    center = img.shape
    h = img.shape[key]
    w = img.shape[key]
    x = center[1]/2 - shape/2
    y = center[0]/2 - shape/2

    crop_img = img[int(y):int(y+h), int(x):int(x+w)]
    return crop_img

for image in oldImages:
    # Load image
    img = cv2.imread(os.path.join(path,image))#[100:300, 100:300, :]
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

    # Apply the mask to the original image to extract the region
    region = cv2.bitwise_and(orig_img, orig_img, mask=sure_fg)

    # Find the bounding box coordinates (non-zero pixels)
    non_zero_coords = np.argwhere(region > 0)
    min_y, min_x, _ = non_zero_coords.min(axis=0)
    max_y, max_x, _ = non_zero_coords.max(axis=0)

    # Crop the region to include only non-zero pixels
    cropped_region = orig_img[min_y:max_y + 1, min_x:max_x + 1]

    # cropped_region = cv2.resize(cropped_region, (new_image_size, new_image_size))\
    shape = cropped_region.shape[0]; key = 0
    if cropped_region.shape[0] > cropped_region.shape[1]: shape = cropped_region.shape[1]; key = 1

    cropped_region = center_crop(cropped_region, shape, key)

    rgb_planes = cv2.split(cropped_region)
    result_planes = []

    # Create a CLAHE object.
    clahe = cv2.createCLAHE(tileGridSize=(3,3),clipLimit=2)
    for plane in rgb_planes:
        processed_image = cv2.medianBlur(plane, 5)
        # processed_image = clahe.apply(processed_image) 
        result_planes.append(processed_image)

    result = cv2.merge(result_planes)

    rgb_planes = cv2.split(result)
    result_planes=[]
    for plane in rgb_planes:
        # processed_image = cv2.medianBlur(plane, 7)
        processed_image = clahe.apply(plane) 
        result_planes.append(processed_image)

    result = cv2.merge(result_planes)
    # plt.imshow(result)
    # plt.axis('off')
    # plt.show()

    hsv = cv2.cvtColor(result,cv2.COLOR_RGB2HSV)
    # plt.imshow(hsv)
    # plt.axis('off')
    # plt.show()

    h,s,v = cv2.split(hsv)
    v *= 0
    # HS = cv2.merge([h,s,v])
    # plt.imshow(HS)
    # plt.axis('off')
    # plt.show()

    H = hsv[:,:,0]

    # S = hsv[:,:,1]
    # plt.imshow(S)
    # plt.axis('off')
    # plt.show()

    gray = cv2.cvtColor(result,cv2.COLOR_RGB2GRAY)
    ret, im_th = cv2.threshold(gray, 31, 255, cv2.THRESH_BINARY)

    IMG = cv2.bitwise_and(result, result, mask=im_th)

    IMG = center_crop(IMG, shape)

    cv2.imwrite(image.replace(path,newPath,1),H)