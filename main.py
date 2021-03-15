import numpy as np
from cv2 import cv2
import sys
import math
import os

pwd = os.path.abspath(os.getcwd())
image_path = f"{pwd}/images"

def distance(x, y, i, j):
    return np.sqrt((x-i)*2 + (y-j)*2)


def gaussian(x, sigma):
    pi = math.pi
    return (1.0 / (2 * pi * (sigma * 2))(1/2)) * math.exp(- (x * 2) / (2 * sigma ** 2))

def find_neighbours(source, x, y,x_def, y_def):
    neighbour_x = x - x_def
    neighbour_y = y -  y_def
    if neighbour_x >= len(source):
        neighbour_x += 2*x_def
    if neighbour_y >= len(source[0]):
        neighbour_y += 2*y_def
    return (neighbour_x, neighbour_y)

def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = int((diameter-1)/2)
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x, neighbour_y = find_neighbours(source, x, y, (hl - i), (hl - j))
            gi = gaussian(int(source[neighbour_x][neighbour_y]) - int(source[x][y]), sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = int(round(i_filtered))



def bilateral_filter_own(source, filter_diameter, sigma_i, sigma_s):
    filtered_image = np.zeros(source.shape)

    i = 0
    while i < len(source):
        print(i)
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
            j += 1
        i += 1
    return filtered_image


if _name_ == "_main_":
    src = cv2.imread(f"{image_path}/03.png", 0)
    filtered_image_OpenCV = cv2.bilateralFilter(src, 5, 12.0, 16.0)
    cv2.imwrite("original_image_grayscale.png", src)
    cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)
    filtered_image_own = bilateral_filter_own(src, 5, 12.0, 16.0)
    cv2.imwrite("filtered_image_own.png", filtered_image_own)



# from cv2 import cv2
# import os

# pwd = os.path.abspath(os.getcwd())
# image_path = f"{pwd}/images"

# img_gaussian_noise = cv2.imread(f"{image_path}/0.png", 0)

# img = img_gaussian_noise

# bilateral_using_cv2 = cv2.bilateralFilter(img, 5, 20, 100, borderType=cv2.BORDER_CONSTANT)

# #save images
# cv2.imwrite(f"{image_path}/01.png", img)
# cv2.imwrite(f"{image_path}/02.png", bilateral_using_cv2)

# #show images
# img_s = cv2.resize(img, (960,540))
# img_bil_s = cv2.resize(bilateral_using_cv2, (960,540))
# cv2.imshow("Original", img_s)
# cv2.imshow("cv2 bilateral", img_bil_s)
# cv2.waitKey(0)
# cv2.destroyAllWindows()