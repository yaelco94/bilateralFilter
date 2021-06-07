import numpy as np
from cv2 import cv2
import math
import os

pwd = os.path.abspath(os.getcwd())
image_path = f"{pwd}/images"
pi = math.pi


def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


class BilateralFilter(object):
    def __init__(self, original, diameter, sigma_d, sigma_r):
        self.original = original
        self.filtered_image = np.zeros(self.original.shape)
        self.sigma_d = sigma_d
        self.sigma_r = sigma_r
        self.diameter = diameter
        self.gaussian_constant_d = (1.0 / (2 * pi * (sigma_d ** 2))**(1/2))
        self.gaussian_constant_r = (1.0 / (2 * pi * (sigma_r ** 2))**(1/2))

    def gaussian(self, x, sigma):
        if sigma == 'r':
            return self.gaussian_constant_r * math.exp(- (x ** 2) / (2 * self.sigma_r ** 2))
        else:
            return self.gaussian_constant_d * math.exp(- (x ** 2) / (2 * self.sigma_d ** 2))

    def find_neighbours(self, x, y, x_def, y_def):
        """finds the coordinate of the current wanted neighbor, in distance x/y_def from the original point
        in case this neighbor is off the image limits, returns the reflected pixel (in the diameter square)"""
        neighbour_x = x - x_def
        neighbour_y = y - y_def
        if neighbour_x >= len(self.original):
            neighbour_x += 2*x_def
        if neighbour_y >= len(self.original[0]):
            neighbour_y += 2*y_def
        return (neighbour_x, neighbour_y)

    def apply_bilateral_filter(self, x, y):
        half_diameter = int((self.diameter-1)/2)
        i_filtered = np.zeros(3)
        Wp = np.zeros(3)
        i = 0
        while i < self.diameter:
            j = 0
            while j < self.diameter:
                neighbour_x, neighbour_y = self.find_neighbours(x, y, (half_diameter - i), (half_diameter - j))
                gaussian_d = self.gaussian(distance(neighbour_x, neighbour_y, x, y), self.sigma_d)
                # gaussian_r = self.gaussian(int(self.original[neighbour_x][neighbour_y][0]) - int(self.original[x][y][0]), self.sigma_r)
                # w = gaussian_r * gaussian_d
                # i_filtered += w * self.original[neighbour_x][neighbour_y] 
                gaussian_r_red = self.gaussian(int(self.original[neighbour_x][neighbour_y][0]) - int(self.original[x][y][0]), self.sigma_r)
                gaussian_r_green = self.gaussian(int(self.original[neighbour_x][neighbour_y][1]) - int(self.original[x][y][1]), self.sigma_r)
                gaussian_r_blue = self.gaussian(int(self.original[neighbour_x][neighbour_y][2]) - int(self.original[x][y][2]), self.sigma_r)
                w_red = gaussian_d * gaussian_r_red
                w_green = gaussian_d * gaussian_r_green
                w_blue = gaussian_d * gaussian_r_blue
                i_filtered[0] += w_red*self.original[neighbour_x][neighbour_y][0]
                i_filtered[1] +=w_green*self.original[neighbour_x][neighbour_y][1]
                i_filtered[2] +=w_blue*self.original[neighbour_x][neighbour_y][2]
                Wp +=(w_red,w_green,w_blue)
                j += 1
            i += 1
        i_filtered[0] =i_filtered[0] / (Wp[0])
        i_filtered[1] = i_filtered[1] / (Wp[1])
        i_filtered[2] = i_filtered[2] / (Wp[2])
        self.filtered_image[x][y] =i_filtered

    def create_bilateral_filter(self):
        # while i < len(self.original):
        for i in range(len(self.original)):
            print(i)
            # while j < len(self.original[0]):
            for j in range(len(self.original[0])):
                self.apply_bilateral_filter(i, j)
                j += 1
            i += 1
        return self.filtered_image


if __name__ == "__main__":
    """open image as RGB matrix"""
    original_image = cv2.imread(f"{image_path}/04.png")[...,::-1]
    # filtered_image_OpenCV = cv2.bilateralFilter(original_image, 5, 12.0, 16.0)
    # cv2.imwrite("original_image_grayscale.png", src)
    # cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)
    bilateral_filter = BilateralFilter(original_image, 3, 12.0, 16.0)
    filtered_image = bilateral_filter.create_bilateral_filter()
    cv2.imwrite("filtered_image_window_3.png", filtered_image)
    bilateral_filter = BilateralFilter(original_image, 5, 12.0, 16.0)
    filtered_image = bilateral_filter.create_bilateral_filter()
    cv2.imwrite("filtered_image_window_5.png", filtered_image)
    bilateral_filter = BilateralFilter(original_image, 7, 12.0, 16.0)
    filtered_image = bilateral_filter.create_bilateral_filter()
    cv2.imwrite("filtered_image_window_7.png", filtered_image)
    bilateral_filter = BilateralFilter(original_image, 9, 12.0, 16.0)
    filtered_image = bilateral_filter.create_bilateral_filter()
    cv2.imwrite("filtered_image_window_9.png", filtered_image)

