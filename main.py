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
    def __init__(self, source, diameter, sigma_d, sigma_r):
        self.source = source
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
        neighbour_x = x - x_def
        neighbour_y = y - y_def
        if neighbour_x >= len(self.source):
            neighbour_x += 2*x_def
        if neighbour_y >= len(self.source[0]):
            neighbour_y += 2*y_def
        return (neighbour_x, neighbour_y)

    def apply_bilateral_filter(self, filtered_image, x, y):
        hl = int((self.diameter-1)/2)
        i_filtered = 0
        Wp = 0
        i = 0
        while i < self.diameter:
            j = 0
            while j < self.diameter:
                neighbour_x, neighbour_y = self.find_neighbours(x, y, (hl - i), (hl - j))
                g_r = self.gaussian(int(self.source[neighbour_x][neighbour_y]) - int(self.source[x][y]), self.sigma_r)
                g_d = self.gaussian(distance(neighbour_x, neighbour_y, x, y), self.sigma_d)
                w = g_r * g_d
                i_filtered += self.source[neighbour_x][neighbour_y] * w
                Wp += w
                j += 1
            i += 1
        i_filtered = i_filtered / Wp
        filtered_image[x][y] = int(round(i_filtered))

    def create_bilateral_filter(self):
        filtered_image = np.zeros(self.source.shape)
        i = 0
        while i < len(self.source):
            print(i)
            j = 0
            while j < len(self.source[0]):
                self.apply_bilateral_filter(filtered_image, i, j)
                j += 1
            i += 1
        return filtered_image


if __name__ == "__main__":
    src = cv2.imread(f"{image_path}/0.png", 0)
    filtered_image_OpenCV = cv2.bilateralFilter(src, 5, 12.0, 16.0)
    cv2.imwrite("original_image_grayscale.png", src)
    cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)
    bilateral_filter = BilateralFilter(src, 5, 12.0, 16.0)
    filtered_image_own = bilateral_filter.create_bilateral_filter()
    cv2.imwrite("filtered_image_own.png", filtered_image_own)

