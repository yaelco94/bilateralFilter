import numpy as np
from cv2 import cv2
import math
import os
from config import *

pwd = os.path.abspath(os.getcwd())
image_path = f"{pwd}/images"
pi = math.pi


def distance(x, y, i, j):
    return np.sqrt((x - i) ** 2 + (y - j) ** 2)


def create_filter_and_sum_values():
    if COLOR:
        i_filtered = np.zeros(3)
        Wp = np.zeros(3)
    else:
        i_filtered = 0
        Wp = 0
    return i_filtered, Wp


class BilateralFilter(object):
    def __init__(self, original, diameter, sigma_d, sigma_r, color=True):
        self.original = original
        self.filtered_image = np.zeros(self.original.shape)
        self.sigma_d = sigma_d
        self.sigma_r = sigma_r
        self.diameter = diameter
        self.gaussian_constant_d = (1.0 / (2 * pi * (sigma_d ** 2)) ** (1 / 2))
        self.gaussian_constant_r = (1.0 / (2 * pi * (sigma_r ** 2)) ** (1 / 2))
        self.color = color

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
        if neighbour_x not in range(len(self.original)):
            neighbour_x += 2 * x_def
        if neighbour_y not in range(len(self.original[0])):
            neighbour_y += 2 * y_def
        return neighbour_x, neighbour_y

    def calculate_single_value(self, neighbor_val, org_val, gaussian_d, sum_value, Wp):
        gaussian_r = self.gaussian(int(neighbor_val) - int(org_val),
                                   'r')
        w = gaussian_r * gaussian_d
        sum_value += neighbor_val * w
        Wp += w
        return sum_value, Wp

    def find_filtered_pixel_sum_value(self, x, y, neighbour_x, neighbour_y, gaussian_d, sum_value, Wp):
        if not self.color:
            return self.calculate_single_value(self.original[neighbour_x][neighbour_y], self.original[x][y], gaussian_d,
                                               sum_value, Wp)
        else:
            # calculate the filtered pixel value for red, green, and blue
            for i in range(3):
                neighbor_val = self.original[neighbour_x][neighbour_y][i]
                sum_value[i], Wp[i] = self.calculate_single_value(neighbor_val, self.original[x][y][i], gaussian_d,
                                                                  sum_value[i], Wp[i])
            return sum_value, Wp

    def apply_bilateral_filter(self, x, y):
        half_diameter = int((self.diameter - 1) / 2)
        sum_filtered_val, Wp = create_filter_and_sum_values()  # initiate to zero
        for i in range(self.diameter):
            for j in range(self.diameter):
                neighbour_x, neighbour_y = self.find_neighbours(x, y, (half_diameter - i), (half_diameter - j))
                gaussian_d = self.gaussian(distance(neighbour_x, neighbour_y, x, y), 'd')
                sum_filtered_val, Wp = self.find_filtered_pixel_sum_value(x, y, neighbour_x, neighbour_y, gaussian_d,
                                                                          sum_filtered_val, Wp)
        self.filtered_image[x][y] = sum_filtered_val / Wp

    def create_bilateral_filter(self):
        for i in range(len(self.original)):
            for j in range(len(self.original[0])):
                self.apply_bilateral_filter(i, j)
        return self.filtered_image


if __name__ == "__main__":
    if COLOR:
        original_image = cv2.imread(f"mini_taj.png")
    else:
        original_image = cv2.imread(f"{image_path}/noised_lizard.png", 0)
    cv2.imwrite("original_image.png", original_image)
    print(original_image[100][100])
    for diameter in DIAMETER:
        for r in SIGMA_R:
            for d in SIGMA_D:
                print(f" vals are {d},{r}")
                bilateral_filter = BilateralFilter(original_image, diameter, d, r, COLOR)
                filtered_image = bilateral_filter.create_bilateral_filter()
                print(filtered_image[100][100])
                cv2.imwrite(f"filtered_image_{diameter},{d},{r}.png", filtered_image)
