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

    def calculate_single_value(self, neighbor_val, org_val, gaussian_d, sum_value, Wp):
        """Calculates the affect that this current neighbor has on are current pixel, and add it to the current sum.
        In addition, add this neighbor value to curr weight sum
        """
        gaussian_r = self.gaussian(int(neighbor_val) - int(org_val),
                                   self.sigma_r)
        w = gaussian_r * gaussian_d
        sum_value += neighbor_val * w
        Wp += w
        return sum_value, Wp

    def sum_filtered_pixel_value_and_weight(self, x, y, neighbour_x, neighbour_y, gaussian_d, sum_value, Wp):
        """sums the values of all of the current pixel's neighbours (in the diameter square) affects on him (sum_value)
        and the sum of those neighbours values (Wp), for each color (for RGB returns a vector size 3)"""
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

    def find_neighbours(self, x, y, x_def, y_def):
        """finds the coordinates of the current wanted neighbor, in distance x/y_def from the original point.
        In case this neighbor is off the image limits, returns the reflected pixel (in the diameter square)"""
        neighbour_x = x - x_def
        neighbour_y = y - y_def
        if neighbour_x >= len(self.original):
            neighbour_x += 2 * x_def
        if neighbour_y >= len(self.original[0]):
            neighbour_y += 2 * y_def
        return neighbour_x, neighbour_y

    def apply_filter_on_pixel(self, x, y):
        """Calculates the values of the current pixel after the bilateral filtering, by going over all the pixel's
        neighbors in the diameter range and activates the BF function
        :returns the pixel's value (in each color) after bilateral filtering"""
        half_diameter = int((self.diameter - 1) / 2)
        sum_filtered_val, Wp = create_filter_and_sum_values()
        for i in range(self.diameter):
            for j in range(self.diameter):
                neighbour_x, neighbour_y = self.find_neighbours(x, y, (half_diameter - i), (half_diameter - j))
                gaussian_d = self.gaussian(distance(neighbour_x, neighbour_y, x, y), self.sigma_d)
                sum_filtered_val, Wp = self.sum_filtered_pixel_value_and_weight(x, y, neighbour_x, neighbour_y,
                                                                                gaussian_d,
                                                                                sum_filtered_val, Wp)
        self.filtered_image[x][y] = sum_filtered_val / Wp

    def resolve(self):
        for i in range(len(self.original)):
            print(i)
            for j in range(len(self.original[0])):
                self.apply_filter_on_pixel(i, j)
        return self.filtered_image


if __name__ == "__main__":
    if COLOR:
        original_image = cv2.imread(f"{image_path}/waves1.png")
    else:
        original_image = cv2.imread(f"{image_path}/dog.png", 0)
    cv2.imwrite("original_image.png", original_image)
    for diameter in DIAMETER:
        for sigma_d in SIGMA_D:
            for sigma_r in SIGMA_R:
                bilateral_filter = BilateralFilter(original_image, diameter, sigma_d, sigma_r, COLOR)
                filtered_image = bilateral_filter.resolve()
                cv2.imwrite(f"filtered_image_({diameter},{sigma_d},{sigma_r}).png", filtered_image)
