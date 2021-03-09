from cv2 import cv2
import os

pwd = os.path.abspath(os.getcwd())
image_path = f"{pwd}/images"

img_gaussian_noise = cv2.imread(f"{image_path}/0.png", 0)

img = img_gaussian_noise

bilateral_using_cv2 = cv2.bilateralFilter(img, 5, 20, 100, borderType=cv2.BORDER_CONSTANT)

#save images
cv2.imwrite(f"{image_path}/01.png", img)
cv2.imwrite(f"{image_path}/02.png", bilateral_using_cv2)

#show images
img_s = cv2.resize(img, (960,540))
img_bil_s = cv2.resize(bilateral_using_cv2, (960,540))
cv2.imshow("Original", img_s)
cv2.imshow("cv2 bilateral", img_bil_s)
cv2.waitKey(0)
cv2.destroyAllWindows()