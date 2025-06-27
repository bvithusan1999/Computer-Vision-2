# File: task_1/task1.py

import cv2
import numpy as np
from skimage import util, filters

# Load image
image = cv2.imread("task_1/original_image.jpg", cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise
noisy_image = util.random_noise(image, mode='gaussian', var=0.01)
noisy_image = (255 * noisy_image).astype(np.uint8)
cv2.imwrite("task_1/noisy_image.jpg", noisy_image)

# Apply Otsu's thresholding
thresh_val = filters.threshold_otsu(noisy_image)
otsu_result = noisy_image > thresh_val
otsu_image = (otsu_result * 255).astype(np.uint8)
cv2.imwrite("task_1/otsu_result.jpg", otsu_image)
