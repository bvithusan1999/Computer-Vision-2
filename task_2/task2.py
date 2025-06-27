import cv2
import numpy as np

def region_growing(img, seeds, threshold=10):
    visited = np.zeros_like(img, dtype=bool)
    result = np.zeros_like(img, dtype=np.uint8)
    h, w = img.shape
    stack = list(seeds)

    for seed in seeds:
        result[seed] = 255
        visited[seed] = True

    while stack:
        x, y = stack.pop()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < h) and (0 <= ny < w) and not visited[nx, ny]:
                    if abs(int(img[nx, ny]) - int(img[x, y])) <= threshold:
                        result[nx, ny] = 255
                        stack.append((nx, ny))
                    visited[nx, ny] = True
    return result

# Load the original image
image = cv2.imread("task_1/original_image.jpg", cv2.IMREAD_GRAYSCALE)
seeds = [(30, 30), (70, 70)]
region_result = region_growing(image, seeds)
cv2.imwrite("task_2/region_growing_result.jpg", region_result)
