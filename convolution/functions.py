import numpy as np
import cv2 as cv

def convolution(arr: np.ndarray, kernel: np.ndarray, mode):

    h, w, _ = arr.shape
    k = kernel.shape[0]
    bias = k // 2

    padded = np.pad(arr, ((bias, bias), (bias, bias), (0, 0)), mode="edge")
    blurred_arr = np.zeros((h, w, 3), dtype=np.uint8)

    kernel = kernel.astype(np.float32)

    for i in range(h):
        for j in range(w):
            for ch in range(3):
                region = padded[i:i+k, j:j+k, ch].astype(np.float32)
                value = np.sum(region * kernel)
                if (mode == "sharp"):
                    blurred_arr[i, j, ch] = np.clip(arr[i, j, ch] + np.round(1.2*(arr[i, j, ch] - np.clip(value, 0, 255))), 0, 255)
                else:
                    blurred_arr[i, j, ch] = np.clip(value, 0, 255)

    return blurred_arr

def generate_kernel(s, mode):
    if mode == "blur":
        return np.full((s, s), 1 / (s * s), dtype=np.float32)

    elif mode == "normal":
        arr = np.random.normal(0, 1, s * s).astype(np.float32)
        arr /= arr.sum()
        return arr.reshape((s, s))

        
    
          

