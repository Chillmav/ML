import numpy as np
import cv2 as cv
from matplotlib.image import imread
from matplotlib import pyplot as plt
from functions import convolution, generate_kernel

img_src = "convolution/donald.jpg"
img_dst = "convolution/blured_donald.jpg"


def img_to_arr(img_src):

    im = imread(img_src)
    print(im.shape)
    return np.asarray(im)

def img_kernel(img_src, img_dst, kernel_size, mode):

    arr = img_to_arr(img_src)
    blur_arr = convolution(arr, generate_kernel(kernel_size, mode), "sharp")
    blur_arr_bgr = cv.cvtColor(blur_arr, cv.COLOR_RGB2BGR)
    cv.imwrite(img_dst, blur_arr_bgr)


def main():

   img_kernel(img_src, img_dst, 31, mode="blur")
   

if "__main__":
    main()