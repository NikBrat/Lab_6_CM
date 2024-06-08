import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def display_image(photo, title, option=0):
    # displaying and saving photo
    plt.imshow(photo, cmap='gray')
    plt.axis('off')
    if option:
        plt.imsave(f'result/task_4/{title}.png', photo, cmap='gray')
    plt.show()
    plt.close()


def fourier_transform(matrix, width, height):
    # 2D fourier transform
    matrix_image = np.fft.fft2(matrix, (height, width))
    return matrix_image


def inverse_fourier_transform(image):
    # 2D inverse fourier transform
    restored_photo = np.fft.ifft2(image)
    restored_photo = np.clip(restored_photo, 0, 1)
    return restored_photo.real


# reading source image
src_image = cv.imread('source_images/task_4/bmw.jpeg', 0) / 255

# parameters for padding in fourier transforms
h, w = src_image.shape
ht, wd = h+3-1, w+3-1
# kernel for edge detection
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# using convolution
edges = cv.filter2D(src_image, ddepth=-1, kernel=kernel).clip(0, 1)
display_image(edges, 'Edges', 0)

# using combination of fourier transforms
f_mp = np.multiply(fourier_transform(src_image, wd, ht), fourier_transform(kernel, wd, ht))
edges = inverse_fourier_transform(f_mp)
display_image(edges, 'Edges_fourier', 0)
