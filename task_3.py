import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def display_image(photo, title, option=0):
    # displaying and saving photo
    plt.imshow(photo, cmap='gray')
    plt.axis('off')
    if option:
        plt.imsave(f'result/task_3/{title}.png', photo, cmap='gray')
    plt.show()
    plt.close()


def fourier_transform(matrix, width, height):
    matrix_image = np.fft.fft2(matrix, (height, width))
    return matrix_image


def inverse_fourier_transform(image):
    restored_photo = np.fft.ifft2(image)
    restored_photo = np.clip(restored_photo, 0, 1)
    return restored_photo.real


src_image = cv.imread('source_images/task_2_3/smoking-boys.jpg', 0)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

sharpened = cv.filter2D(src_image, ddepth=-1, kernel=kernel)
display_image(sharpened, 'Sharpened', 1)


h, w = src_image.shape
# print(src_image.shape)
ht, wd = h+3-1, w+3-1

f_mp = np.multiply(fourier_transform(src_image, wd, ht), fourier_transform(kernel, wd, ht))
sharpened = inverse_fourier_transform(f_mp)
display_image(sharpened, f'Sharpened_fourier', 0)
