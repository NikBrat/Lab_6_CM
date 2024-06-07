import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as img


def display_image(photo, title, option=0):
    # displaying and saving photo
    plt.imshow(photo, cmap='gray')
    plt.axis('off')
    if option:
        plt.imsave(f'result/task_2/{title}.png', photo, cmap='gray')
    plt.show()
    plt.close()


def fourier_transform(matrix, width, height):
    matrix_image = np.fft.fft2(matrix, (height, width))
    return matrix_image


def inverse_fourier_transform(image):
    restored_photo = np.fft.ifft2(image)
    # restored_photo = np.fmax(restored_photo/np.max(restored_photo), np.zeros(restored_photo.shape))
    # restored_photo = np.clip(restored_photo, 0, 1)
    return restored_photo.real


def blurring(image, option, ks):
    # photo blurring
    match option:
        case 1:
            # averaging blurring
            kernel = np.ones((ks, ks), np.float64) / ks**2
            return cv.filter2D(image, ddepth=-1, kernel=kernel)
        case 2:
            # gaussian blurring
            gauss = np.array([[np.e**((-9/(ks**2))*((i - (ks+1)/2)**2 + (j - (ks+1)/2)**2)) for i in range(1, 10)] for j in range(1, 10)])
            kernel = gauss / np.sum(gauss)
            return cv.filter2D(image, ddepth=-1, kernel=kernel)
        case _:
            print('Empty option =)')


# kernel size
n = [5, 9, 11]
src_image = cv.imread('source_images/task_2_3/smoking-boys.jpg', 0)
h, w = src_image.shape
"""
for k in n:
    blurred = blurring(src_image, 1, k)
    display_image(blurred, f'Averaging_{k}', 1)
"""
for k in n:
    ht, wd = h+k-1, w+k-1
    gauss = np.array([[np.e ** ((-9 / (k ** 2)) * ((i - (k + 1) / 2) ** 2 + (j - (k + 1) / 2) ** 2)) for i in range(1, 10)] for j in range(1, 10)])
    kernel = gauss / np.sum(gauss)
    f_mp = np.multiply(fourier_transform(src_image, wd, ht), fourier_transform(kernel, wd, ht))
    blurred = inverse_fourier_transform(f_mp)
    display_image(blurred, f'Gaussian_fourier_{k}', 1)





