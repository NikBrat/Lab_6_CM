import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img


def display_image(photo, title, option=0):
    plt.figure()
    # displaying and saving photo
    plt.imshow(photo)
    plt.axis('off')
    if option:
        plt.imsave(f'result/task_1/{title}.png', photo, cmap='gray')
    plt.show()
    plt.close()


def fourier_transform(photo):
    image_r = np.fft.fftshift(np.fft.fft2(photo[:, :, 0]))
    image_g = np.fft.fftshift(np.fft.fft2(photo[:, :, 1]))
    image_b = np.fft.fftshift(np.fft.fft2(photo[:, :, 2]))
    fourier_image = np.stack([image_r, image_g, image_b], axis=2)
    return fourier_image


def inverse_fourier_transform(image, log_max, angle):
    fourier_image = np.exp(1j*angle)*(np.exp(image*log_max) - 1)

    photo_r = np.fft.ifft2(np.fft.ifftshift(fourier_image[:, :, 0]))
    photo_g = np.fft.ifft2(np.fft.ifftshift(fourier_image[:, :, 1]))
    photo_b = np.fft.ifft2(np.fft.ifftshift(fourier_image[:, :, 2]))
    restored_photo = np.stack([photo_r, photo_g, photo_b], axis=2)
    # restored_photo = np.fmax(restored_photo/np.max(restored_photo), np.zeros(restored_photo.shape))
    restored_photo = np.clip(restored_photo, 0, 1)
    return restored_photo.real


src_photo = img.imread('source_images/task_1/13.png')
src_photo_1 = np.copy(src_photo)
f_im = fourier_transform(src_photo_1)
# creating log of fourier image

f_abs = np.abs(f_im)
f_log = np.log(f_abs + 1)
f_log_mx = np.max(f_log)
f_log /= np.max(f_log)
# display_image(f_log, 'Fourier_13', 1)

# restoring photo from modified log of image
f_ang = np.angle(f_im)
f_log_m = img.imread('result/task_1/Fourier_13_modified_1.png')[:, :, :3]
result = inverse_fourier_transform(f_log_m, f_log_mx, f_ang)
display_image(result, 'result_13_2', 1)
