import numpy as np
import matplotlib.pyplot as plt

imagefile = ['camerabin.sec', 'salesmanbin.sec', 'headbin.sec', 'eyeRbin.sec']

def process(imagefile):

    image_data = np.fromfile(imagefile, dtype=np.uint8).reshape(256, 256)
    image_dft = np.fft.fftshift(np.fft.fft2(image_data))
    real_part = np.real(image_dft)
    imaginary_part = np.imag(image_dft)
    magnitude_spectrum = np.log(np.abs(image_dft) + 1)
    phase = np.angle(image_dft)


    plt.figure(figsize=(13, 3))

    plt.subplot(1,5,1)
    plt.title('Original Image')
    plt.imshow(image_data, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.title('Real Part DFT')
    plt.imshow(real_part, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.title('Imaginary Part DFT')
    plt.imshow(imaginary_part, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.title('Log-Magnitude Spectrum')
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.title('Phase DFT')
    plt.imshow(phase, cmap='gray', vmin=-np.pi, vmax=np.pi)
    plt.axis('off')

    plt.show()

for imagefile in imagefile:
    process(imagefile)
