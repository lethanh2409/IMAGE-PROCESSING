import matplotlib.pyplot as plt
import numpy as np

COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))

u0 = 2
v0 = 2
I2 = np.exp(-1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / 8)

Itilde2 = np.fft.fftshift(np.fft.fft2(I2))

print("Real part of I2:")
print(np.real(I2))
print("\nImaginary part of I2:")
print(np.imag(I2))

np.set_printoptions(precision=4, suppress=True)
print("\nReal part of DFT(I2):")
print(np.real(Itilde2))
print("\nImaginary part of DFT(I2):")
print(np.imag(Itilde2))

plt.figure(figsize=(8, 6))

plt.subplot(2,2,1)
plt.imshow(np.real(I2), cmap='gray')
plt.title('Real part of I2')

plt.subplot(2,2,2)
plt.imshow(np.imag(I2), cmap='gray')
plt.title('Imaginary part of I2')

plt.subplot(2,2,3)
plt.imshow(np.real(Itilde2), cmap='gray')
plt.title('Real part of DFT(I2)')

plt.subplot(2,2,4)
plt.imshow(np.imag(Itilde2), cmap='gray')
plt.title('Imaginary part of DFT(I2)')

plt.show()



plt.figure(figsize=(8,6))

plt.subplot(2,2,1)
plt.text(0.1, 0.5, "Real(I2)\n\n" + str(np.real(I2)), fontsize=10)
plt.axis('off')
plt.subplot(2,2,2)
plt.text(0.1, 0.5, "Imag(I2)\n\n" + str(np.imag(I2)), fontsize=10)
plt.axis('off')
plt.subplot(2,2,3)
plt.text(0.1, 0.5, "Real(DFT(I2))\n\n" + str(np.real(Itilde2)), fontsize=10)
plt.axis('off')
plt.subplot(2,2,4)
plt.text(0.1, 0.5, "Imag(DFT(I2))\n\n" + str(np.imag(Itilde2)), fontsize=10)
plt.axis('off')

plt.show()
