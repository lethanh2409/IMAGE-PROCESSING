import matplotlib.pyplot as plt
import numpy as np

COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))

u0 = 2
v0 = 2
I1 = 0.5 * np.exp(1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / 8)
I2 = np.exp(-1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / 8)
I4 = -1j * (I1 - I2)

Itilde4 = np.fft.fftshift(np.fft.fft2(I4))

print("Real part of I4:")
print(np.real(I4))
print("\nImaginary part of I4:")
print(np.imag(I4))

np.set_printoptions(precision=4, suppress=True)
print("\nReal part of DFT(I4):")
print(np.real(Itilde4))
print("\nImaginary part of DFT(I4):")
print(np.imag(Itilde4))

plt.figure(figsize=(8, 6))

plt.subplot(2,2,1)
plt.imshow(np.real(I4), cmap='gray')
plt.title('Real part of I4')

plt.subplot(2,2,2)
plt.imshow(np.imag(I4), cmap='gray')
plt.title('Imaginary part of I4')

plt.subplot(2,2,3)
plt.imshow(np.real(Itilde4), cmap='gray')
plt.title('Real part of DFT(I4)')

plt.subplot(2,2,4)
plt.imshow(np.imag(Itilde4), cmap='gray')
plt.title('Imaginary part of DFT(I4)')

plt.show()



plt.figure(figsize=(8,6))

plt.subplot(2,2,1)
plt.text(0.1, 0.5, "Real(I4)\n\n" + str(np.real(I4)), fontsize=10)
plt.axis('off')
plt.subplot(2,2,2)
plt.text(0.1, 0.5, "Imag(I4)\n\n" + str(np.imag(I4)), fontsize=10)
plt.axis('off')
plt.subplot(2,2,3)
plt.text(0.1, 0.5, "Real(DFT(I4))\n\n" + str(np.real(Itilde4)), fontsize=10)
plt.axis('off')
plt.subplot(2,2,4)
plt.text(0.1, 0.5, "Imag(DFT(I4))\n\n" + str(np.imag(Itilde4)), fontsize=10)
plt.axis('off')

plt.show()
