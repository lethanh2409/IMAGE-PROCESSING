import matplotlib.pyplot as plt
import numpy as np

COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))

u0 = 2
v0 = 2
I1 = 0.5 * np.exp(1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / 8)
I2 = np.exp(-1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / 8)
I3 = I1 + I2

Itilde3 = np.fft.fftshift(np.fft.fft2(I3))

print("Real part of I3:")
print(np.real(I3))
print("\nImaginary part of I3:")
print(np.imag(I3))

np.set_printoptions(precision=4, suppress=True)
print("\nReal part of DFT(I3):")
print(np.real(Itilde3))
print("\nImaginary part of DFT(I3):")
print(np.imag(Itilde3))

plt.figure(figsize=(8, 6))

plt.subplot(2,2,1)
plt.imshow(np.real(I3), cmap='gray')
plt.title('Real part of I3')

plt.subplot(2,2,2)
plt.imshow(np.imag(I3), cmap='gray')
plt.title('Imaginary part of I3')

plt.subplot(2,2,3)
plt.imshow(np.real(Itilde3), cmap='gray')
plt.title('Real part of DFT(I3)')

plt.subplot(2,2,4)
plt.imshow(np.imag(Itilde3), cmap='gray')
plt.title('Imaginary part of DFT(I3)')

plt.show()



plt.figure(figsize=(8,6))

plt.subplot(2,2,1)
plt.text(0.1, 0.5, "Real(I3)\n\n" + str(np.real(I3)), fontsize=10)
plt.axis('off')
plt.subplot(2,2,2)
plt.text(0.1, 0.5, "Imag(I3)\n\n" + str(np.imag(I3)), fontsize=10)
plt.axis('off')
plt.subplot(2,2,3)
plt.text(0.1, 0.5, "Real(DFT(I3))\n\n" + str(np.real(Itilde3)), fontsize=10)
plt.axis('off')
plt.subplot(2,2,4)
plt.text(0.1, 0.5, "Imag(DFT(I3))\n\n" + str(np.imag(Itilde3)), fontsize=10)
plt.axis('off')

plt.show()
