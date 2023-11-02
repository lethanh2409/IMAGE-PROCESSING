import matplotlib.pyplot as plt
import numpy as np

COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))

u1 = 1.5
v1 = 1.5

I5 = np.cos(2 * np.pi * (u1 * COLS + v1 * ROWS))

Itilde5 = np.fft.fftshift(np.fft.fft2(I5))

print("Real part of I5:")
print(np.real(I5))
print("\nImaginary part of I5:")
print(np.imag(I5))

np.set_printoptions(precision=4, suppress=True)
print("\nReal part of DFT(I5):")
print(np.real(Itilde5))
print("\nImaginary part of DFT(I5):")
print(np.imag(Itilde5))

plt.figure(figsize=(8, 6))

plt.subplot(2,2,1)
plt.imshow(np.real(I5), cmap='gray')
plt.title('Real part of I5')

plt.subplot(2,2,2)
plt.imshow(np.imag(I5), cmap='gray')
plt.title('Imaginary part of I5')

plt.subplot(2,2,3)
plt.imshow(np.real(Itilde5), cmap='gray')
plt.title('Real part of DFT(I5)')

plt.subplot(2,2,4)
plt.imshow(np.imag(Itilde5), cmap='gray')
plt.title('Imaginary part of DFT(I5)')

plt.show()



plt.figure(figsize=(8,6))

plt.subplot(2,2,1)
plt.text(0.1, 0.5, "Real(I5)\n\n" + str(np.real(I5)), fontsize=10)
plt.axis('off')
plt.subplot(2,2,2)
plt.text(0.1, 0.5, "Imag(I5)\n\n" + str(np.imag(I5)), fontsize=10)
plt.axis('off')
plt.subplot(2,2,3)
plt.text(0.1, 0.5, "Real(DFT(I5))\n\n" + str(np.real(Itilde5)), fontsize=10)
plt.axis('off')
plt.subplot(2,2,4)
plt.text(0.1, 0.5, "Imag(DFT(I5))\n\n" + str(np.imag(Itilde5)), fontsize=10)
plt.axis('off')

plt.show()
