import matplotlib.pyplot as plt
import numpy as np

COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))

u0 = 2
v0 = 2
I1 = 0.5 * np.exp(1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / 8)

Itilde1 = np.fft.fftshift(np.fft.fft2(I1))

print("Real part of I1:")
print(np.real(I1))
print("\nImaginary part of I1:")
print(np.imag(I1))

np.set_printoptions(precision=4, suppress=True)
print("\nReal part of DFT(I1):")
print(np.real(Itilde1))
print("\nImaginary part of DFT(I1):")
print(np.imag(Itilde1))

plt.figure(figsize=(8, 6))

plt.subplot(2,2,1)
plt.imshow(np.real(I1), cmap='gray')
plt.title('Real part of I1')

plt.subplot(2,2,2)
plt.imshow(np.imag(I1), cmap='gray')
plt.title('Imaginary part of I1')

plt.subplot(2,2,3)
plt.imshow(np.real(Itilde1), cmap='gray')
plt.title('Real part of DFT(I1)')

plt.subplot(2,2,4)
plt.imshow(np.imag(Itilde1), cmap='gray')
plt.title('Imaginary part of DFT(I1)')


plt.show()


plt.figure(figsize=(8,6))
plt.axis('off')


plt.subplot(2,2,1)
plt.text(0.1, 0.5, "Real(I1)\n\n" + str(np.real(I1)), fontsize=10)
plt.axis('off')
plt.subplot(2,2,2)
plt.text(0.1, 0.5, "Imag(I1)\n\n" + str(np.imag(I1)), fontsize=10)
plt.axis('off')
plt.subplot(2,2,3)
plt.text(0.1, 0.5, "Real(DFT(I1))\n\n" + str(np.real(Itilde1)), fontsize=10)
plt.axis('off')
plt.subplot(2,2,4)
plt.text(0.1, 0.5, "Imag(DFT(I1))\n\n" + str(np.imag(Itilde1)), fontsize=10)
plt.axis('off')

plt.show()
