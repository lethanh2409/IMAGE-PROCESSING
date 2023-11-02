import numpy as np
import matplotlib.pyplot as plt

I6 = np.fromfile('camerabin.sec', dtype=np.uint8).reshape(256, 256)

J1 = np.abs(I6)
J2 = np.angle(I6)

plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plt.title('J2: DFT Phase Contribution')
plt.imshow(J2, cmap='gray', vmin=-np.pi, vmax=np.pi)
plt.axis('off')

JJ1 = np.log(J1 + 1e-10)

plt.subplot(1,2,2)
plt.title('JJ1: Log(DFT Magnitude Contribution)')
plt.imshow(JJ1, cmap='gray')
plt.axis('off')

plt.show()