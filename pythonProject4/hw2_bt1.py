import numpy as np
import matplotlib.pyplot as plt


# Create figure
plt.figure(figsize=(12, 9))

# a)Read file and show image
with open('img/bt2/lenabin.sec', 'rb') as lenabin:
    img1 = lenabin.read()
image1 = np.frombuffer(img1, dtype=np.uint8)

image2 = np.fromfile('img/bt2/peppersbin.sec', dtype=np.uint8)

# Reshape(256,256)
LENA = image1.reshape(256, 256)
PEPPERS = image2.reshape(256, 256)

# Show Lena image
plt.subplot(2, 2, 1)
plt.imshow(LENA, cmap='gray')
plt.title('Lena Image')

# Show Peppers image
plt.subplot(2, 2, 2)
plt.imshow(PEPPERS)
plt.gray()
plt.title('Peppers Image')



# b)Define J image
J = np.zeros((256, 256), dtype=np.uint8)
J[:, :128] = LENA[:, :128]
J[0:256, 128:256] = PEPPERS[0:256, 128:256]
plt.subplot(2, 2, 3)
plt.imshow(J)
plt.title('Image J')


# c)Define K image
K = np.zeros((256, 256), dtype=np.uint8)
K[:, 128:] = J[:, :128]
K[:, :128] = J[:, 128:]
plt.subplot(2, 2, 4)
plt.imshow(K)
plt.title('Image K')


# d)Show result
plt.show()
