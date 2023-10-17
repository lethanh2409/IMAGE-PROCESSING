import cv2
import matplotlib.pyplot as plt
from skimage import exposure


# Create figure
plt.figure(figsize=(12, 9))

# OG image
OG_img = cv2.imread('img/bt1/parrot.jpg', cv2.IMREAD_GRAYSCALE)
plt.subplot(2, 2, 1)
plt.imshow(OG_img, cmap='gray')
plt.title('Original image')

# OG HE
eqHistImg = cv2.equalizeHist(OG_img)
plt.subplot(2, 2, 2)
plt.imshow(eqHistImg, cmap='gray')
plt.title('Original histogram equalization')

# AHE 8x8
ahe_img1 = exposure.equalize_adapthist(OG_img, kernel_size=(8, 8))
plt.subplot(2, 2, 3)
plt.imshow(ahe_img1, cmap='gray')
plt.title('Adaptive histogram equalization 8x8 tiles')

# AHE 16x16
ahe_img2 = exposure.equalize_adapthist(OG_img, kernel_size=(16, 16))
plt.subplot(2, 2, 4)
plt.imshow(ahe_img2, cmap='gray')
plt.title('Adaptive histogram equalization 16x16 tiles')

# Show result
plt.show()
