import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2


# Create figure
plt.figure(figsize=(12, 9))

# Read original image
OG_img = cv2.imread('img/bt1/moon.jpg', cv2.IMREAD_GRAYSCALE)

# Using CLAHE
CLH = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_image1 = CLH.apply(OG_img)

# Show OG image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(OG_img, cv2.COLOR_BGR2RGB))
plt.title('Original image')

# Show enhanced image
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(enhanced_image1, cv2.COLOR_BGR2RGB))
plt.title('Contrast limit histogram equalization')

# Show histogram for OG image
hist_original = cv2.calcHist([OG_img], [0], None, [256], [0, 256])
plt.subplot(2, 2, 3)
plt.bar(np.arange(256), hist_original.ravel())
plt.xlabel('Gray level')
plt.xlim([0, 255])
plt.title('Histogram of original image')

# Show histogram for enhanced image
hist_enhanced = cv2.calcHist([enhanced_image1], [0], None, [256], [0, 256])
plt.subplot(2, 2, 4)
plt.bar(np.arange(256), hist_enhanced.ravel())
plt.xlabel('Gray level')
plt.xlim([0, 255])
plt.title('Histogram of contrast limit histogram equalization')

# Show result
plt.show()
