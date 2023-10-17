import cv2
import matplotlib.pyplot as plt


# a)
help(cv2.imread)
help(cv2.imwrite)

# Create figure
plt.figure(figsize=(12, 9))

# c) Read image lenagray
J1 = cv2.imread("img/bt2/lenagray.jpg", cv2.IMREAD_GRAYSCALE)
plt.subplot(1, 2, 1)
plt.imshow(J1)
plt.title('Image J1')

# d) Create image J2
J2 = 255 - J1
plt.subplot(1, 2, 2)
plt.imshow(J2)
plt.title('Image J2')

# Write image J2
cv2.imwrite('img/bt2/lenanegative.jpg', J2)

# e) Show result
plt.show()