import cv2
import matplotlib.pyplot as plt


# Create figure
plt.figure(figsize=(12, 9))

# b) Read image
J1 = cv2.imread('img/bt2/lena512color.jpg', cv2.IMREAD_COLOR)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(J1, cv2.COLOR_BGR2RGB))
plt.title('Image J1')

# c) Create image J2
J2 = J1.copy()
J2[:, :, 0] = J1[:, :, 2]
J2[:, :, 1] = J1[:, :, 0]
J2[:, :, 2] = J1[:, :, 1]

# d) Show J2 and write it
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(J2, cv2.COLOR_BGR2RGB))
plt.title('Image J2')
cv2.imwrite('img/bt2/lenaswapcolor.jpg', J2)

# e) Show result
plt.show()