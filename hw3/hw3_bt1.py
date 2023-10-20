import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# a)
plt.figure(figsize=(10, 10))

# Đọc hình ảnh
with open('img/Mammogrambin (2).sec', 'rb') as mammo:
    img1 = mammo.read()
    image = np.frombuffer(img1, dtype=np.uint8).reshape(256, 256)
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('OG Image')

# Xác định ngưỡng
nguong = 128  # Điều chỉnh theo nhu cầu

# Áp dụng chuyển đổi ngưỡng
bin_image = np.where(image > nguong, 255, 0).astype(np.uint8)
plt.subplot(2, 2, 2)
plt.imshow(bin_image, cmap='gray')
plt.title('Binary Image')

# Lưu hình ảnh
image_to_save = Image.fromarray(bin_image)
image_to_save.save("img/BinaryMammogram.png")


# b)
# Tạo hình ảnh đường viền nhị phân
contour_image = cv2.Canny(bin_image, 108, 148)  # Điều chỉnh ngưỡng theo nhu cầu

# Lưu hình ảnh đường viền
cv2.imwrite("ContourMammogram.png", contour_image)
plt.subplot(2, 2, 3)
plt.imshow(contour_image, cmap='gray')
plt.title('Contour Image')

plt.show()
