import numpy as np
import matplotlib.pyplot as plt

# Đọc hình ảnh xám từ tệp "lady.bin"
img = np.fromfile("img/ladybin.sec", dtype=np.uint8).reshape((256, 256))

# Vẽ lược đồ cường độ xám
plt.figure(figsize=(11, 10))
plt.subplot(2, 2, 1)
plt.hist(img.flatten(), bins=256, range=(0, 256), density=True, color='green', alpha=0.7)
plt.title("Histogram of Original Image")
plt.xlabel("Pixel Value")
plt.ylabel("Normalized Frequency")

# Hiển thị hình ảnh gốc
plt.subplot(2, 2, 2)
plt.imshow(img, cmap='gray')
plt.title("Original Image")

# Tìm giá trị cường độ tối thiểu và tối đa trên hình ảnh gốc
min_intensity = np.min(img)
max_intensity = np.max(img)

# Thực hiện tăng cường độ tương phản toàn phạm vi
contrast_stretched_image = (((img - min_intensity) / (max_intensity - min_intensity)) * 255).astype(np.uint8)

# Vẽ lược đồ cường độ xám sau tăng cường
plt.subplot(2, 2, 3)
plt.hist(contrast_stretched_image.flatten(), bins=256, range=(0, 256), density=True, color='green', alpha=0.7)
plt.title("Histogram of Contrast Stretched Image")
plt.xlabel("Pixel Value")
plt.ylabel("Normalized Frequency")

# Hiển thị hình ảnh sau tăng cường độ
plt.subplot(2, 2, 4)
plt.imshow(contrast_stretched_image, cmap='gray')
plt.title("Contrast Stretched Image")

plt.show()
