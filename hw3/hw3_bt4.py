import numpy as np
import matplotlib.pyplot as plt

# Đọc hình ảnh từ tệp "johnny.bin"
image = np.fromfile("img/johnnybin.sec", dtype=np.uint8).reshape((256, 256))

# Tạo histogram của hình ảnh gốc
hist_original, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))

# Tính tổng tích lũy của histogram
cdf = hist_original.cumsum()

# Chuyển đổi CDF thành phép ánh xạ mới
cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
cdf_normalized = cdf_normalized.astype(np.uint8)

# Sử dụng phép ánh xạ để biến đổi hình ảnh gốc
equalized_image = cdf_normalized[image]

# Tạo histogram của hình ảnh đã cân bằng
hist_equalized, _ = np.histogram(equalized_image.flatten(), bins=256, range=(0, 256))

plt.figure(figsize=(12, 10))
# Hiển thị hình ảnh gốc
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('OG Image')

# Hiển thị hình ảnh đã cân bằng
plt.subplot(2, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

# Vẽ biểu đồ histogram của hình ảnh gốc
plt.subplot(2, 2, 3)
plt.hist(image.flatten(), bins=256, range=(0, 256), density=True, color='g', alpha=0.6)
plt.xlabel('Pixel Value')
plt.ylabel('Normalized Frequency')
plt.title('Histogram of Original Image')

# Vẽ biểu đồ histogram của hình ảnh đã cân bằng
plt.subplot(2, 2, 4)
plt.hist(equalized_image.flatten(), bins=256, range=(0, 256), density=True, color='r', alpha=0.6)
plt.xlabel('Pixel Value')
plt.ylabel('Normalized Frequency')
plt.title('Histogram of Equalized Image')

plt.show()