import numpy as np
import cv2
import matplotlib.pyplot as plt

# Đọc hình ảnh "actontBin.bin"
image = np.fromfile("img/actontBinbin.sec", dtype=np.uint8).reshape((256, 256))

# Thiết kế template dựa trên hình "T"
template = np.array([[255, 255, 255],
                    [0, 255, 0],
                    [0, 255, 0]], dtype=np.uint8)





# Thực hiện Template Matching để tính toán match measure M2
J1 = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# Xây dựng hình ảnh đầu ra J1
# Đặt J1 bằng 0 tại các pixel nơi không có vùng lớn đủ để so sánh với template
threshold = 0.4  # Ngưỡng để xác định có vùng lớn đủ hay không
J1[J1 < threshold] = 0

# Ngưỡng hình ảnh J1 để tạo hình ảnh nhị phân J2
J2 = np.where(J1 > 0, 255, 0).astype(np.uint8)

# Hiện ảnh gốc
plt.figure(figsize=(12, 10))
plt.imshow(image, cmap='gray')
plt.title('OG Image')
plt.show()

# Hiển thị hình ảnh kết quả
cv2.imshow("Binary Image J2", J2)
cv2.waitKey(0)
cv2.destroyAllWindows()
