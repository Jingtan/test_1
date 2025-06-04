import cv2
import numpy as np
import glob
import json

"""
利用存放在文件夹dh_img各子文件夹下peremeters.json文件中的标定参数，对各相机进行标定。
"""



name = '/2025-04-07_17_13_18_506.bmp'   # 待标定图像
# name_1 = 'GCK23040245'
# name_1 = 'GCK24130223'
name_1 = 'GCK24130015'
# name_1 = 'GCK24130224'
path = 'dh_img\\' + name_1 # 标定参数位置
path_1 = 'dh_imgtest_in\\' + name_1  # 待标定图像位置
path_2 = 'dh_imgtest_out\\' + name_1  # 标定后的图像输出位置


with open(path+ "/peremeters.json", 'r') as file:
    data = json.load(file)
print(f'data:{data}')
mtx = data['mtx']
mtx = np.array(mtx)
dist = data['dist']
dist = np.array(dist)
img2 = cv2.imread(path_1 + name)
print(f'img2:{img2}')

# w = 8
# h = 11
h, w = img2.shape[:2]
# 我们已经得到了相机内参和畸变系数，在将图像去畸变之前，
# 我们还可以使用cv.getOptimalNewCameraMatrix()优化内参数和畸变系数，
# 通过设定自由自由比例因子alpha。当alpha设为0的时候，
# 将会返回一个剪裁过的将去畸变后不想要的像素去掉的内参数和畸变系数；
# 当alpha设为1的时候，将会返回一个包含额外黑色像素点的内参数和畸变系数，并返回一个ROI用于将其剪裁掉
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))  # 自由比例参数

dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
# 根据前面ROI区域裁剪图片
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
print('test_1')
# cv2.imwrite(path_2 + name, dst)
cv2.imwrite(path_2 + name, dst)
