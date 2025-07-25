import cv2
import numpy as np
import glob
import json

'''
此程序对四个相机：GCK24130224，GCK23040245，GCK24130223，GCK24130015进行标定，
获取每个相机的标定参数并保存为peremeters.json文件，存放在文件夹dh_img中的各相机文件夹中。
'''

# path = 'dh_img\GCK24130224'
# path = 'dh_img\GCK23040245'
# path = 'dh_img\GCK24130223'
path = 'dh_img\GCK24130015'  # 标定板图像位置
# name = '/2025-04-07_15_16_53_158.bmp'
# 找棋盘格角点
# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 阈值
# 棋盘格模板规格 内角点数目
w = 8
h = 11
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objp = objp * 35  # 35mm

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点
# 加载文件夹下所有的jpg图像
images = glob.glob(path + '/*.bmp')  # 拍摄的十几张棋盘图片所在目录
print(f'images:{images}')
# image = cv2.imread(images[0])
# cv2.imshow('test', image)
# cv2.waitKey()


i = 0
for fname in images:
    print('test')
    img = cv2.imread(fname)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # 获取画面中心点
    # 获取图像的长宽
    h1, w1 = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('igray', gray)
    # cv2.waitKey()
    # print(f'gray:{gray}')
    u, v = img.shape[:2]
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        print("i:", i)
        i = i + 1
        # 在原角点的基础上寻找亚像素角点
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # 追加进入世界三维点和平面二维点中
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 640, 480)
        cv2.imshow('findCorners', img)
        # cv2.waitKey()
    else:
        print("标定失败", fname)
cv2.destroyAllWindows()
print('正在计算')
# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("ret标定误差:", ret)  # 标定误差 0.1-0.5
print("mtx内参数矩阵:\n", mtx.reshape(-1).tolist())  # 内参数矩阵
print("dist畸变值:\n", dist.reshape(-1).tolist())  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs旋转（向量）外参:\n", rvecs)  # 旋转向量  # 外参数
print("tvecs平移（向量）外参:\n", tvecs)  # 平移向量  # 外参数

# 写入文本中
data = {}
data['mtx'] = mtx.tolist()
data['dist'] = dist.tolist()
print(f'data:{data}')
with open(path+ '/peremeters.json', 'w') as file:
    json.dump(data, file, indent=4)
