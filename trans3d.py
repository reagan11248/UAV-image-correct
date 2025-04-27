
import numpy as np

# 初始化图像角点
def initial_corner(pix_size, pos):
    f = pos['FocalLength']
    image_w = pos['ImageWidth']
    image_h = pos['ImageHeight']
    w = image_w * pix_size * 0.001
    h = image_h * pix_size * 0.001
    corner = np.array([[f, w / 2, h / 2],
                       [f, - w / 2, h / 2],
                       [f, - w / 2, - h / 2],
                       [f, w / 2, - h / 2]])
    return corner

def initial_grid(pix_size, pos):
    f = pos['FocalLength']
    img_w = pos['ImageWidth']
    img_h = pos['ImageHeight']
    w = img_w * pix_size * 0.001
    h = img_h * pix_size * 0.001

    y = np.linspace(w / 2, - w / 2, img_w)
    z = np.linspace(h / 2, - h / 2, img_h)
    Y, Z = np.meshgrid(y, z)

    Y = Y.flatten()
    Z = Z.flatten()
    X = np.zeros_like(Y)
    X[:] = f
    grid = np.array((X, Y, Z)).T

    return grid


# 计算旋转矩阵
def rotate(roll, pitch, yaw):
    # 绕X轴的旋转矩阵（滚转角）
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), np.sin(roll)],
                   [0, -np.sin(roll), np.cos(roll)]])

    # 绕Y轴的旋转矩阵（俯仰角）
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    # 绕Z轴的旋转矩阵（偏航角）
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    R = np.dot(Rx, np.dot(Ry, Rz))
    return R


# 根据pos数据计算旋转矩阵
def rotation_matrix(pos):
    gr = pos['GimbalRollDegree']
    gp = pos['GimbalPitchDegree']
    gy = pos['GimbalYawDegree']

    g_roll = np.radians(gr)
    g_pitch = np.radians(gp)
    g_yaw = np.radians(gy)

    GR = rotate(g_roll, g_pitch, g_yaw)
    return GR


# 图像点投影到地面
def imgaxis2ground(image_point, relative_altitude):
    t = - relative_altitude / image_point[:, 2]
    x = image_point[:, 0] * t
    y = image_point[:, 1] * t
    return np.array([x, y]).transpose((1, 0))

# 地面点投影到像面
def ground2imgaxis(ground_points, plant_equation):
    x1 = ground_points[:, 0]
    y1 = ground_points[:, 1]
    z1 = ground_points[:, 2]

    A, B, C, D = plant_equation
    t = - D / (A * x1 + B * y1 + C * z1)

    x = t * x1
    y = t * y1
    z = t * z1
    return np.array((x, y, z)).T







if __name__ == '__main__':
    pass










