# -*- coding: utf-8 -*-

import argparse
import os
import math
from distutils.util import strtobool

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from get_exif import get_pos
from trans3d import *
from osgeo import gdal, osr
import plotly.graph_objects as go

# 打开tif图像
def open_tif(path):
    dataset = gdal.Open(path)
    if dataset is None:
        raise IOError("Open file error:{}".format(path))
    cols = dataset.RasterXSize  # 图像长度
    rows = dataset.RasterYSize  # 图像宽度
    geotransform = dataset.GetGeoTransform()
    space = dataset.GetProjection()
    return dataset, cols, rows, geotransform, space

# 保存tif图像
def creat_tif(path, array, space, geotransform, datatype=None):
    driver = gdal.GetDriverByName("GTiff")
    shape = np.shape(array)
    if datatype is None:
        array_type = array.dtype
        if array_type == np.uint8:
            datatype = 1
        elif array_type == np.uint16:
            datatype = 2
        elif array_type == np.int16:
            datatype = 3
        elif array_type == np.uint32:
            datatype = 4
        elif array_type == np.int32:
            datatype = 5
        elif array_type == np.float32:
            datatype = 6
        elif array_type == np.float64:
            datatype = 7
        elif array_type == np.complex64:
            datatype = 10
        elif array_type == np.complex128:
            datatype = 11
        else:
            datatype = 6

    if len(shape) != 2 and len(shape) != 3:
        raise ValueError("Image shape must be 2D or 3D, but get shape{}".format(shape))

    if len(shape) == 2:
        s = [1, shape[0], shape[1]]
    else:
        s = shape

    outdata = driver.Create(path, s[2], s[1], s[0], datatype)
    if outdata == None:
        raise IOError("Creat file error:{}".format(path))
    if s[0] == 1:
        if len(shape) == 3 and shape[0] == 1:
            array = array[0]
        outdata.GetRasterBand(1).WriteArray(array)
    else:
        for i in range(s[0]):
            outdata.GetRasterBand(i + 1).WriteArray(array[i])

    if isinstance(space, int):
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(space)
        space = srs.ExportToWkt()
    outdata.SetProjection(space)  # 投影信息
    outdata.SetGeoTransform(geotransform)
    outdata.FlushCache()

# 模型可视化
class plot_points():
    def __init__(self):
        self.datas = []

    def plot_point_cloud(self, xy, color):
        l = np.shape(xy)[0]
        if l > 10000:
            i = np.linspace(0, l - 1, 10000, dtype=np.int32)
            xy = xy[i]
            if isinstance(color, np.ndarray):
                color = color[i]


        if np.shape(xy)[1] == 2:
            self.datas.append(
                go.Scatter3d(
                    x=xy[:, 0], y=xy[:, 1],
                    z=np.zeros_like(xy[:, 0]),
                    mode='markers',
                    marker=dict(size=2, color=color)
                )

            )
        else:
            self.datas.append(
                go.Scatter3d(
                    x=xy[:, 0], y=xy[:, 1],
                    z=xy[:, 2],
                    mode='markers',
                    marker=dict(size=2, color=color)
                )

            )

    def plot_lines_cloud(self, lines, color='red'):

        for l in lines:
            self.datas.append(
                go.Scatter3d(
                    x=l[0],
                    y=l[1],
                    z=l[2],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=3
                    )
                )
            )

    def show(self):
        self.fig = go.Figure(self.datas)
        self.fig.update_layout(showlegend=False)
        self.fig.show()



# 计算地面分辨率
def get_resolution(pix_size, pos):
    image_w = pos['ImageWidth']
    image_h = pos['ImageHeight']
    focal = pos['FocalLength']
    w = image_w * pix_size * 0.001
    h = image_h * pix_size * 0.001
    fh = pos['RelativeAltitude']
    gps_lat = pos['GpsLatitude']

    det_lat = 0.000008993
    det_lon = det_lat / math.cos(math.radians(gps_lat))

    #resolution_w = w / image_w / focal * fh * det_lon
    resolution_w = w / image_w / focal * fh * det_lat
    resolution_h = h / image_h / focal * fh * det_lat

    return (resolution_w, resolution_h)


# 创建格网
def creat_grid(geo_corners, resolution):
    # 计算栅格大小
    minx = min(geo_corners[:, 0])
    maxx = max(geo_corners[:, 0])
    miny = min(geo_corners[:, 1])
    maxy = max(geo_corners[:, 1])
    w = int((maxx - minx) // resolution[0])
    h = int((maxy - miny) // resolution[1])
    correct_image_size = (w, h)

    # 创建栅格仿射变换参数
    geotransform = (minx, resolution[0], 0.0, maxy, 0.0, -resolution[1])

    # 生成栅格坐标
    x = np.linspace(minx, maxx, w)
    y = np.linspace(maxy, miny, h)
    X, Y = np.meshgrid(x, y)
    Y = Y.flatten()
    X = X.flatten()

    points = np.array((X, Y)).T
    return correct_image_size, geotransform, points


# 获取图像角点84坐标
def image2wgs84(corners, R, relative_altitude, gps, show):
    if show:
        lines = []
        for i in range(np.shape(corners)[0]):
            x1 = [0, corners[i][0]]
            y1 = [0, corners[i][1]]
            z1 = [0, corners[i][2]]
            lines.append([x1, y1, z1])

            x2 = [corners[i - 1][0], corners[i][0]]
            y2 = [corners[i - 1][1], corners[i][1]]
            z2 = [corners[i - 1][2], corners[i][2]]

            lines.append([x2, y2, z2])

        plt.plot_lines_cloud(lines, color='blue')

    # 旋转
    rotated_corners = np.dot(corners, R)

    if show:
        lines = []
        for i in range(np.shape(rotated_corners)[0]):
            x1 = [0, rotated_corners[i][0]]
            y1 = [0, rotated_corners[i][1]]
            z1 = [0, rotated_corners[i][2]]
            lines.append([x1, y1, z1])

            x2 = [rotated_corners[i - 1][0], rotated_corners[i][0]]
            y2 = [rotated_corners[i - 1][1], rotated_corners[i][1]]
            z2 = [rotated_corners[i - 1][2], rotated_corners[i][2]]

            lines.append([x2, y2, z2])

        plt.plot_lines_cloud(lines, color='red')


    # 投影至地面
    ground_points = imgaxis2ground(rotated_corners, relative_altitude)

    if show:
        lines = []
        for i in range(np.shape(rotated_corners)[0]):
            x1 = [ground_points[i - 1][0], ground_points[i][0]]
            y1 = [ground_points[i - 1][1], ground_points[i][1]]
            z1 = [-relative_altitude, -relative_altitude]
            lines.append([x1, y1, z1])

            x2 = [ground_points[i][0], rotated_corners[i][0]]
            y2 = [ground_points[i][1], rotated_corners[i][1]]
            z2 = [-relative_altitude, rotated_corners[i][2]]

            lines.append([x2, y2, z2])

        plt.plot_lines_cloud(lines, color='red')


    # 转为84坐标
    gps_lon, gps_lat = gps
    det_lat = 0.000008993
    det_lon = det_lat / math.cos(math.radians(gps_lat))
    #ground_points[:, 0] = ground_points[:, 0] * det_lon
    ground_points[:, 0] = ground_points[:, 0] * det_lat
    ground_points[:, 1] = ground_points[:, 1] * det_lat

    geo_points = np.zeros_like(ground_points, dtype=np.float64)

    geo_points[:, 0] = gps_lon - ground_points[:, 1]
    geo_points[:, 1] = gps_lat + ground_points[:, 0]

    return geo_points


# 获取面方程
def coplanar_equation(points):
    x1, y1, z1 = points[0]
    x2, y2, z2 = points[1]
    x3, y3, z3 = points[2]

    # 计算面方程参数
    A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
    B = -((x2 - x1) * (z3 - z1) - (z2 - z1) * (x3 - x1))
    C = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    D = -(A * x1 + B * y1 + C * z1)

    return A, B, C, D

# 84坐标转为地面坐标
def wgs842ground(geo_points, gps, relative_altitude):
    # 转到地面坐标
    ground_points = np.zeros((geo_points.shape[0], 3))
    gps_lon, gps_lat = gps
    ground_points[:, 1] = gps_lon - geo_points[:, 0]
    ground_points[:, 0] = geo_points[:, 1] - gps_lat

    det_lat = 0.000008993
    det_lon = det_lat / math.cos(math.radians(gps_lat))
    #ground_points[:, 0] = ground_points[:, 0] / det_lon
    ground_points[:, 0] = ground_points[:, 0] / det_lat
    ground_points[:, 1] = ground_points[:, 1] / det_lat
    ground_points[:, 2] = - relative_altitude
    return ground_points


# 地面坐标投影至传感器坐标
def ground2sensor(ground_points, R, corners):
    '''
    # 转到地面坐标
    ground_points = np.zeros_like(geo_points, dtype=np.float64)
    gps_lon, gps_lat = gps
    ground_points[:, 1] = gps_lon - geo_points[:, 0]
    ground_points[:, 0] = geo_points[:, 1] - gps_lat

    det_lat = 0.000008993
    det_lon = det_lat / math.cos(math.radians(gps_lat))
    #ground_points[:, 0] = ground_points[:, 0] / det_lon
    ground_points[:, 0] = ground_points[:, 0] / det_lat
    ground_points[:, 1] = ground_points[:, 1] / det_lat
    '''
    # 地面点到像面投影
    rotated_corners = np.dot(corners, R)
    A, B, C, D = coplanar_equation(rotated_corners)
    plane_points = ground2imgaxis(ground_points, (A, B, C, D))

    # 旋转到传感器坐标
    inv_R = np.linalg.inv(R)
    image_points = np.dot(plane_points, inv_R)

    return image_points[:, (1, 2)]



def correct(image_path, output_path, pix_size, show):
    print(show)
    print(type(show))
    # 读取影像元数据
    data_dict = get_pos(image_path)

    # 读取影像
    dataset, image_w, image_h, geotransform, space = open_tif(image_path)
    img = dataset.ReadAsArray().transpose(1, 2, 0)

    # 获取旋转矩阵
    R = rotation_matrix(data_dict)

    # 读取相对高度
    if 'RelativeAltitude' in data_dict:
        relative_altitude = data_dict['RelativeAltitude']
    elif 'GpsAltitude' in data_dict:
        relative_altitude = data_dict['GpsAltitude']
        raise Warning('no RelativeAltitude, use GPSAltitude instead')
    else:
        raise ValueError(f"Can't get Altitude from file:{os.path.basename(image_path)}")

    # 初始化角点坐标
    corners = initial_corner(pix_size, data_dict)
    if show:
        img_points = initial_grid(pix_size, data_dict)
        plt.plot_point_cloud(img_points, color=img.reshape((-1, img.shape[2])))


    # 转换为投影坐标（单个影像范围较小，不考虑变形）
    if 'GpsLongitude' in data_dict:
        gps_lon = data_dict['GpsLongitude']
        gps_lat = data_dict['GpsLatitude']
    else:
        raise ValueError(f"Can't get GPS from image:{os.path.basename(image_path)}")

    gps = (gps_lon, gps_lat)

    # 角点投影至地面
    geo_corners = image2wgs84(corners, R, relative_altitude, gps, show)

    # 计算分辨率
    resolution = get_resolution(pix_size, data_dict)

    # 栅格化
    correct_image_size, geotransform, geo_points = creat_grid(geo_corners, resolution)

    # 栅格点反向投影
    ground_points = wgs842ground(geo_points, gps, relative_altitude)
    image_points = ground2sensor(ground_points, R, corners)

    # 重采样，获取栅格点值
    w = image_w * pix_size * 0.001
    h = image_h * pix_size * 0.001
    x_coords = np.linspace(w / 2, - w / 2, image_w)
    y_coords = np.linspace(h / 2, - h / 2, image_h)
    rgb = np.array(img).transpose((1, 0, 2))
    interpolator = RegularGridInterpolator(
        (x_coords, y_coords),
        rgb,
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    interpolated_values = interpolator(image_points)
    if show:
        plt.plot_point_cloud(ground_points, color=interpolated_values)

    # 重组栅格点
    interpolated_values = interpolated_values.T
    correct_image_w, correct_image_h = correct_image_size
    interpolated_values = interpolated_values.reshape((3, correct_image_h, correct_image_w))

    # 保存tif
    creat_tif(output_path, interpolated_values, 4326, geotransform, datatype=1)

    if show:
        plt.show()

if __name__ == '__main__':
    #python main.py --input_image D:\pycode\AerialPictureCorrection-master\dist\org3\DJI_0018.JPG --output_folder out --pixl_size 3.3 --show_3D_model False
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, help='image(with pos info) file or folder')
    parser.add_argument('--output_folder', type=str, help='output folder')
    parser.add_argument('--pixl_size', type=float, default=3.3, help='the sensor pixel size')
    parser.register('type', 'boolean', strtobool)
    parser.add_argument('--show_3D_model', type='boolean', default=False, help='show 3D image projection model')
    plt = plot_points()
    # 解析参数
    args = parser.parse_args()
    print('args.show_3D_model', args.show_3D_model)

    if os.path.isdir(args.input_image):

        images_list = os.listdir(args.input_image)

        for image_name in images_list:
            if os.path.splitext(image_name)[1].lower() in ['.jpg', '.jpeg', '.png']:
                image_path = os.path.join(args.input_image, image_name)
                tif_image_name = os.path.splitext(image_name)[0] + '.tif'
                output_path = os.path.join(args.output_folder, tif_image_name)
                correct(image_path, output_path, args.pixl_size, args.show_3D_model)
                print('矫正完成：', image_name)

    elif os.path.splitext(args.input_image)[1].lower() in ['.jpg', '.jpeg', '.png']:
        image_name = os.path.basename(args.input_image)
        tif_image_name = os.path.splitext(image_name)[0] + '.tif'
        output_path = os.path.join(args.output_folder, tif_image_name)
        correct(args.input_image, output_path, args.pixl_size, args.show_3D_model)
        print('矫正完成：', image_name)

    else:
        print('图像格式不正确！')








