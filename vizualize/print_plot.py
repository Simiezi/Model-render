import random
import math

import matplotlib.pyplot as plt
import numpy as np


from parse_file import file_parse
from convert import array_converter
"""
    Рисуем картинку на черном фоне
                                    """


class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def vector_mul(self, vec):
        x = self.y * vec.z - self.z * vec.y
        y = self.z * vec.x - self.x * vec.z
        z = self.x * vec.y - self.y * vec.x
        return Point3D(x, y, z)

    def norm(self):
        return float(math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z))

    def normalize(self):
        return Point3D(self.x * 1./self.norm(), self.y * 1./self.norm(), self.z * 1./self.norm())

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def print_image(vector_array, f_array, breesenham, image=np.zeros((512, 512, 3), dtype=np.uint8)):
    if breesenham is True:
        for first, second, third in f_array:
            line(image, vector_array[first - 1, 0], vector_array[first - 1, 1], vector_array[second - 1, 0], vector_array[second - 1, 1])
            line(image, vector_array[first - 1, 0], vector_array[first - 1, 1], vector_array[third - 1, 0], vector_array[third - 1, 1])
            line(image, vector_array[third - 1, 0], vector_array[third - 1, 1], vector_array[second - 1, 0], vector_array[second - 1, 1])
    else:
        for first, second, third in f_array:
            raster(image, vector_array[first - 1, 0],
                   vector_array[first - 1, 1],
                   vector_array[second - 1, 0],
                   vector_array[second - 1, 1],
                   vector_array[third - 1, 0],
                   vector_array[third - 1, 1],
                   random.randint(1, 255),
                   random.randint(1, 255),
                   random.randint(1, 255))
    plt.imshow(image)
    plt.show()


def line(img, x0, y0, x1, y1):
    steep = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    dy = abs(y1 - y0)
    signy = -1 if y0 > y1 else 1
    error = 0
    y = y0
    for x in range(x0, x1):
        if steep:
            if x > 0 and y > 0:
                img[512 - x - 1, 512 - y - 1] = 255
        else:
            if x > 0 and y > 0:
                img[512 - y - 1, 512 - x - 1] = 255
        error += dy
        if abs(x1 - x0) <= 2 * error:
            y += signy
            error -= abs(x1 - x0)


def raster(img, x0, y0, x1, y1, x2, y2, red, green, blue):
    pts = [[x0, y0], [x1, y1], [x2, y2]]
    triangle(pts, img, red, green, blue)



def barycentric(point, x, y):
    A = Point3D(point[2][0] - point[0][0], point[1][0] - point[0][0], point[0][0] - x)
    B = Point3D(point[2][1] - point[0][1], point[1][1] - point[0][1], point[0][1] - y)
    u = Point3D(A.y * B.z - B.y * A.z, -1 * (A.x * B.z - B.x * A.z), A.x * B.y - B.x * A.y)
    return np.array([1.0 - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z])


def triangle(point, image, red, green, blue):
    minPoint = Point(min(point[0][0], min(point[1][0], point[2][0])), min(point[0][1], min(point[1][1], point[2][1])))
    maxPoint = Point(max(point[0][0], max(point[1][0], point[2][0])), max(point[0][1], max(point[1][1], point[2][1])))
    for y in range(minPoint.y, maxPoint.y):
        for x in range(minPoint.x, maxPoint.x):
            temp = barycentric(point, x, y)
            if temp[0] < 0 or temp[1] < 0 or temp[2] < 0:
                continue
            elif x > 0 and y > 0:
                image[512 - y, 512 - x] = [red, green, blue]
            else:
                continue

def print_african(image, fv_array, fn_array, vector_array, vn_array):
    light_dir = Point3D(0, 0, -1)
    for i in range(len(fv_array)):
        first_normal = np.array(vn_array[fn_array[i][0] - 1, 0],
                               vn_array[fn_array[i][0] - 1, 1],
                               vn_array[fn_array[i][0] - 1, 2])

        second_normal = np.array(vn_array[fn_array[i][1] - 1, 0],
                                vn_array[fn_array[i][1] - 1, 1],
                                vn_array[fn_array[i][1] - 1, 2])

        third_normal = np.array(vn_array[fn_array[i][2] - 1, 0],
                               vn_array[fn_array[i][2] - 1, 1],
                               vn_array[fn_array[i][2] - 1, 2])

        frst_sec_mul = np.cross(first_normal, second_normal)
        normal = np.cross(frst_sec_mul, third_normal)
        
        # normal = temp_normal.normalize()
        
        first_vertex = Point(vector_array[fv_array[i][0] - 1, 0],
                             vector_array[fv_array[i][0] - 1, 1])

        second_vertex = Point(vector_array[fv_array[i][1] - 1, 0],
                              vector_array[fv_array[i][1] - 1, 1])

        third_vertex = Point(vector_array[fv_array[i][2] - 1, 0],
                             vector_array[fv_array[i][2] - 1, 1])
        intensity = normal[0] * light_dir.x + normal[1] * light_dir.y + normal[2] * light_dir.z
        if intensity < 0:
            raster(image,
                   first_vertex.x,
                   first_vertex.y,
                   second_vertex.x,
                   second_vertex.y,
                   third_vertex.x,
                   third_vertex.y,
                   intensity * 255, intensity * 255, intensity * 255)
    plt.imshow(image)
    plt.show()


# result_array = array_converter.sz_conv(
#         file_parse.parse_vector('C:/Users/Kmondzy/Desktop/Компьютерная графика/graphics_task/african_head.obj', 'v'), 400)
# print_african(np.zeros((600, 600, 3), dtype=np.uint8),
#               file_parse.parse_place('C:/Users/Kmondzy/Desktop/Компьютерная графика/graphics_task/african_head.obj', 'v'),
#               file_parse.parse_place('C:/Users/Kmondzy/Desktop/Компьютерная графика/graphics_task/african_head.obj', 'vn'),
#               result_array,
#               file_parse.parse_vector('C:/Users/Kmondzy/Desktop/Компьютерная графика/graphics_task/african_head.obj', 'vn'))