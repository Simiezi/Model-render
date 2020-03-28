from parse_file import file_parse
from convert import array_converter
from vizualize import print_plot
from affine_transformations import rotation_matrix
from camera import lookAt
from viewsss import view_projections
from illumination import lambert, phong
from PIL import Image

import numpy as np


if __name__ == "__main__":
    file_pth = 'C:/Users/adelk/Desktop/Универ/Компьютерная графика/graphics_task/african_head.obj'
    result_array = array_converter.sz_conv(
        file_parse.parse_vector(file_pth, 'v'), 512)
    f_array = file_parse.parse_place(file_pth, 'v')
    texture_v = file_parse.parse_vector(file_pth,
                                        'vt')
    texture_faces = file_parse.parse_place(
        file_pth, 'vt')
    normal_v = file_parse.parse_vector(file_pth,
                                       'vn')
    normal_faces = file_parse.parse_place(
        file_pth, 'vn')
    image = Image.open('C:/Users/adelk/Desktop/Универ/Компьютерная графика/graphics_task/african_head_diffuse.tga')
    cam_point = np.array([-60, 10, 650], dtype=np.float64)
    imag = np.zeros((512, 512, 3), dtype=np.uint8)
    params = [[120, np.pi/12, 120], [1, 1, 1], [40, 0, 0]]
    result_array = rotation_matrix.magic_func(result_array, params, 'y')
    result_array = lookAt.cam_view_matrix(result_array,
                                          cam_point,
                                          np.array([0,
                                                    0,
                                                    0], dtype=np.float64))
    # lambert.lambert_illumination(result_array, f_array, np.array([-60, 10, 650], dtype=np.float64), True, True)
    # result_array = view_projections.orthogonal_projection(result_array)
    # result_array = view_projections.viewport(result_array, 512, 512)
    # phong.phong_illumination(result_array, cam_point,
    #                          f_array, np.array([-100, 80, 650], dtype=np.float64), texture_v, normal_v, normal_faces, texture_faces, image, imag)
    lambert.lambert_illumination(result_array, f_array, np.array([-60, 10, 650], dtype=np.float64), True, True, texture_v, texture_faces, image,imag)
    # print_plot.print_image(result_array, f_array, True)
