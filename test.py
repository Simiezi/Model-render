from vizualize import print_plot
from convert import array_converter
from parse_file import file_parse
import numpy as np
import math
from core import *
import re


def vertexes_to_projective(vertexes):
    return np.concatenate([vertexes[:, :].copy(), np.ones(vertexes.shape[0]).reshape(-1, 1)], axis=1)



result_array = array_converter.sz_conv(
    file_parse.parse_vector('C:/Users/Kmondzy/Desktop/Компьютерная графика/graphics_task/african_head.obj', 'v'), 512)
f_array = file_parse.parse_place('C:/Users/Kmondzy/Desktop/Компьютерная графика/graphics_task/african_head.obj', 'v')

rot_matrix = np.array(
        [[np.cos(np.pi/6),  0, np.sin(np.pi/6),  0],
         [0,              1,       0,        0],
         [-np.sin(np.pi/6), 0, np.cos(np.pi/6),  0],
         [0,              0,       0,        1]],
        dtype=np.float64
    )

tr_matrix = np.array(
        [[1,    0,    0,   70],
         [0,    1,    0,    0],
         [0,    0,    1,   0],
         [0,    0,    0,       1]],
        dtype=np.float64
    )

t_matrix = np.array(
        [[0.8,   0,        0,   0],
         [0,    0.8,        0,   0],
         [0,       0,    0.8,   0],
         [0,       0,        0,   1]],
        dtype=np.float64
    )


affine_matrix = t_matrix.dot(rot_matrix)
# affine_matrix = tr_matrix.dot(affine_matrix)

result_array = np.around(vertexes_to_projective(result_array).dot(affine_matrix)[:, : 3]).astype(int)

result_array = np.around(tr_matrix.dot(vertexes_to_projective(result_array).T).T[:, :3]).astype(int)
print_plot.print_image(result_array, f_array, True)
