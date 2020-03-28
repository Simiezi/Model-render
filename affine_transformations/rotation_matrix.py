import numpy as np
from parse_file import file_parse
from convert import array_converter
from vizualize import print_plot
from affine_transformations import tensile_matrix
from affine_transformations import transfer_matrix


def vertexes_to_projective(vertexes):
    return np.concatenate([vertexes[:, :].copy(), np.ones(vertexes.shape[0]).reshape(-1, 1)], axis=1)


def x_rotate(angle):
    rot_matrix = np.array(
        [[1,        0,          0,          0],
         [0, np.cos(angle), -np.sin(angle), 0],
         [0, np.sin(angle),  np.cos(angle), 0],
         [0,         0,          0,          1]],
        dtype=np.float64
    )
    return rot_matrix


def z_rotate(angle):
    rot_matrix = np.array(
        [[np.cos(angle), -np.sin(angle), 0, 0],
         [np.sin(angle), np.cos(angle),  0, 0],
         [0,                0,           1, 0],
         [0,                0,           0, 1]],
        dtype=np.float64
    )
    return rot_matrix


def y_rotate(angle):
    rot_matrix = np.array(
        [[np.cos(angle),  0, np.sin(angle),  0],
         [0,              1,       0,        0],
         [-np.sin(angle), 0, np.cos(angle),  0],
         [0,              0,       0,        1]],
        dtype=np.float64
    )
    return rot_matrix


def magic_func(vertexes, parameters, rotations):
    # temp_arr = np.zeros((4, 4), dtype=np.float64)
    if rotations == 'x':
        temp_arr = x_rotate(parameters[0][0])
        
    if rotations == 'y':
        temp_arr = y_rotate(parameters[0][1])

    if rotations == 'z':
        temp_arr = z_rotate(parameters[0][2])

    if rotations == 'xy':
        temp_arr = x_rotate(parameters[0][0])
        temp_arr = temp_arr.dot(y_rotate(parameters[0][1]))

    if rotations == 'yz':
        temp_arr = y_rotate(parameters[0][1])
        temp_arr = temp_arr.dot(z_rotate(parameters[0][2]))

    if rotations == 'xz':
        temp_arr = x_rotate(parameters[0][0])
        temp_arr = temp_arr.dot(z_rotate(parameters[0][2]))

    if rotations == 'xyz':
        temp_arr = x_rotate(parameters[0][0])
        temp_arr = temp_arr.dot(y_rotate(parameters[0][1]))
        temp_arr = temp_arr.dot(z_rotate(parameters[0][2]))

    temp_arr = tensile_matrix.tens_matrix(parameters[1][0], parameters[1][1], parameters[1][2]).dot(temp_arr)
    temp_arr = np.around(vertexes_to_projective(vertexes).dot(temp_arr)[:, : 3]).astype(int)
    return np.around(transfer_matrix.trans_matrix(parameters[2][0], parameters[2][1], parameters[2][2]).dot(vertexes_to_projective(temp_arr).T).T[:, :3]).astype(int)


