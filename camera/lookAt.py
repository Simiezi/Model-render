import numpy as np
from affine_transformations import rotation_matrix


def camera_point_offset(point):
    cam_offset_matrix = np.array(
        [[1, 0, 0, -point[0]],
         [0, 1, 0, -point[1]],
         [0, 0, 1, -point[2]],
         [0, 0, 0,        1]],
        dtype=np.float64
    )
    return cam_offset_matrix


def new_cam_basis(right, up, forward):
    matrix = np.array(
        [[right[0], right[1], right[2], 0],
         [up[0], up[1], up[2], 0],
         [forward[0], forward[1], forward[2], 0],
         [0, 0, 0, 1]],
        dtype=np.float64
    )
    return matrix


def new_basis(cam_point, obj_point):
    forward_dir = (cam_point - obj_point) * (1.0 / np.linalg.norm(cam_point - obj_point))
    right_dir = (np.cross(np.array([0, 1, 0], dtype=np.float64), forward_dir)) * 1.0 / np.linalg.norm(np.cross(np.array([0, 1, 0], dtype=np.float64), forward_dir))
    up_dir = np.cross(forward_dir, right_dir)
    cam_matrix = new_cam_basis(right_dir, up_dir, forward_dir)
    return cam_matrix


def cam_view_matrix(model_vertexes, cam_vector, model_point):
    return np.around(camera_point_offset(cam_vector).dot(new_basis(cam_vector, model_point)).dot(rotation_matrix.vertexes_to_projective(model_vertexes).T).T[:, : 3]).astype(int)
