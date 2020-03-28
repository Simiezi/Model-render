import numpy as np
from affine_transformations import rotation_matrix


def viewport(model_vertexes, width, height):
    model_vertexes[:, 0] = ((width - 1) / 2) * model_vertexes[:, 0] + (width - 1) / 2
    model_vertexes[:, 1] = ((height - 1) / 2) * model_vertexes[:, 1] + (height - 1) / 2
    return model_vertexes


def orthogonal_projection(model_vertexes):
    x, y, z = model_vertexes[:, 0], model_vertexes[:, 1], model_vertexes[:, 2]
    l, r = np.min(x), np.max(x)
    b, t = np.min(y), np.max(x)
    n, f = np.min(z), np.max(z)
    ortho_matrix = np.array([[2 / (r - l), 0, 0, (-r - l) / (r - l)],
                             [0, 2 / (t - b), 0, (-t - b) / (t - b)],
                             [0, 0, -2 / (f - n), (-f - n) / (f - n)],
                             [0, 0, 0, 1]])
    model_vertexes = np.dot(ortho_matrix, rotation_matrix.vertexes_to_projective(model_vertexes).T)
    return np.around(model_vertexes.T[:, : 3]).astype(int)