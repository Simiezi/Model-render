import numpy as np


def vertexes_to_projective(vertexes):
    return np.concatenate([vertexes[:, :].copy(), np.ones(vertexes.shape[0]).reshape(-1, 1)], axis=1)


def tens_matrix(alpha, beta, gamma):
    t_matrix = np.array(
        [[alpha,   0,        0,   0],
         [0,    beta,        0,   0],
         [0,       0,    gamma,   0],
         [0,       0,        0,   1]],
        dtype=np.float64
    )
    return t_matrix