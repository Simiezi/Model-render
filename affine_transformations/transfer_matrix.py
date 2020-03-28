import numpy as np


def vertexes_to_projective(vertexes):
    return np.concatenate([vertexes[:, :].copy(), np.ones(vertexes.shape[0]).reshape(-1, 1)], axis=1)


def trans_matrix(alpha, beta, gamma):
    tr_matrix = np.array(
        [[1,    0,    0,   alpha],
         [0,    1,    0,    beta],
         [0,    0,    1,   gamma],
         [0,    0,    0,       1]],
        dtype=np.float64
    )
    return tr_matrix