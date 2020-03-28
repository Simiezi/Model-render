import numpy as np

"""
    Масштабируем картинку, используя максимумы и минимумы x и y
                                                                """


def sz_conv(array, size):
    temp = array
    temp[:, 0] -= temp[:, 0].min()
    temp[:, 1] -= temp[:, 1].min()
    coefficient = size / max(temp[:, 0].max(), temp[:, 1].max())
    result = np.around(temp * coefficient)
    result = np.array(result, dtype='int32')
    return result
