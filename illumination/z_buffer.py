import numpy as np
from vizualize import print_plot

def buffer(face_index, image_shape):
    z_buff = np.ones(image_shape)
    min_x =