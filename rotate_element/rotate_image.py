from core import *
import numpy as np
import math

kv1_1 = np.array([200, 200])
kv1_3 = np.array([400, 200])
kv1_4 = np.array([200, 400])
kv2_2 = np.array([200, 200])
kv2_3 = np.array([100, 200])
kv2_4 = np.array([200, 100])
vertexes = np.array([kv1_1, kv1_3, kv1_4, kv2_2, kv2_3, kv2_4])
faces = np.array([[0, 1, 2], [3, 4, 5]])

matrix = np.zeros((3, 3), dtype=float)
matrix[0, 0] = math.cos(math.pi/6)
matrix[0, 1] = -math.sin(math.pi/6)
matrix[1, 0] = math.sin(math.pi/6)
matrix[1, 1] = math.cos(math.pi/6)
matrix[2, 2] = 1
#
pr_vert = vertexes_to_projective(vertexes)
pr_vert = pr_vert.dot(matrix)
print(pr_vert)
pr_vert = np.array(np.around(pr_vert).astype(int))
image = prepare_image()
wireframe_render_float(image, pr_vert, faces, 255)
show_image(image)