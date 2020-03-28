import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from vizualize import print_plot
from parse_file import file_parse

def phong_illumination(model_vertexes, vision_dir, faces, light_dir, texture_v, normal_v, normal_faces, texture_faces, image, img):
    distance = light_dir / np.linalg.norm(light_dir)
    vision = vision_dir / np.linalg.norm(vision_dir)
    z_buff = np.full((512, 512), -np.inf)
    width = image.size[0]
    height = image.size[1]
    pix = image.load()
    for i in range(len(faces)):
        frst_point = model_vertexes[faces[i][0] - 1]
        sec_point = model_vertexes[faces[i][1] - 1]
        thrd_point = model_vertexes[faces[i][2] - 1]
        frst_texture = texture_v[texture_faces[i][0] - 1]
        sec_texture = texture_v[texture_faces[i][1] - 1]
        thrd_texture = texture_v[texture_faces[i][2] - 1]
        frst_norm = normal_v[normal_faces[i][0] - 1]
        sec_norm = normal_v[normal_faces[i][1] - 1]
        thrd_norm = normal_v[normal_faces[i][2] - 1]
        normal = np.cross(sec_point - frst_point,
                          thrd_point - frst_point)
        normal = normal / np.linalg.norm(normal)
        back_face = np.dot(normal.T, np.array([0, 0, -1]))
        face_points = np.array([[frst_point[0], frst_point[1]],
                                [sec_point[0], sec_point[1]],
                                [thrd_point[0], thrd_point[1]]])
        min_point = np.array([min(frst_point[0], min(sec_point[0], thrd_point[0])),
                              min(frst_point[1], min(sec_point[1], thrd_point[1]))])
        max_point = np.array([max(frst_point[0], max(sec_point[0], thrd_point[0])),
                              max(frst_point[1], max(sec_point[1], thrd_point[1]))])
        if back_face < 0:
            for y in range(min_point[1], max_point[1]):
                for x in range(min_point[0], max_point[0]):
                    bar_coords = print_plot.barycentric(face_points, x, y)
                    if bar_coords[0] < 0 or bar_coords[1] < 0 or bar_coords[2] < 0:
                        continue
                    elif 0 <= x and 0 <= y:
                        z = bar_coords[0] * frst_point[2] + bar_coords[1] * sec_point[2] + bar_coords[2] * thrd_point[2]
                        if z > z_buff[512 - y - 1, 512 - x - 1]:
                            norm = np.array([bar_coords[0] * frst_norm[0] + bar_coords[1] * sec_norm[0] + bar_coords[2] * thrd_norm[0],
                                             bar_coords[0] * frst_norm[1] + bar_coords[1] * sec_norm[1] + bar_coords[2] * thrd_norm[1],
                                             bar_coords[0] * frst_norm[2] + bar_coords[1] * sec_norm[2] + bar_coords[2] * thrd_norm[2]])
                            u = (bar_coords[0] * frst_texture[0] + bar_coords[1] * sec_texture[0] + bar_coords[2] *
                                 thrd_texture[0]) * width
                            v = (bar_coords[0] * frst_texture[1] + bar_coords[1] * sec_texture[1] + bar_coords[2] *
                                 thrd_texture[1]) * height
                            r = pix[width - u, height - v][0]
                            g = pix[width - u, height - v][1]
                            b = pix[width - u, height - v][2]

                            phong_diff = 0.7 * np.dot(norm, distance) + 0.3 * np.power(2 * np.dot(np.dot(norm, distance), np.dot(norm, vision)) - np.dot(distance, vision), 4)
                            if phong_diff < 0:
                                phong_diff = 0
                            img[512 - y - 1, 512 - x - 1] = phong_diff * np.array([r, g, b], dtype=np.float64)
                            z_buff[512 - y - 1, 512 - x - 1] = z
                        else:
                            continue
    plt.imshow(img)
    plt.show()
