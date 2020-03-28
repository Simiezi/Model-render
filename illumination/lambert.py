import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from vizualize import print_plot
from parse_file import file_parse


def lambert_illumination(model_vertexes, fv_array, light_dir, back_face_culling, z_buffer, texture_v, texture_faces, image, img):
    distance = light_dir / np.linalg.norm(light_dir)
    z_buff = np.full((512, 512), -np.inf)
    width = image.size[0]
    height = image.size[1]
    pix = image.load()
    for i in range(len(fv_array)):
        frst_point = model_vertexes[fv_array[i][0] - 1]
        sec_point = model_vertexes[fv_array[i][1] - 1]
        thrd_point = model_vertexes[fv_array[i][2] - 1]
        frst_texture = texture_v[texture_faces[i][0] - 1]
        sec_texture = texture_v[texture_faces[i][1] - 1]
        thrd_texture = texture_v[texture_faces[i][2] - 1]
        normal = np.cross(sec_point - frst_point,
                          thrd_point - frst_point)
        normal = normal / np.linalg.norm(normal)
        back_face = np.dot(normal.T, np.array([0, 0, -1]))
        diffuse = np.dot(normal.T, distance)
        if diffuse < 0:
            diffuse = 0
        face_points = np.array([[frst_point[0], frst_point[1]],
                                [sec_point[0], sec_point[1]],
                                [thrd_point[0], thrd_point[1]]])
        min_point = np.array([min(frst_point[0], min(sec_point[0], thrd_point[0])),
                              min(frst_point[1], min(sec_point[1], thrd_point[1]))])
        max_point = np.array([max(frst_point[0], max(sec_point[0], thrd_point[0])),
                              max(frst_point[1], max(sec_point[1], thrd_point[1]))])
        if back_face_culling:
            if back_face < 0:
                if z_buffer:
                    for y in range(min_point[1], max_point[1]):
                        for x in range(min_point[0], max_point[0]):
                            bar_coords = print_plot.barycentric(face_points, x, y)
                            if bar_coords[0] < 0 or bar_coords[1] < 0 or bar_coords[2] < 0:
                                continue
                            elif 0 <= x and 0 <= y:
                                z = bar_coords[0] * frst_point[2] + bar_coords[1] * sec_point[2] + bar_coords[2] * thrd_point[2]
                                if z > z_buff[512 - y - 1, 512 - x - 1]:
                                    u = (bar_coords[0] * frst_texture[0] + bar_coords[1] * sec_texture[0] + bar_coords[2] * thrd_texture[0]) * width
                                    v = (bar_coords[0] * frst_texture[1] + bar_coords[1] * sec_texture[1] + bar_coords[2] * thrd_texture[1]) * height
                                    r = pix[width - u, height - v][0]
                                    g = pix[width - u, height - v][1]
                                    b = pix[width - u, height - v][2]
                                    img[512 - y - 1, 512 - x - 1] = 0.8 * 0.5 * diffuse * np.array([r, g, b], dtype=np.float64)
                                    z_buff[512 - y - 1, 512 - x - 1] = z
                            else:
                                continue
                else:
                    print_plot.raster(img, frst_point[0],
                                      frst_point[1],
                                      sec_point[0],
                                      sec_point[1],
                                      thrd_point[0],
                                      thrd_point[1],
                                      255 * diffuse,
                                      255 * diffuse,
                                      255 * diffuse)
    plt.imshow(img)
    plt.show()


