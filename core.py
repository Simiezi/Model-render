import tkinter #Работа с GUI
import numpy as np #Работа с массивами
import matplotlib
import matplotlib.pyplot as plt #Вывод на экран
import random


root = tkinter.Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
pic_size = min(int(width/2), int(height/2))
print("Pic size: (%s,%s)" % (pic_size, pic_size))


def vertexes_to_projective(vertexes):
    return np.concatenate([vertexes[:, :2].copy(), np.ones(vertexes.shape[0]).reshape(-1, 1)], axis=1)


def show_image(image):
    plt.imshow(image.T, cmap="gray", interpolation="none")
    plt.show()


def prepare_image():
    img = np.zeros(shape=(pic_size+1,pic_size+1)).astype(np.uint8)
    # img = np.eye(pic_size+1)
    # img.fill(0)
    return img


def wireframe_render_float(img, vertexes, faces, color):
    draw_line_bad_float(img, vertexes[faces[0][0]][1], vertexes[faces[0][0]][0], vertexes[faces[0][1]][1], vertexes[faces[0][1]][0], color)
    draw_line_bad_float(img, vertexes[faces[0][1]][1], vertexes[faces[0][1]][0], vertexes[faces[0][2]][1], vertexes[faces[0][2]][0], color)
    draw_line_bad_float(img, vertexes[faces[0][0]][1], vertexes[faces[0][0]][0], vertexes[faces[0][2]][1], vertexes[faces[0][2]][0], color)
    draw_line_bad_float(img, vertexes[faces[1][0]][1], vertexes[faces[1][0]][0], vertexes[faces[1][1]][1], vertexes[faces[1][1]][0], color)
    draw_line_bad_float(img, vertexes[faces[1][0]][1], vertexes[faces[1][0]][0], vertexes[faces[1][2]][1], vertexes[faces[1][2]][0], color)
    draw_line_bad_float(img, vertexes[faces[1][1]][1], vertexes[faces[1][1]][0], vertexes[faces[1][2]][1], vertexes[faces[1][2]][0], color)


def draw_line_bad_float(img, x0, y0, x1, y1, color):
    steep = False
    #если ширина меньше высоты
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True
    #если первая координата больше второй
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(x0, x1):
        if (x1-x0) == 0:
            t = 0
        else:
            t = (x-x0) / (x1-x0)
        y = int(round(y0 * (1.-t) + y1 * t))
        #поменяли координаты, при отрисовке меняем обратно
        if x > 0 and y > 0 and x <= pic_size and y <= pic_size:
            if steep:
                img[x, y] = color
            else:
                img[y, x] = color

if __name__=="__main__":
    print(width, height)
