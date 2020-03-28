import numpy as np
import re


"""
    Парсим файл, используя регулярные выражения, берем оттуда координаты x и y, 
    возвращаем результат в виде массива Numpy
                                                                                """


def parse_vector(filepath, mode):
    file = open(filepath, 'r')
    array = file.read().split('\n')
    res_arr = []
    if mode == 'v':
        for line in array:
            if 'f' in line:
                continue
            elif 'vn' in line:
                continue
            elif 'vt' in line:
                continue
            else:
                res_arr.append(np.fromstring(line.replace('v ', ''), dtype=np.float64, sep=' '))
    if mode == 'vt':
        for line in array:
            if 'f' in line:
                continue
            elif line.split(' ')[0] == 'v':
                continue
            elif 'vn' in line:
                continue
            else:
                res_arr.append(np.fromstring(line.replace('vt ', ''), dtype=np.float64, sep=' '))
    if mode == 'vn':
        for line in array:
            if 'f' in line:
                continue
            elif line.split(' ')[0] == 'v':
                continue
            elif 'vt' in line:
                continue
            else:
                res_arr.append(np.fromstring(line.replace('vn ', ''), dtype=np.float64, sep=' '))
    return np.array(res_arr, dtype=np.float64)


"""
     Парсим числа f из файла
                                """


def parse_place(filepath, mode):
    if mode == 'v':
        index = 0
    elif mode == 'vt':
        index = 1
    else:
        index = 2
    file = open(filepath, 'r')
    array = (''.join(file.read().split('f ')[1:])).split('\n')
    parse_data = []
    for line in array:
        temp_arr = re.findall(r"\d+", line)[index::3]
        if temp_arr != []:
            parse_data.append(temp_arr)
    return np.array(parse_data, dtype="int32")
