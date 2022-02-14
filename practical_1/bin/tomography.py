import os
import sys

import numpy as ny
from numpy import ndarray as arr, minimum as mini, maximum as maxi
import cv2
import scipy.io

ny.set_printoptions(threshold=sys.maxsize)
MAX_ITER: int = 150
EPSILON: float = 0.05

IMG_SET: list = [('oval', 'png'), ('rectangle', 'png'), ('triangle', 'png'), ('rock', 'png')]

OUT_DIR: str = "output"
RECON_DIR: str = "reconstructions"
RES_DIR: str = "resources"


def _sum_matrix(matrix_in):
    m = ny.array(matrix_in)
    row_sums = ny.sum(m, axis=1)
    col_sums = ny.sum(m, axis=0)
    return row_sums, col_sums


def _sum_axis(matrix_in: arr, axis: int):
    return ny.sum(matrix_in, axis)


def _to_ndarray(matrix_in: list, dt) -> arr:
    return ny.array(matrix_in, dtype=dt)


def calloc(n) -> list:
    return [0 for _ in range(n)]


def analyze():
    for file, ext in IMG_SET:
        matrix = _read_file('resources', file, ext=ext)
        row_sums, col_sums = _sum_matrix(matrix)
        out = dict([("rowsum", row_sums), ("colsum", col_sums)])
        export_file(OUT_DIR + '/' + file, 'mat', out)


def reconstruct():
    for file, ext in IMG_SET:
        imp: dict = _import_file(OUT_DIR + '/' + file, 'mat')
        recon = _kaczmarz_binary(imp, MAX_ITER)
        matrix = _read_file(RES_DIR, file, ext=ext)
        print(relative_err(matrix, recon))
        _write_file(RECON_DIR, name='diff_' + file, ext=ext, file=recon-matrix)
        _write_file(RECON_DIR, name=file, ext=ext, file=recon)


def _init_matrix(init_val, row_len: int, col_len: int) -> list:
    matrix = [[init_val for _ in range(row_len)] for _ in range(col_len)]
    return matrix


def relative_err(m_1, m_2):
    return (ny.sum(ny.abs(ny.subtract(m_1, m_2)))) / (len(m_1) ^ 2)


def _stop_criteria(matrix, m_n, epsilon) -> bool:
    m_out = relative_err(matrix, m_n)
    # print(m_out)
    return m_out < epsilon


def _kaczmarz_binary(sums: dict, max_iter: int):
    rowsum: arr = sums['rowsum']
    colsum: arr = sums['colsum']
    matrix = _to_ndarray(_init_matrix(0, rowsum.size, colsum.size), float)
    i = 0
    while i < max_iter:
        m_n = matrix.copy()
        _axis_loop(m_n, colsum[0], 0)
        _axis_loop(m_n, rowsum[0], 1)
        b = _stop_criteria(matrix, m_n, EPSILON)
        if b:
            break
        matrix = m_n
        i += 1
    matrix = matrix.round()
    return matrix


def _axis_loop(matrix: arr, a: arr, axis: int):
    i, n = 0, a.size
    sums = _sum_axis(matrix, axis)
    m = ny.transpose(matrix) if axis == 0 else matrix
    while i < n:
        b_i = (sums[i] - a[i]) / n
        m[i] = mini(maxi(m[i] - b_i, 0), 1)
        i += 1
    return


def export_file(file, extension, out):
    mkdir_here(OUT_DIR)
    f = file + '.' + extension
    scipy.io.savemat(f, out)


def _import_file(file: str, extension: str) -> dict:
    return scipy.io.loadmat(file + '.' + extension)


def _read_file(*path, sep='/', ext='png'):
    file = cv2.imread(sep.join(path) + '.' + ext, 0)
    file = cv2.bitwise_not(file) / 255
    return file


def _write_file(*path, sep='/', ext='png', name='out', file=None):
    mkdir_here(*path)
    return cv2.imwrite(sep.join(path)
                       + sep + name + '.' + ext, ny.abs(file - 1) * 255)


def mkdir_here(*path: str, sep='/'):
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, sep.join(path))
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
