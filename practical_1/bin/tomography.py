import math
import os
import sys
from typing import Any

import cv2
import numpy as ny
import scipy.io
from numpy import ndarray as arr, minimum as mini, maximum as maxi

ny.set_printoptions(threshold=sys.maxsize)
MAX_ITER: int = 100
EPSILON: float = 0.01

IMG_SET: list = [('oval', 'png'), ('rectangle', 'png'), ('triangle', 'png'), ('rock', 'png')]

OUT_DIR: str = "output"
RECON_DIR: str = "reconstructions"
RES_DIR: str = "resources"


def _sum_matrix(matrix_in):
    m = ny.array(matrix_in)
    row_sums = ny.sum(m, axis=1)
    col_sums = ny.sum(m, axis=0)
    sn_diag_sums = _sum_diagonals(m, axis=0)
    ns_diag_sums = _sum_diagonals(m, axis=1)
    return row_sums, col_sums, sn_diag_sums, ns_diag_sums


def _sum_diagonals(matrix: arr, axis=0):
    diag_sums = []
    row_len = len(matrix[0])
    col_len = len(matrix)
    m = ny.rot90(matrix, -1) if axis == 0 else matrix
    for span in range(1, row_len + col_len):
        _sum = 0
        start_col = max(0, span - row_len)
        count = min(span, (col_len - start_col), row_len)
        for j in range(0, count):
            _sum += m[min(row_len, span) - j - 1][start_col + j]
        diag_sums.append(_sum)
    return diag_sums


def _sum_axis(matrix_in: arr, axis: int):
    return ny.sum(matrix_in, axis)


def _to_ndarray(matrix_in: list, dt) -> arr:
    return ny.array(matrix_in, dtype=dt)


def calloc(n) -> list:
    return [0 for _ in range(n)]


def analyze():
    for file, ext in IMG_SET:
        matrix = _read_file(RES_DIR, file, ext=ext)
        row_sums, col_sums, sn_diag_sums, ns_diag_sums = _sum_matrix(matrix)
        out = dict([("rowsum", row_sums), ("colsum", col_sums),
                    ("sn_diagsum", sn_diag_sums), ("ns_diagsum", ns_diag_sums)])
        export_file(OUT_DIR + '/' + file, 'mat', out)


def reconstruct():
    for file, ext in IMG_SET:
        imp: dict = _import_file(OUT_DIR + '/' + file, 'mat')
        matrix = _read_file(RES_DIR, file, ext=ext)
        recon = _kaczmarz_binary(matrix, imp, MAX_ITER)
        print(relative_err(matrix, recon))
        _write_file(RECON_DIR, name='diff_' + file, ext=ext, file=recon - matrix)
        _write_file(RECON_DIR, name=file, ext=ext, file=recon)


def _init_matrix(init_val, row_len: int, col_len: int) -> list:
    matrix = [[init_val for _ in range(row_len)] for _ in range(col_len)]
    return matrix


def relative_err(m_1, m_2) -> float:
    return (ny.sum(ny.abs(ny.subtract(m_1, m_2)))) / (len(m_1) ^ 2)


def _stop_criteria(original, matrix, m_n, prev_err: float, epsilon) -> tuple[float, bool | Any]:
    m_out: float = relative_err(matrix, m_n)
    new_err: float = relative_err(original, ny.rint(m_n))
    print("err:", m_out)
    noise = (new_err > prev_err)
    return new_err, (m_out < epsilon) or noise


def _diag_loop(matrix: arr, a, axis: int):
    row_len = len(matrix[0])
    col_len = len(matrix)
    rc = row_len + col_len
    sums: list = _sum_diagonals(matrix)
    m = ny.rot90(matrix, -1) if axis == 0 else matrix
    for span in range(1, row_len + col_len - 1):
        _sum = 0
        start_col = max(0, span - row_len)
        count = min(span, (col_len - start_col), row_len)
        for j in range(0, count):
            b_i = (sums[span] - a[span]) / rc
            x = min(row_len, span) - j - 1
            y = start_col + j
            m[x][y] = mini(maxi(m[x][y] - b_i, 0), 1)


def _kaczmarz_binary(original, sums: dict, max_iter: int):
    rowsum: arr = sums['rowsum']
    colsum: arr = sums['colsum']
    sn_diagsum = sums['sn_diagsum']
    ns_diagsum = sums['ns_diagsum']
    matrix = _to_ndarray(_init_matrix(0, rowsum.size, colsum.size), float)
    prev_err: float = math.inf
    i = 0
    while i < max_iter:
        m_n = matrix.copy()
        _axis_loop(m_n, colsum[0], 0)
        _diag_loop(m_n, ns_diagsum[0], 1)
        _diag_loop(m_n, sn_diagsum[0], 0)
        _axis_loop(m_n, rowsum[0], 1)
        prev_err, b = _stop_criteria(original, matrix, m_n, prev_err, EPSILON)
        if b:
            break
        matrix = m_n
        i += 1
    return ny.rint(matrix)


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
