import os
import sys

import numpy as ny
from numpy import ndarray as arr, minimum as mini, maximum as maxi
import cv2
import scipy.io

ny.set_printoptions(threshold=sys.maxsize)
# how many times the kaczmarz algorithm should process the rays
MAX_ITER: int = 9
# error threshold
EPSILON: float = 0.01

# images to be processed
IMG_SET: list = [('oval', 'png'), ('rectangle', 'png'),
                 ('triangle', 'png'), ('rock', 'png')]

OUT_DIR: str = "output"
RECON_DIR: str = "reconstructions"
RES_DIR: str = "resources"


def _sum_matrix(matrix_in):
    # computes the sums of horizontal, vertical and diagonal rays through the matrix representation of an image
    # for the horizontal and vertical rays the dependency `ny` is used
    # for the diagonal rays additional implementation was needed, which is defined in the `_sum_diagonals` function
    m = ny.array(matrix_in)
    row_sums = ny.sum(m, axis=1)
    col_sums = ny.sum(m, axis=0)
    sn_diag_sums = _sum_diagonals(m)
    return row_sums, col_sums, sn_diag_sums


def _sum_diagonals(matrix: arr):
    # sums diagonal rays
    diag_sums = []
    row_len = len(matrix[0])
    col_len = len(matrix)
    for span in range(1, row_len + col_len):
        _sum = 0
        start_col = max(0, span - row_len)
        count = min(span, (col_len - start_col), row_len)
        for j in range(0, count):
            _sum += matrix[min(row_len, span) - j - 1][start_col + j]
        diag_sums.append(_sum)
    return diag_sums


def _sum_axis(matrix_in: arr, axis: int):
    return ny.sum(matrix_in, axis)


def _to_ndarray(matrix_in: list, dt) -> arr:
    return ny.array(matrix_in, dtype=dt)

# def calloc(n) -> list:
#     return [0 for _ in range(n)]


def analyze():
    for file, ext in IMG_SET:
        matrix = _read_file(RES_DIR, file, ext=ext)
        row_sums, col_sums, sn_diag_sums = _sum_matrix(matrix)
        out = dict([("rowsum", row_sums), ("colsum", col_sums),
                   ("diagsum", sn_diag_sums)])
        export_file(OUT_DIR + '/' + file, 'mat', out)


def reconstruct():
    for file, ext in IMG_SET:
        imp: dict = _import_file(OUT_DIR + '/' + file, 'mat')
        recon = _kaczmarz_binary(imp, MAX_ITER)
        matrix = _read_file(RES_DIR, file, ext=ext)
        print(relative_err(matrix, recon))
        _write_file(RECON_DIR, name='diff_' + file,
                    ext=ext, file=recon - matrix)
        _write_file(RECON_DIR, name=file, ext=ext, file=recon)


def _init_matrix(init_val, row_len: int, col_len: int) -> list:
    # creates a matrix with size `row_len`*`col_len` with initial values `init_val`
    matrix = [[init_val for _ in range(row_len)] for _ in range(col_len)]
    return matrix


def relative_err(m_1, m_2):
    # calculates relative error between the input array m_1 and the reconstruction m_2
    return (ny.sum(ny.abs(ny.subtract(m_1, m_2)))) / (len(m_1) ^ 2)


def _stop_criteria(matrix, m_n, epsilon) -> bool:
    # tests if the relative error between `matrix` and `m_n` is less than the a threshold `epsilon`
    m_out = relative_err(matrix, m_n)
    return m_out < epsilon


def _diag_loop(matrix: arr, a):
    # implementation for update step for diagonal lines in Kaczmarz method
    row_len = len(matrix[0])
    col_len = len(matrix)
    rc = row_len + col_len
    sums: list = _sum_diagonals(matrix)
    for span in range(1, row_len + col_len - 1):
        _sum = 0
        start_col = max(0, span - row_len)
        count = min(span, (col_len - start_col), row_len)
        for j in range(0, count):
            b_i = (sums[span] - a[span]) / rc
            x = min(row_len, span) - j - 1
            y = start_col + j
            matrix[x][y] = mini(maxi(matrix[x][y] - b_i, 0), 1)


def _kaczmarz_binary(sums: dict, max_iter: int):
    # implementation of the Kaczmarz algorithm
    # this implementation is extended to also process diagonal rays
    rowsum: arr = sums['rowsum']
    colsum: arr = sums['colsum']
    diagsum = sums['diagsum']
    matrix = _to_ndarray(_init_matrix(0, rowsum.size, colsum.size), float)
    i = 0
    while i < max_iter:
        m_n = matrix.copy()
        _axis_loop(m_n, colsum[0], 0)
        _axis_loop(m_n, rowsum[0], 1)
        _diag_loop(m_n, diagsum[0])
        b = _stop_criteria(matrix, m_n, EPSILON)
        if b:
            break
        matrix = m_n
        i += 1
    matrix = matrix.round()
    return matrix


def _axis_loop(matrix: arr, a: arr, axis: int):
    # implementation for update step for horizontal and vertical lines in Kaczmarz method
    i, n = 0, a.size
    sums = _sum_axis(matrix, axis)
    m = ny.transpose(matrix) if axis == 0 else matrix
    while i < n:
        b_i = (sums[i] - a[i]) / n
        m[i] = mini(maxi(m[i] - b_i, 0), 1)
        i += 1
    return


def export_file(file: str, extension: str, out):
    # exports a matlab file
    mkdir_here(OUT_DIR)
    f = file + '.' + extension
    scipy.io.savemat(f, out)


def _import_file(file: str, extension: str) -> dict:
    # read a matlab file
    return scipy.io.loadmat(file + '.' + extension)


def _read_file(*path, sep='/', ext='png'):
    # used for loading pictures
    file = cv2.imread(sep.join(path) + '.' + ext, 0)
    file = cv2.bitwise_not(file) / 255
    return file


def _write_file(*path, sep='/', ext='png', name='out', file=None):
    # used for saving an image
    mkdir_here(*path)
    return cv2.imwrite(sep.join(path)
                       + sep + name + '.' + ext, ny.abs(file - 1) * 255)


def mkdir_here(*path: str, sep='/'):
    # creates a directory
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, sep.join(path))
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
