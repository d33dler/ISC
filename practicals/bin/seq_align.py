import argparse
import copy

import numpy as np
from numpy import ndarray as mx

from utils import min_dict

nw1: str = 'resources/nw_test1.txt'


def _output_file_init():
    output = open('nw3-output.txt', 'w')
    output.write("Name: Radu Rebeja + Laurens van Heerde\n")
    output.write("ISC, Practical 3 \n")
    output.close()


def _output_file_write(d: str):
    output = open('nw3-output.txt', 'a')
    output.write(d)
    output.write("\n")
    output.close()


class Cf:
    """Cost functions wrapper class .
    Includes hardcoded strings for keys (params) used for prudent value accessing"""

    match: str = 'match'
    delete: str = 'delete'
    insert: str = 'insert'
    out: str = 'out'
    sym: str = 'sym'
    cost: str = 'cost'
    p: str = 'p'
    q: str = 'q'
    g: str = 'g'
    f: str = 'f'
    xy: str = 'xy'

    @staticmethod
    def f_match(D: mx, prms: dict, xy: tuple, *seq):
        """
                Matching function:
        Assigns cost based on values compared
        Parameters
        ----------
        D :  Alignment matrix
        prms : parameters dictionary
        xy : (x,y) coordinates
        seq : Iterable of strings

        Returns
        -------
        """
        prms[Cf.match][Cf.xy] = (xy[0] - 1, xy[1] - 1)
        prms[Cf.match][Cf.out] = D[xy[0] - 1][xy[1] - 1] \
                                 + Cf.w(prms[Cf.cost], seq[0][xy[0] - 1], seq[1][xy[1] - 1])

    @staticmethod
    def f_delete(D: mx, prms: dict, xy: tuple):
        """
        Delete cost function
        Parameters
        ----------
        D :  Alignment matrix
        prms : parameters dictionary
        xy : (x,y) coordinates

        Returns
        -------

        """
        prms[Cf.delete][Cf.xy] = (xy[0] - 1, xy[1])
        prms[Cf.delete][Cf.out] = D[xy[0] - 1][xy[1]] + prms[Cf.cost][Cf.g]

    @staticmethod
    def f_insert(D: mx, prms: dict, xy: tuple):
        """
        Insertion cost function
        Parameters
        ----------
        D :  Alignment matrix
        prms : parameters dictionary
        xy : (x,y) coordinates

        Returns
        -------

        """
        prms[Cf.insert][Cf.xy] = (xy[0], xy[1] - 1)
        prms[Cf.insert][Cf.out] = D[xy[0]][xy[1] - 1] + prms[Cf.cost][Cf.g]

    @staticmethod
    def w(prms: dict, a: str, b: str):
        """
        Helper function for Matching
        """
        return prms[Cf.p] if (a == b) else prms[Cf.q]

    """
    Function dictionary
    """
    f_dict: dict = {
        match: dict(f=f_match, sym=chr(92)),
        delete: dict(f=f_delete, sym='|'),
        insert: dict(f=f_insert, sym='-')
    }


def needleman(d_seq: list[str], p: int, q: int, g: int):
    """
    Main function:
    Creates dictionaries for fast function-symbol access and
    cost variables usage.
    Creates alignment (D) and predecessor (P) matrices
    Populates the matrices before calculations
    Calls align_calc to calculate optimal alignment
    Parameters
    ----------
    d_seq : Strings to compare
    p : match cost value
    q : mismatch cost value
    g : gap cost value

    Returns
    -------

    """
    _output_file_init()
    _output_file_write("\n\nString s: \n")
    _output_file_write(d_seq[0])
    _output_file_write("\n\nString t: \n")
    _output_file_write(d_seq[1])

    if len(d_seq) < 2:
        raise IOError
    f_dict = Cf.f_dict
    f_dict[Cf.cost] = {Cf.p: p, Cf.q: q, Cf.g: g}
    D: mx = np.ones((len(d_seq[0]) + 1, len(d_seq[1]) + 1))
    P: mx = np.full(
        (len(d_seq[0]) + 1, len(d_seq[1]) + 1), fill_value="-", dtype=str)
    populate(D, P, f_dict[Cf.cost], d_seq[0], d_seq[1])
    calc_align(D, P, f_dict, d_seq[0], d_seq[1])

    _output_file_write("\n\nMatrix D: \n")
    _output_file_write(str(D))

    _output_file_write("\n\nMatrix P: \n")
    _output_matrix(P)

    optimal_align(P, d_seq)


def _output_matrix(M: mx):
    for row in M:
        _output_file_write(''.join(row))


def print_matrix(M: mx):
    for row in M:
        print(''.join(row))


def calc_align(D: mx, P: mx, prms: dict, *seq):
    """
    Applies the Needleman-Wunsch algorithm
    For optimization we run it once to populate both
    alignment (D) and predecessor (P) matrices

    Parameters
    ----------
    D : alignment matrix
    P : predecessor matrix
    prms : parameter dictionary
    seq : iterable of compared strings [string_1, string_2]

    Returns
    -------

    """
    f = Cf.f
    for ix in range(1, D.shape[0]):
        mini: dict
        for jx in range(1, D.shape[1]):
            prms[Cf.match][f](D, prms, (ix, jx), seq[0], seq[1])
            prms[Cf.insert][f](D, prms, (ix, jx))
            prms[Cf.delete][f](D, prms, (ix, jx))
            res = [prms[Cf.match], prms[Cf.insert], prms[Cf.delete]]
            mini = min_dict(res, Cf.out)
            D[ix, jx] = mini[Cf.out]
            P[ix, jx] = mini[Cf.sym]


"""
    Dictionary mapping symbols for writing out optimal alignment
"""
al_map = {
    chr(92): '|',
    '|': '-'
}
"""
    Dictionary mapping symbols to matrix step progressions
    in predecessor matrix
"""
f_map = {
    '\\': (-1, -1),
    '|': (-1, 0),
    '-': (0, -1)
}


def optimal_align(P: mx, seq: list[str]):
    """
    Calculates optimal alignment by reverse walking the
    predecessor matrix

    Parameters
    ----------
    P : predecessor matrix
    seq : list of compared strings [string_1, string_2]

    Returns
    -------

    """
    pred_map: list[tuple] = []
    first: str = seq[0]
    second: str = seq[1]
    if len(seq[0]) < len(seq[1]):
        o = first
        first = second
        second = o
    xy: tuple = np.subtract(P.shape, (1, 1))
    pred_map.append(xy)
    align: str = al_map[P[xy[0]][xy[1]]]
    for ix in range(len(first), 1, -1):
        xy = np.add(xy, f_map[P[xy[0]][xy[1]]])
        align = al_map[P[xy[0]][xy[1]]] + align

    _output_file_write("\n\nAlignment P: \n")
    _output_file_write(first)
    _output_file_write(align)
    s = 0
    align = list(align)
    for i in range(len(align)):
        if align[i] == "|":
            align[i] = second[s]
            s += 1
    _output_file_write(''.join(align))


def populate(D: mx, P: mx, d_cost: dict, a: str, b: str):
    """
    Function to populate and initialize the matrices with
    default values :
    -gap penalties on 1st column&row in D
    -marking gap penalties with symbol in P
    Parameters
    ----------
    D : alignment (cost) matrix
    P : predecessor matrix
    d_cost : dictionary of cost variables (p,q,g)
    a : 1st string sequence
    b : 2nd string sequence

    Returns
    -------

    """
    g = d_cost[Cf.g]
    for ix, l1 in enumerate(a):
        for jx, l2 in enumerate(b):
            D[ix + 1][jx + 1] = 0 if l1 == l2 else g
    D[0][:] = gen_arr(len(D[0]) * g, step=g)
    np.transpose(D)[0][:] = gen_arr(len(D) * g, step=g)
    np.transpose(P)[0][:] = '|'
    P[0][0] = '*'


def gen_arr(length: int, step: int = 1):
    return [_ for _ in range(0, length, step)]


def get_seq(path: str) -> list[str]:
    """
    Function to read the string from file
    Parameters
    ----------
    path : txt file to read strings from

    Returns
    -------

    """
    seq_stack = []
    try:
        with open(path) as file:
            seq_stack = [line.rstrip() for line in file]
    except OSError:
        print('Error reading dict')
        exit(1)
    return seq_stack


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('a', metavar='N', type=int,
                        nargs='+', help='revisions')
    return parser.parse_args()


def pre_process():
    seq_stack = get_seq(path=nw1)
    needleman(seq_stack, args.a[0], args.a[1], args.a[2])


if __name__ == '__main__':
    args = parse_config()
    pre_process()
