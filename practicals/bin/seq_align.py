import argparse
import copy

import numpy as np
from numpy import ndarray as mx

from utils import min_dict

nw1: str = 'resources/nw_test1.txt'


class Cf:
    """Cost functions wrapper class .
    Includes hardcoded strings for keys (params) for value access"""

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
        prms[Cf.match][Cf.xy] = (xy[0] - 1, xy[1] - 1)
        prms[Cf.match][Cf.out] = D[xy[0] - 1][xy[1] - 1] \
                                 + Cf.w(prms[Cf.cost], seq[0][xy[0] - 1], seq[1][xy[1] - 1])

    @staticmethod
    def f_delete(D: mx, prms: dict, xy: tuple):
        prms[Cf.delete][Cf.xy] = (xy[0] - 1, xy[1])
        prms[Cf.delete][Cf.out] = D[xy[0] - 1][xy[1]] + prms[Cf.cost][Cf.g]

    @staticmethod
    def f_insert(D: mx, prms: dict, xy: tuple):
        prms[Cf.insert][Cf.xy] = (xy[0], xy[1] - 1)
        prms[Cf.insert][Cf.out] = D[xy[0]][xy[1] - 1] + prms[Cf.cost][Cf.g]

    @staticmethod
    def w(prms: dict, a: str, b: str):
        return prms[Cf.p] if (a == b) else prms[Cf.q]

    f_dict: dict = {
        match: dict(f=f_match, sym=chr(92)),
        delete: dict(f=f_delete, sym='|'),
        insert: dict(f=f_insert, sym='-')
    }


def needleman(d_seq: list[str], p: int, q: int, g: int):
    if len(d_seq) < 2:
        raise IOError
    f_dict = Cf.f_dict
    f_dict[Cf.cost] = {Cf.p: p, Cf.q: q, Cf.g: g}
    D: mx = np.ones((len(d_seq[0]) + 1, len(d_seq[1]) + 1))
    P: mx = np.full((len(d_seq[0]) + 1, len(d_seq[1]) + 1), fill_value="-", dtype=str)
    populate(D, P, f_dict[Cf.cost], d_seq[0], d_seq[1])
    calc_align(D, P, f_dict, d_seq[0], d_seq[1])
    print(D)
    print(P)
    optimal_align(P, d_seq)


def calc_align(D: mx, P: mx, prms: dict, *seq):
    for ix in range(1, D.shape[0]):
        mini: dict
        for jx in range(1, D.shape[1]):
            f = Cf.f
            prms[Cf.match][f](D, prms, (ix, jx), seq[0], seq[1])
            prms[Cf.insert][f](D, prms, (ix, jx))
            prms[Cf.delete][f](D, prms, (ix, jx))
            res = [prms[Cf.match], prms[Cf.insert], prms[Cf.delete]]
            mini = min_dict(res, Cf.out)
            D[ix, jx] = mini[Cf.out]
            P[ix, jx] = mini[Cf.sym]


al_map = {
    chr(92): '|',
    '|': '-'
}

f_map = {
    '\\': (-1, -1),
    '|': (-1, 0),
    '-': (0, -1)
}


def optimal_align(P: mx, seq: list[str]):
    pred_map: list[tuple] = []
    first: str = seq[0]
    second: str = seq[1]
    if len(seq[0]) < len(seq[1]):
        o = first
        first = second
        second = o
    print(first)
    xy: tuple = np.subtract(P.shape, (1, 1))
    pred_map.append(xy)
    align: str = al_map[P[xy[0]][xy[1]]]
    for ix in range(len(first), 1, -1):
        xy = np.add(xy, f_map[P[xy[0]][xy[1]]])
        align = al_map[P[xy[0]][xy[1]]] + align
    print(align)
    print(second)


def populate(D: mx, P: mx, d_cost: dict, a: str, b: str):
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
    parser.add_argument('a', metavar='N', type=int, nargs='+', help='revisions')
    return parser.parse_args()


def pre_process():
    seq_stack = get_seq(path=nw1)
    needleman(seq_stack, args.a[0], args.a[1], args.a[2])


if __name__ == '__main__':
    args = parse_config()
    pre_process()
