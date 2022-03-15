import argparse
import numpy as np
from numpy import ndarray as mx
from typing import IO, Dict, Any

nw1: str = 'resources/nw_test1.txt'


def needleman(d_seq: list[str], p: int, q: int, g: int):
    if len(d_seq) < 2:
        raise IOError
    d_costs = {'p': p, 'q': q, 'g': g}
    D: mx = np.ones((len(d_seq[0]) + 1, len(d_seq[1]) + 1))
    populate(D, d_costs, d_seq[0], d_seq[1])
    calc_align(D, d_costs, d_seq[0], d_seq[1])
    print(D)


def calc_align(D: mx, d_cost, a: str, b: str):
    for ix in range(1, D.shape[0]):
        for jx in range(1, D.shape[1]):
            _match = D[ix - 1][jx - 1] + w(d_cost, a[ix - 1], b[jx - 1])
            _del = D[ix-1][jx] + d_cost['g']
            _ins = D[ix][jx - 1] + d_cost['g']
            D[ix, jx] = min(_match, _ins, _del)


def w(d_cost: dict, a: str, b: str):
    return d_cost['p'] if (a == b) else d_cost['q']


def populate(D: mx, d_cost: dict, a: str, b: str):
    for ix, l1 in enumerate(a):
        for jx, l2 in enumerate(b):
            D[ix + 1][jx + 1] = 0 if l1 == l2 else 5  # ???
    D[0][:] = [_ for _ in range(0, (len(b)+1) * d_cost['g'], d_cost['g'])]
    np.transpose(D)[0][:] = [_ for _ in range(0, (len(a)+1) * d_cost['g'], d_cost['g'])]
    print(D)


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
