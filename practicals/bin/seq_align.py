import argparse
import numpy as np
from typing import IO, Dict, Any

nw1: str = 'resources/nw_test1.txt'


def needleman(d_seq: dict, p: int, q: int, g: int):
    pass


def get_seq(path: str, n: int = 0) -> dict[Any, str]:
    dict_seq = {}
    try:
        with open(path) as file:
            for ix, line in range(0, n + 1):
                dict_seq[ix] = file.readline()
    except OSError:
        print('Error reading dict')
        exit(1)
    return dict_seq


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('a', metavar='N', type=int, nargs='+', help='revisions')
    return parser.parse_args()


def pre_process():
    dict_seq1 = get_seq(path=nw1, n=2)
    needleman(dict_seq1, args.a[0], args.a[1], args.a[2])


if __name__ == '__main__':
    args = parse_config()
    pre_process()
