import os

from practicals.bin.tomography import analyze, reconstruct
import sys
import logging
import argparse
from logging import critical, error, info, warning, debug

logging.basicConfig(format='%(message)s',
                    level=logging.DEBUG, stream=sys.stdout)

modules = {
    'tomography': 'practicals/bin/tomography.py',
    'sequence': 'practicals/bin/seq_align.py'
}


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('-ass', type=str, default='tomography',
                        help='specify the config for demo')
    parser.add_argument('a', metavar='N', type=str,
                        nargs='+', help='revisions')
    return parser.parse_args()


def practical_exec():
    analyze()
    reconstruct()


if __name__ == '__main__':
    args = parse_config()
    s = " "
    # print(args.a)
    try:
        os.system(f"python {modules[args.ass]} {s.join(args.a)}")
    except KeyError:
        print(f"Module with name {args.ass} not found")
        exit(1)
