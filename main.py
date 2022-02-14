from practical_1.bin.tomography import analyze, reconstruct
import sys
import logging
import argparse
from logging import critical, error, info, warning, debug

logging.basicConfig(format='%(message)s', level=logging.DEBUG, stream=sys.stdout)


def parse_arguments():
    """Read arguments from a command line."""
    parser = argparse.ArgumentParser(description='Arguments get parsed via --commands')
    parser.add_argument("-i", metavar='input file', required=True,
                        help='an input dataset in .txt file')

    args = parser.parse_args()

    return args


def practical_exec():
    analyze()
    reconstruct()


if __name__ == '__main__':
    practical_exec()
