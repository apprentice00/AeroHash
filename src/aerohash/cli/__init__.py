"""Lazy command dispatch keeps CLI help independent of data and GPU initialization."""
import argparse
import importlib
import sys

COMMANDS = {'train': 'train', 'evaluate': 'evaluate', 'preprocess': 'preprocess',
            'clean': 'clean', 'build-val': 'build_val', 'collect': 'collect'}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog='aerohash', description='Weather time-series hashing and retrieval')
    parser.add_argument('command', choices=COMMANDS)
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    command = parser.parse_args(argv[:1]).command
    return importlib.import_module(f'aerohash.cli.{COMMANDS[command]}').main(argv[1:])
