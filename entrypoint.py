"""
Entrypoint (CLI)
"""

import argparse

from train import train_model


def main():
    """
    The CLI entrypoint to the APIs

    :return: None
    """
    parser = argparse.ArgumentParser(
        description='CLI for Cart Pole RL Demonstration.')
    parser.add_argument('--verbose', dest='verbose', type=str,
                        default='progress',
                        choices=['progress', 'loss', 'policy', 'init'],
                        help='If logging information should show up in the '
                             'terminal.')
    parser.add_argument('--visualizer_type', dest='visualizer_type', type=str,
                        default='none',
                        choices=['none', 'streamlit'],
                        help='Which type of visualizer to use for training '
                             'progress.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32,
                        help='The batch size of training.')
    parser.add_argument('--num_iterations', dest='num_iterations', type=int,
                        default=2000,
                        help='The number of episodes the '
                             'agent should train on.')
    args = parser.parse_args()
    train_model(
        verbose=args.verbose,
        visualizer_type=args.visualizer_type,
    )


if __name__ == '__main__':
    main()
