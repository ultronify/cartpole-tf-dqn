import argparse

from train import train_model


def main():
    parser = argparse.ArgumentParser(description='CLI for Cart Pole RL Demonstration.')
    parser.add_argument('--verbose', dest='verbose', type=str, default='progress',
                        choices=['progress', 'loss', 'policy', 'init'],
                        help='If logging information should show up in the terminal.')
    parser.add_argument('--visualizer_type', dest='visualizer_type', type=str, default='none',
                        choices=['none', 'stream_lit'],
                        help='Which type of visualizer to use for training progress.')
    args = parser.parse_args()
    train_model(
        verbose=args.verbose,
        visualizer_type=args.visualizer_type,
    )


if __name__ == '__main__':
    main()
