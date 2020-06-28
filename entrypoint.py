import argparse

from train import train_model


def main():
    parser = argparse.ArgumentParser(description='CLI for Cart Pole RL Demonstration.')
    parser.add_argument('--verbose', dest='verbose', type=str, default='progress',
                        choices=['progress', 'loss', 'policy', 'init'],
                        help='If logging information should show up in the terminal.')
    args = parser.parse_args()
    train_model(
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
