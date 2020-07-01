"""
Entrypoint (CLI)
"""

import argparse

import config
from train import train_model


def main():
    """
    The CLI entrypoint to the APIs

    :return: None
    """
    parser = argparse.ArgumentParser(
        description='CLI for Cart Pole DQN with TensorFlow 2.0.')
    parser.add_argument('--num_iterations', dest='num_iterations', type=int,
                        default=config.DEFAULT_NUM_ITERATIONS,
                        help='The number of episodes the '
                             'agent should train on.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=config.DEFAULT_BATCH_SIZE,
                        help='The batch size of training.')
    parser.add_argument('--max_replay_history', dest='max_replay_history', type=int,
                        default=config.DEFAULT_MAX_REPLAY_HISTORY,
                        help='The max length of the replay buffer.')
    parser.add_argument('--gamma', dest='gamma', type=int,
                        default=config.DEFAULT_GAMMA,
                        help='The discount rate.')
    parser.add_argument('--eval_eps', dest='eval_eps', type=int,
                        default=config.DEFAULT_EVAL_EPS,
                        help='Number of episode for evaluation.')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                        default=config.DEFAULT_LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--checkpoint_location', dest='checkpoint_location', type=str,
                        default=config.DEFAULT_CHECKPOINT_LOCATION,
                        help='The location for storing training checkpoints.')
    parser.add_argument('--model_location', dest='model_location', type=str,
                        default=config.DEFAULT_MODEL_LOCATION,
                        help='The location for storing pre-trained models.')
    parser.add_argument('--verbose', dest='verbose', type=str,
                        default=config.DEFAULT_VERBOSITY_OPTION,
                        choices=config.VERBOSITY_OPTIONS,
                        help='If logging information should show up in the '
                             'terminal.')
    parser.add_argument('--visualizer_type', dest='visualizer_type', type=str,
                        default=config.DEFAULT_VISUALIZER_TYPE,
                        choices=config.VISUALIZER_TYPES,
                        help='Which type of visualizer to use for training '
                             'progress.')
    parser.add_argument('--render_option', dest='render_option', type=str,
                        default=config.DEFAULT_RENDER_OPTION,
                        choices=config.RENDER_OPTIONS,
                        help='How the game plays should be rendered.')
    parser.add_argument('--persist_progress_option', dest='persist_progress_option',
                        type=str, default=config.DEFAULT_PERSIST_PROGRESS_OPTION,
                        choices=config.PERSIST_PROGRESS_OPTIONS,
                        help='How the training should be persisted.')
    args = parser.parse_args()
    max_avg_reward, benchmark_reward = train_model(
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        max_replay_history=args.max_replay_history,
        gamma=args.gamma,
        eval_eps=args.eval_eps,
        learning_rate=args.learning_rate,
        checkpoint_location=args.checkpoint_location,
        model_location=args.model_location,
        verbose=args.verbose,
        visualizer_type=args.visualizer_type,
        render_option=args.render_option,
        persist_progress_option=args.persist_progress_option
    )
    print(
        'Final best reward achieved is {0} against'
        'benchmark reward {1}'.format(max_avg_reward, benchmark_reward))


if __name__ == '__main__':
    main()
