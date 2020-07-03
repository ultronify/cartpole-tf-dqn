"""
Test

This module is for testing pre-trained agents
"""
import gym

import config
import utils
from dqn_agent import DqnAgent


# pylint: disable=too-many-arguments
def test_model(
        model_location=config.DEFAULT_MODEL_LOCATION,
        gamma=config.DEFAULT_GAMMA,
        verbose=config.DEFAULT_VERBOSITY_OPTION,
        learning_rate=config.DEFAULT_LEARNING_RATE,
        checkpoint_location=config.DEFAULT_CHECKPOINT_LOCATION,
        persist_progress_option=config.DEFAULT_PERSIST_PROGRESS_OPTION,
        render_option=config.DEFAULT_RENDER_OPTION,
        eval_eps=config.DEFAULT_EVAL_EPS,
        pause_time=config.DEFAULT_PAUSE_TIME,
        min_steps=config.DEFAULT_MIN_STEPS,
):
    """
    Test model tests agents

    :param min_steps: the minimum steps per episode for evaluation
    :param pause_time: the time paused for preparing screen recording
    :param eval_eps: the number of episode per evaluation
    :param render_option: how the game play should be rendered
    :param persist_progress_option:
    :param checkpoint_location: (not used in testing)
    :param learning_rate: (not used in testing)
    :param verbose: the verbosity level
    :param gamma: (not used in testing)
    :param model_location: used to load the pre-trained model
    :return: None
    """
    env_name = config.DEFAULT_ENV_NAME
    test_env = gym.make(env_name)
    agent = DqnAgent(state_space=test_env.observation_space.shape[0],
                     action_space=test_env.action_space.n,
                     gamma=gamma, verbose=verbose, lr=learning_rate,
                     checkpoint_location=checkpoint_location,
                     model_location=model_location,
                     persist_progress_option=persist_progress_option,
                     mode='test')
    avg_reward = utils.play_episodes(env=test_env, policy=agent.random_policy,
                                     render_option=render_option, num_eps=eval_eps,
                                     pause_time=pause_time, min_steps=min_steps)
    test_env.close()
    return avg_reward
