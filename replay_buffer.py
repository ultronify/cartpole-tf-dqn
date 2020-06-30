"""
DQN replay buffer
"""

import random
from collections import deque
import numpy as np


class DqnReplayBuffer:
    """
    DQN replay buffer to keep track of game play records
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self.experiences = deque(maxlen=max_size)

    # pylint: disable=too-many-arguments
    def record(self, state, reward, next_state, action, done):
        """
        Puts a game play state into records

        :param state: current game state
        :param reward: reward after taking action
        :param next_state: state after taking action
        :param action: action taken
        :param done: if the episode is finished
        :return: None
        """
        self.experiences.append((state, next_state, action, reward, done))

    def get_volume(self):
        """
        Gets the current length of the records

        :return: (int) the length of the records
        """
        return len(self.experiences)

    def can_sample_batch(self, batch_size):
        """
        Returns if a batch can be sampled

        :param batch_size: the size of the batch to be sampled
        :return: (bool) if can sample
        """
        return self.get_volume() > batch_size

    def sample_batch(self, batch_size):
        """
        Samples a batch from the records

        :param batch_size: the size of the batch to be sampled
        :return: sample batch
        """
        sampled_batch = random.sample(self.experiences, batch_size)
        state_batch = []
        next_state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        for record in sampled_batch:
            state_batch.append(record[0])
            next_state_batch.append(record[1])
            action_batch.append(record[2])
            reward_batch.append(record[3])
            done_batch.append(record[4])
        return np.array(state_batch), np.array(next_state_batch), np.array(
            action_batch), np.array(reward_batch), np.array(done_batch)
