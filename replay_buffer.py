import numpy as np
import random
from collections import deque


class DqnReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.experiences = deque(maxlen=max_size)

    def record(self, state, reward, next_state, action, done):
        self.experiences.append((state, next_state, action, reward, done))

    def get_volume(self):
        return len(self.experiences)

    def can_sample_batch(self, batch_size):
        return self.get_volume() > batch_size

    def sample_batch(self, batch_size):
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
