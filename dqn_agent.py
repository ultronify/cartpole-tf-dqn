import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


class DqnAgent(object):
    def __init__(self, state_space, action_space, gamma, lr, verbose):
        self.action_space = action_space
        self.state_space = state_space
        self.gamma = gamma
        self.verbose = verbose
        if self.verbose == 'init':
            print('Construct DQN agent with: ')
            print('Action space: ')
            print(action_space)
            print('State space: ')
            print(state_space)
        self.q_net = self._build_dqn_model(state_space=state_space, action_space=action_space, lr=lr)

    @staticmethod
    def _build_dqn_model(state_space, action_space, lr):
        q_net = Sequential()
        q_net.add(Dense(128, input_dim=state_space, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(action_space, activation='linear', kernel_initializer='he_uniform'))
        q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), loss='mse')
        q_net.summary()
        return q_net

    def train(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch, batch_size):
        current_q = self.q_net(state_batch).numpy()
        target_q = np.copy(current_q)
        next_q = self.q_net(next_state_batch)
        max_next_q = np.amax(next_q, axis=1)
        for batch_idx in range(batch_size):
            if done_batch[batch_idx]:
                target_q[batch_idx][action_batch[batch_idx]] = reward_batch[batch_idx]
            else:
                target_q[batch_idx][action_batch[batch_idx]] = reward_batch[batch_idx] + self.gamma * max_next_q[
                    batch_idx]
        if self.verbose == 'loss':
            print('reward batch shape: ', reward_batch.shape)
            print('next Q shape: ', next_q.shape)
            print('next state batch shape: ', next_state_batch.shape)
            print('max next Q shape: ', max_next_q.shape)
            print('target Q shape: ', target_q.shape)
            print('sample target Q: ', target_q[0])
            print('sample current Q: ', current_q[0])
        # loss = self.q_net.train_on_batch(x=state_batch, y=target_q)
        # return loss
        self.q_net.fit(x=state_batch, y=target_q)
        return 0

    def random_policy(self, state):
        return np.random.randint(0, self.action_space)

    def policy(self, state):
        state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_q = self.q_net(state_input)
        optimal_action = np.argmax(action_q.numpy()[0], axis=0)
        if self.verbose == 'policy':
            print('state: ', state)
            print('state_input: ', state_input)
            print('action Q: ', action_q)
            print('optimal action: ', optimal_action)
        return optimal_action
