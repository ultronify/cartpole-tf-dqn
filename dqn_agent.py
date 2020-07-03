"""
DQN agent
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


# pylint: disable=too-many-instance-attributes
class DqnAgent:
    """
    DQN agent with production policy and benchmark
    """

    # pylint: disable=too-many-arguments
    def __init__(self, state_space, action_space, gamma, lr, verbose,
                 checkpoint_location, model_location, persist_progress_option, mode, epsilon):
        self.action_space = action_space
        self.mode = mode
        self.state_space = state_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.persist_progress_option = persist_progress_option
        self.verbose = verbose
        self.model_location = model_location
        self.checkpoint_location = checkpoint_location
        if self.verbose == 'init':
            print('Construct DQN agent with: ')
            print('Action space: ')
            print(action_space)
            print('State space: ')
            print(state_space)
        self.q_net = self._build_dqn_model(state_space=state_space,
                                           action_space=action_space, learning_rate=lr)
        self.target_q_net = self._build_dqn_model(state_space=state_space,
                                                  action_space=action_space, learning_rate=lr)
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                              net=self.q_net)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_location, max_to_keep=10)
        if self.persist_progress_option == 'all':
            if self.mode == 'train':
                self.load_checkpoint()
                self.update_target_network()
            if self.mode == 'test':
                self.load_model()

    @staticmethod
    def _build_dqn_model(state_space, action_space, learning_rate):
        """
        Builds a neural network for the agent

        :param state_space: state specification
        :param action_space: action specification
        :param learning_rate: learning rate
        :return: model
        """
        q_net = Sequential()
        q_net.add(Dense(128, input_dim=state_space, activation='relu',
                        kernel_initializer='he_uniform'))
        q_net.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(action_space, activation='linear',
                        kernel_initializer='he_uniform'))
        q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                      loss='mse')
        q_net.summary()
        return q_net

    def save_model(self):
        """
        Saves model to file system

        :return: None
        """
        tf.saved_model.save(self.q_net, self.model_location)

    def load_model(self):
        """
        Loads previously saved model
        :return: None
        """
        self.q_net = tf.saved_model.load(self.model_location)

    def save_checkpoint(self):
        """
        Saves training checkpoint

        :return: None
        """
        self.checkpoint_manager.save()

    def load_checkpoint(self):
        """
        Loads training checkpoint into the underlying model

        :return: None
        """
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

    def update_target_network(self):
        """
        Updates the target Q network with the parameters
        from the currently trained Q network.

        :return: None
        """
        if self.verbose != 'none':
            print('Update target Q network')
        self.target_q_net.set_weights(self.q_net.get_weights())

    def train(self, state_batch, next_state_batch, action_batch, reward_batch,
              done_batch, batch_size):
        """
        Train the model on a batch

        :param state_batch: batch of states
        :param next_state_batch: batch of next states
        :param action_batch: batch of actions
        :param reward_batch: batch of rewards
        :param done_batch: batch of done status
        :param batch_size: the size of the batch
        :return: loss history
        """
        current_q = self.q_net(state_batch).numpy()
        target_q = np.copy(current_q)
        next_q = self.target_q_net(next_state_batch)
        max_next_q = np.amax(next_q, axis=1)
        for batch_idx in range(batch_size):
            if done_batch[batch_idx]:
                target_q[batch_idx][action_batch[batch_idx]] = \
                    reward_batch[batch_idx]
            else:
                target_q[batch_idx][action_batch[batch_idx]] = \
                    reward_batch[batch_idx] + self.gamma * max_next_q[batch_idx]
        if self.verbose == 'loss':
            print('reward batch shape: ', reward_batch.shape)
            print('next Q shape: ', next_q.shape)
            print('next state batch shape: ', next_state_batch.shape)
            print('max next Q shape: ', max_next_q.shape)
            print('target Q shape: ', target_q.shape)
            print('sample target Q: ', target_q[0])
            print('sample current Q: ', current_q[0])
        history = self.q_net.fit(x=state_batch, y=target_q, verbose=0)
        if self.persist_progress_option == 'all':
            self.save_checkpoint()
        loss = history.history['loss']
        return loss

    # pylint: disable=unused-argument
    def random_policy(self, state):
        """
        Outputs a random action

        :param state: current state
        :return: action
        """
        return np.random.randint(0, self.action_space)

    def collect_policy(self, state):
        """
        The policy for collecting data points which can contain some randomness to
        encourage exploration.

        :return: action
        """
        if np.random.random() < self.epsilon:
            return self.random_policy(state=state)
        else:
            return self.policy(state=state)

    def policy(self, state):
        """
        Outputs a action based on model

        :param state: current state
        :return: action
        """
        state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_q = self.q_net(state_input)
        optimal_action = np.argmax(action_q.numpy()[0], axis=0)
        if self.verbose == 'policy':
            print('state: ', state)
            print('state_input: ', state_input)
            print('action Q: ', action_q)
            print('optimal action: ', optimal_action)
        return optimal_action
