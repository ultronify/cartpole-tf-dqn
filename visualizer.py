"""
Training progress visualizer
"""
import time
from abc import ABC, abstractmethod


def get_training_visualizer(visualizer_type):
    """
    A factory wrapper to generate training progress
    visualizers.

    :param visualizer_type: (str) the type of the visualizer to create
    :return: TrainingVisualizer
    """
    if visualizer_type == 'none':
        return DummyTrainingVisualizer()
    if visualizer_type == 'streamlit':
        return StreamlitTrainingVisualizer()
    raise RuntimeError('Visualizer of type {0} not found'.format(type))


class TrainingVisualizer(ABC):
    """
    Base training visualizer
    """

    @abstractmethod
    def log_loss(self, loss):
        """
        Logs a loss history to the desired visualization

        :param loss: a list of loss history
        :return: None
        """
        return

    @abstractmethod
    def log_reward(self, reward):
        """
        Logs a reward history to the desired visualization

        :param reward: a list of reward history
        :return: None
        """
        return

    @abstractmethod
    def get_ui_feedback(self):
        """
        Gets the configuration from UI

        :return: None
        """
        return


class DummyTrainingVisualizer(TrainingVisualizer):
    """
    Used when no logging is required
    """

    def log_loss(self, loss):
        """
        A dummy logger that does nothing

        :param loss: a list of loss history
        :return: None
        """
        return None

    def log_reward(self, reward):
        """
        A dummy logger that does nothing

        :param reward: a list of reward history
        :return: None
        """
        return None

    def get_ui_feedback(self):
        """
        A dummy logger that does nothing

        :return: None
        """
        return None


class StreamlitTrainingVisualizer(TrainingVisualizer):
    """
    Used when runs with stream lit
    """

    def __init__(self):
        """
        Initializes the streamlit dashboard with elements
        """
        print('Initializing stream lit visualizer')
        # pylint: disable=import-outside-toplevel)
        import streamlit as st
        self.loss_history = []
        self.reward_history = []
        # pylint: disable=no-value-for-parameter
        st.sidebar.text('Cart Pole DQN Training (TensorFlow 2.0)')
        start_bar = st.sidebar.progress(0)
        for percent_complete in range(100):
            time.sleep(0.02)
            start_bar.progress(percent_complete + 1)
        self.update_freq = st.sidebar.slider('Update frequency', 0, 500, 120)
        self.epsilon = float(st.sidebar.slider('Epsilon', 0, 100, 10)) / 100.0
        self.eval_eps = st.sidebar.slider('Eval episodes', 0, 100, 10)
        st.text('Training loss history')
        self.loss_chart = st.line_chart(self.loss_history)
        st.text('Average reward history')
        self.reward_chart = st.line_chart(self.reward_history)

    def log_loss(self, loss):
        """
        Adds a loss history to the chart

        :param loss: a list of loss history
        :return: None
        """
        self.loss_chart.add_rows(loss)

    def log_reward(self, reward):
        """
        Adds a reward history to the chart

        :param reward: a list of reward history
        :return:
        """
        self.reward_chart.add_rows(reward)

    def get_ui_feedback(self):
        """
        Gets the user defined config from the UI

        :return: config
        """
        return {
            'update_freq': self.update_freq,
            'epsilon': self.epsilon,
            'eval_eps': self.eval_eps,
        }
