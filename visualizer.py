"""
Training progress visualizer
"""

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
        return

    def log_reward(self, reward):
        """
        A dummy logger that does nothing

        :param reward: a list of reward history
        :return: None
        """
        return


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
        st.title('Training progress for tf2 cart pole experiment')
        self.loss_chart = st.line_chart(self.loss_history)
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
