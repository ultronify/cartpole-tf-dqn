from abc import ABC, abstractmethod


def get_training_visualizer(visualizer_type):
    if visualizer_type == 'none':
        return DummyTrainingVisualizer()
    if visualizer_type == 'stream_lit':
        return StreamLitTrainingVisualizer()
    raise RuntimeError('Training visualizer of type {0} not found'.format(type))


class TrainingVisualizer(ABC):
    @abstractmethod
    def log_loss(self, loss):
        pass

    @abstractmethod
    def log_reward(self, reward):
        pass


class DummyTrainingVisualizer(TrainingVisualizer):
    def log_loss(self, loss):
        pass

    def log_reward(self, reward):
        pass


class StreamLitTrainingVisualizer(TrainingVisualizer):
    def __init__(self):
        print('Initializing stream lit visualizer')
        import streamlit as st
        self.loss_history = []
        self.reward_history = []
        st.title('Training progress for tf2 cart pole experiment')
        self.loss_chart = st.line_chart(self.loss_history)
        self.reward_chart = st.line_chart(self.reward_history)

    def log_loss(self, loss):
        self.loss_chart.add_rows(loss)

    def log_reward(self, reward):
        self.reward_chart.add_rows(reward)
