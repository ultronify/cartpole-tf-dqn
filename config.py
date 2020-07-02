"""
Config
"""

DEFAULT_MODE = 'train'
"""
The default mode the program should run in
"""

MODE_OPTIONS = ['train', 'test']
"""
The supported modes
"""

DEFAULT_ENV_NAME = 'CartPole-v0'
"""
The OpenAI environment name to be used
"""

DEFAULT_NUM_ITERATIONS = 50000
"""
The default number of iteration to train the model
"""

DEFAULT_BATCH_SIZE = 128
"""
The default batch size the model should be trained on
"""

DEFAULT_MAX_REPLAY_HISTORY = 1000000
"""
The default max length of the replay buffer
"""

DEFAULT_GAMMA = 0.95
"""
The default discount rate for the Q learning
"""

DEFAULT_EVAL_EPS = 10
"""
The default number of episode the model should be evaluated with
"""

DEFAULT_LEARNING_RATE = 0.001
"""
The default learning rate
"""

DEFAULT_CHECKPOINT_LOCATION = './checkpoints'
"""
The default location to store the training checkpoints
"""

DEFAULT_MODEL_LOCATION = './model'
"""
The default location to store the best performing models
"""

DEFAULT_TARGET_NETWORK_UPDATE_FREQUENCY = 200
"""
How often the target Q network should get parameter update
from the training Q network.
"""

DEFAULT_RENDER_OPTION = 'none'
RENDER_OPTIONS = ['none', 'collect']

DEFAULT_VERBOSITY_OPTION = 'progress'
VERBOSITY_OPTIONS = ['progress', 'loss', 'policy', 'init']

DEFAULT_VISUALIZER_TYPE = 'none'
VISUALIZER_TYPES = ['none', 'streamlit']

DEFAULT_PERSIST_PROGRESS_OPTION = 'all'
PERSIST_PROGRESS_OPTIONS = ['none', 'all']
