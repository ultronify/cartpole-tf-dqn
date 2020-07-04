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

DEFAULT_EPSILON = 0.05
"""
The default value for epsilon
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

DEFAULT_TARGET_NETWORK_UPDATE_FREQUENCY = 120
"""
How often the target Q network should get parameter update
from the training Q network.
"""

DEFAULT_RENDER_OPTION = 'none'
"""
The default value for rendering option
"""

RENDER_OPTIONS = ['none', 'collect']
"""
The available render options:

* none: don't render anything
* collect: render the game play while collecting data
"""

DEFAULT_VERBOSITY_OPTION = 'progress'
"""
The default verbosity option
"""

VERBOSITY_OPTIONS = ['progress', 'loss', 'policy', 'init']
"""
The available verbosity options:

* progress: show the training progress
* loss: show the logging information from loss calculation
* policy: show the logging information from policy generation
* init: show the logging information from initialization
"""

DEFAULT_VISUALIZER_TYPE = 'none'
"""
The default visualizer type
"""

VISUALIZER_TYPES = ['none', 'streamlit']

DEFAULT_PERSIST_PROGRESS_OPTION = 'all'
PERSIST_PROGRESS_OPTIONS = ['none', 'all']

DEFAULT_PAUSE_TIME = 0
"""
The default value for pausing before execution starts
to make time for screen recording. It's only available
in testing mode since it's pointless to do so while
training.
"""

DEFAULT_MIN_STEPS = 10
"""
The minimum number of steps the evaluation should run
per episode so that the tester can better visualize how
the agent is doing.
"""
