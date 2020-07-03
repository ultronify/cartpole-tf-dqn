"""
Utilities
"""
import time


def play_episodes(env, policy, render_option, num_eps, pause_time, min_steps):
    """
    Play episodes with the given policy

    :param min_steps: the minimum steps the game should be played
    :param pause_time: the time that should pause to prepare for screen recording
    :param env: the OpenAI gym environment
    :param policy: the policy that should be used to generate actions
    :param render_option: how the game play should be rendered
    :param num_eps: how many episodes should be played
    :return: average episode reward
    """
    env.reset()
    if pause_time != 0:
        for _ in range(int(pause_time / 0.01)):
            env.render()
            time.sleep(0.01)
    total_reward = 0.0
    for _ in range(num_eps):
        episode_reward = play_episode(env, policy, render_option, min_steps)
        total_reward += episode_reward
    avg_reward = total_reward / num_eps
    print('finished {0} episodes with average reward {1}'.format(num_eps, avg_reward))
    return avg_reward


def play_episode(env, policy, render_option, min_steps):
    """
    Play an episode with the given policy.

    :param min_steps: the minimum steps the game should be played
    :param env: the OpenAI gym environment
    :param policy: the policy that should be used to generate actions
    :param render_option: how the game play should be rendered
    :return: episode reward
    """
    state = env.reset()
    done = False
    episode_reward = 0.0
    step_cnt = 0
    while not done or step_cnt < min_steps:
        if render_option == 'collect':
            env.render()
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state
        step_cnt += 1
    print('episode finished with reward {0}'.format(episode_reward))
    return episode_reward


def collect_steps(env, policy, buffer, render_option, current_state, n_steps):
    """
    Collects a single step from the game environment with policy specified. It is
    currently not used in favor of collect_episode API.

    :param n_steps: the number of steps to collect
    :param current_state: the current state of the environment
    :param env: OpenAI gym environment
    :param policy: DQN agent policy
    :param buffer: reinforcement learning replay buffer
    :param render_option: (bool) if should render the game play
    :return: None
    """
    state = current_state
    for _ in range(n_steps):
        if render_option == 'collect':
            env.render()
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -1.0
            state = env.reset()
        else:
            state = next_state
        buffer.record(state, reward, next_state, action, done)
    return state


def collect_episode(env, policy, buffer, render_option):
    """
    Collect steps from a single episode play and record
    with replay buffer

    :param env: OpenAI gym environment
    :param policy: DQN agent policy
    :param buffer: reinforcement learning replay buffer
    :param render_option: (bool) if should render the game play
    :return: None
    """
    state = env.reset()
    done = False
    while not done:
        if render_option == 'collect':
            env.render()
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -1.0
        buffer.record(state, reward, next_state, action, done)
        state = next_state


def compute_avg_reward(env, policy, num_episodes):
    """
    Compute the average reward across num_episodes under policy

    :param env: OpenAI gym environment
    :param policy: DQN agent policy
    :param num_episodes: the number of episode to take average from
    :return: (int) average reward
    """
    total_return = 0.0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            if done:
                reward = -1.0
            episode_return += reward
            state = next_state
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return
