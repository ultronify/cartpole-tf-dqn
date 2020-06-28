import gym
import config
from dqn_agent import DqnAgent
from replay_buffer import DqnReplayBuffer
from utils import collect_steps, compute_avg_reward, collect_episode


def train_model(
        num_iterations=config.default_num_iterations,
        collect_steps_per_iteration=config.default_collect_steps_per_iteration,
        batch_size=config.default_batch_size,
        max_replay_history=config.default_max_replay_history,
        gamma=config.default_gamma,
        eval_eps=config.default_eval_eps,
        learning_rate=config.default_learning_rate,
        verbose=False,
):
    env_name = config.default_env_name
    train_env = gym.make(env_name)
    eval_env = gym.make(env_name)
    agent = DqnAgent(state_space=train_env.observation_space.shape[0], action_space=train_env.action_space.n,
                     gamma=gamma, verbose=verbose, lr=learning_rate)
    benchmark_reward = compute_avg_reward(eval_env, agent.random_policy, eval_eps)
    buffer = DqnReplayBuffer(max_size=max_replay_history)
    for eps_cnt in range(num_iterations):
        # collect_steps(train_env, agent.policy, buffer, collect_steps_per_iteration)
        collect_episode(train_env, agent.policy, buffer)
        if buffer.can_sample_batch(batch_size):
            state_batch, next_state_batch, action_batch, reward_batch, done_batch = buffer.sample_batch(
                batch_size=batch_size)
            loss = agent.train(state_batch=state_batch, next_state_batch=next_state_batch, action_batch=action_batch,
                               reward_batch=reward_batch, done_batch=done_batch, batch_size=batch_size)
            avg_reward = compute_avg_reward(eval_env, agent.policy, eval_eps)
            if verbose != 'none':
                print(
                    'Episode {0}/{1}({2}%) finished with avg reward {3} w/ benchmark reward {4} and '
                    'buffer volume {5}'.format(
                        eps_cnt, num_iterations,
                        round(eps_cnt / num_iterations * 100.0, 2),
                        avg_reward, benchmark_reward, buffer.get_volume()))
        else:
            if verbose != 'none':
                print('Not enough sample, skipping...')
    train_env.close()
    eval_env.close()
