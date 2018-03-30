"""
original
https://github.com/chainer/chainerrl/blob/master/examples/quickstart/quickstart.ipynb
"""
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np


def main():
    env = get_env()
    agent = get_agent(env)
    train(agent, env)
    test(agent, env)
    agent.save('agent')


def get_env():
    return gym.make('CartPole-v0')


def get_agent(env):
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    q_func = QFunction(obs_size, n_actions)
    # _q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
    #     obs_size, n_actions,
    #     n_hidden_layers=2, n_hidden_channels=50)  # ??

    optimizer = chainer.optimizers.Adam(eps=1e-2)

    gamma = 0.95

    # Use epsilon-greedy for exploration
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.3, random_action_func=env.action_space.sample)

    # DQN uses Experience Replay.
    # Specify a replay buffer and its capacity.
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

    # Since observations from CartPole-v0 is numpy.float64 while
    # Chainer only accepts numpy.float32 by default, specify
    # a converter as a feature extractor function phi.

    def phi(x):
        return x.astype(np.float32, copy=False)

    # Now create an agent that will interact with the environment.
    agent = chainerrl.agents.DoubleDQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=500, update_interval=1,
        target_update_interval=100, phi=phi)
    optimizer.setup(q_func)
    return agent


class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))


def train(agent, env):
    # train
    n_episodes = 200
    max_episode_len = 200
    for i in range(1, n_episodes + 1):
        obs = env.reset()
        reward = 0
        done = False
        sum_of_rewards = 0
        for _ in range(max_episode_len):
            env.render()
            action = agent.act_and_train(obs, reward)  # 便利
            obs, reward, done, _ = env.step(action)
            sum_of_rewards += reward
            if done:
                break
        if i % 10 == 0:
            print('episode:', i,
                  'Rewards:', sum_of_rewards,
                  'statistics:', agent.get_statistics())
        agent.stop_episode_and_train(obs, reward, done)


def test(agent, env):
    for i in range(10):
        obs = env.reset()
        done = False
        sum_of_rewards = 0
        for _ in range(200):
            env.render()
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            sum_of_rewards += reward
            if done:
                break
        print('test episode:', i, 'Rewards:', sum_of_rewards)
        agent.stop_episode()


if __name__ == '__main__':
    main()

# 別パターン
# gym.undo_logger_setup()  # Turn off gym's default logger settings
# logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
#
# chainerrl.experiments.train_agent_with_evaluation(
#     agent, env,
#     steps=2000,           # Train the agent for 2000 steps
#     eval_n_runs=10,       # 10 episodes are sampled for each evaluation
#     max_episode_len=200,  # Maximum length of each episodes
#     eval_interval=1000,   # Evaluate the agent after every 1000 steps
#     outdir='result')      # Save everything to 'result' directory
