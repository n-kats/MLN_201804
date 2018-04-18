"""
ChainerRLを用いた例

オリジナルはChaienrRLのサンプル
https://github.com/chainer/chainerrl/blob/master/examples/quickstart/quickstart.ipynb

簡単にいじれるものは
* ステップ数
* 学習率
* 割引率
* ネットワーク

"""
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
from gym.wrappers import Monitor
import numpy as np


def main():
    env = get_env()
    agent = get_agent(env)
    train(agent, env)
    agent.save('_agent')  # 保存
    env.close()

    env_test = get_env(is_train=False)
    test(agent, env_test)  # 訓練した結果でagentを動かす


def get_env(is_train=True):
    env = gym.make('CartPole-v0')
    # 最大ステップ数を実質取り除く
    env._max_episode_steps = 10 ** 100
    env._max_episode_seconds = 10 ** 100

    if is_train:
        return env  # 訓練時は動画を出力しない

    return Monitor(
        env=env,
        directory="_output",
        force=True
    )


def get_agent(env):
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    q_func = QFunction(obs_size, n_actions)
    optimizer = chainer.optimizers.Adam(eps=1e-3)
    optimizer.setup(q_func)

    # 割引率
    gamma = 0.95

    # epsilon-greedyの設定
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.3,
        random_action_func=env.action_space.sample
    )

    # Experience Replayの設定
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

    # gymがnp.float64で出力するが、
    # chainerはnp.float32しか入力できない。
    # そのための変換。
    def phi(x):
        return x.astype(np.float32, copy=False)

    agent = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer,
        gamma, explorer,
        replay_start_size=500, update_interval=1,
        target_update_interval=100, phi=phi
    )
    return agent


class QFunction(chainer.Chain):
    """
    Q^*をDNNを使って近似する
    """
    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, obs, test=False):
        h = F.tanh(self.l0(obs))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))


def train(agent, env):
    # train
    n_episodes = 200  # 訓練の回数
    max_episode_len = 2000  # このステップ数で一旦終了させる
    for i in range(1, n_episodes + 1):
        obs = env.reset()
        reward = 0
        done = False
        sum_of_rewards = 0
        for _ in range(max_episode_len):
            env.render()
            action = agent.act_and_train(obs, reward)  # 訓練
            obs, reward, done, _ = env.step(action)
            sum_of_rewards += reward
            if done:
                break
        if i % 10 == 0:
            print(
                'episode:', i,
                'Rewards:', sum_of_rewards,
                'statistics:', agent.get_statistics()
            )
        agent.stop_episode_and_train(obs, reward, done)


def test(agent, env):
    for i in range(10):
        obs = env.reset()
        sum_of_rewards = 0
        while True:
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
