"""
ランダムに動くエージェントの例

observationをもとにactionを決定

例えば、次のようなインターフェースを採用する
"""
import time

import gym
from gym.wrappers import Monitor


def get_agent(env):
    return Agent(env)


class Agent:
    def __init__(self, env):
        self.__env = env

    def act(self, obs):
        """
        状態に対して行動を返す
        """
        return self.__env.action_space.sample()


def main():
    env_name = 'CartPole-v0'
    # env_name = 'Pendulum-v0'
    env = get_env(env_name)

    agent = get_agent(env)  # 環境の情報をもとにagentを作成

    obs = env.reset()  # 初期化して、初期状態を取得
    for _ in range(200):
        env.render()
        action = agent.act(obs)
        obs, _, done, _ = env.step(action)
        if done:
            break  # ゲームオーバーになると終了
        time.sleep(0.1)
    env.env.close()
    env.close()


def get_env(env_name):
    env = gym.make(env_name)
    return Monitor(
        env=env,
        directory="_output",
        force=True
    )


if __name__ == '__main__':
    main()
