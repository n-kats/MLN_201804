"""
人間が評価値を決めるエージェントの例
observationをもとにactionを決定
"""
import time

import gym
from gym.wrappers import Monitor


def main():
    env_name = 'CartPole-v0'  # 倒立振子
    # env_name = 'Pendulum-v0'  # 振り子
    env = get_env(env_name)  # 環境を作成
    agent = get_agent(env)  # 環境の情報をもとにagentを作成

    obs = env.reset()  # 初期化して、初期状態を取得
    # for _ in range(2000):
    cnt = 0
    while True:
        cnt += 1
        env.render()
        action = agent.act(obs)
        obs, _, done, _ = env.step(action)
        if done:
            break  # ゲームオーバーになると終了
    print(cnt)
    env.close()
    env.env.close()


def get_agent(env):
    return Agent(env)


class Agent:
    def __init__(self, env):
        self.__env = env

    def act(self, obs):
        cart_position, cart_velocity, pole_angle, pole_velocity = obs

        # 左に移動する場合の評価値
        score_0 = cart_velocity + cart_position

        # 右に移動する場合の評価値
        score_1 = pole_angle * 25 + pole_velocity
        return 0 if score_0 >= score_1 else 1


def get_env(env_name):
    env = gym.make(env_name)
    env._max_episode_steps = 10**100
    env._max_episode_seconds = 10**100
    return Monitor(
        env=env,
        directory="_output",
        force=True
    )


if __name__ == '__main__':
    main()
