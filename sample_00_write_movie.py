"""
dockerでGUI共有をしない場合用に動画を出力する
"""
import time

import gym
from gym.wrappers import Monitor


def get_env():
    env = gym.make('CartPole-v0')
    # env = gym.make('SpaceInvaders-v0')
    return Monitor(
        env=env,
        directory="_output",
        force=True
    )


def main():
    env = get_env()
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        if done:
            break
        time.sleep(0.1)
    env.env.close()
    env.close()


if __name__ == '__main__':
    main()
