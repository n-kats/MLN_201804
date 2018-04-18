"""
とりあえず動かしてみる
"""
import time

import gym


def main():
    env = gym.make('CartPole-v0')  # 倒立振子
    # env = gym.make('Pendulum-v0')  # 振り子
    # env = gym.make('SpaceInvaders-v0')  # SpaceInvaders
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        if done:
            break
        time.sleep(0.1)
    env.close()


if __name__ == '__main__':
    main()
