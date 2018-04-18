"""
actionについて

倒立振子の場合、右か左か2値
振り子の場合、どれくらい力を加えるかの連続値(-2から2)

環境ごとに違うので注意

https://github.com/openai/gym/wiki/Table-of-environments
"""
import gym
from gym.wrappers import Monitor


def get_env():
    env_name = 'CartPole-v0'
    # env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    return Monitor(
        env=env,
        directory="_output",
        force=True
    )


def main():
    env = get_env()
    print(env.action_space)

    for _ in range(10):
        print(env.action_space.sample())


if __name__ == '__main__':
    main()
