"""
人手で価値関数を決めるエージェントの例
"""
import gym
from gym.wrappers import Monitor


class Agent:
    def __init__(self, env):
        self.__env = env

    def act(self, obs):
        """
        価値関数をここで定義する
        """
        cart_position, cart_velocity, pole_angle, pole_velocity = obs

        # 左に移動する場合の価値関数の値
        score_0 = cart_velocity + cart_position

        # 右に移動する場合の価値関数の値
        score_1 = pole_angle * 25 + pole_velocity
        return 0 if score_0 >= score_1 else 1


def main():
    env_name = 'CartPole-v0'  # 倒立振子
    # env_name = 'Pendulum-v0'  # 振り子
    env = get_env(env_name)  # 環境を作成
    agent = get_agent(env)  # 環境の情報をもとにagentを作成

    obs = env.reset()  # 初期化して、初期状態を取得
    cnt = 0
    while True:
        cnt += 1
        env.render()
        action = agent.act(obs)
        obs, _, done, _ = env.step(action)
        if done:
            break  # ゲームオーバーになると終了
    print(cnt, "step")  # どれだけ生存したかを表示
    env.env.close()
    env.close()


def get_agent(env):
    return Agent(env)


def get_env(env_name):
    env = gym.make(env_name)
    # 最大ステップ数を実質取り除く
    env._max_episode_steps = 10 ** 100
    env._max_episode_seconds = 10 ** 100

    return Monitor(
        env=env,
        directory="_output",
        force=True
    )


if __name__ == '__main__':
    main()
