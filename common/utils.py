import numpy as np
import matplotlib


def train(env, agent, N, print_progress=True):
    agent.train()
    for i in range(1, N + 1):
        if print_progress:
            print('training case [%d]' % i)
        obs = env.reset()
        agent.reset()
        done = False

        while not done:
            action = agent.respond(obs)
            next_obs, reward, done, info = env.step(action)
            # target action is only for supervised-learning style agent, e.g. LSTM
            target_act = info["target_act"] if "target_act" in info else None
            agent.learn(obs, action, reward, done, target_act)
            obs = next_obs
            # TODO: reward/accuracy record

        if print_progress:
            env.render()


def test(env, agent, N, print_progress=True):
    agent.eval()
    ep_rewards = []
    for i in range(1, N + 1):
        if print_progress:
            print('test case [%d]' % i)
        obs = env.reset()
        agent.reset()
        done = False

        ep_reward = 0.

        while not done:
            action = agent.respond(obs)
            next_obs, reward, done, info = env.step(action)
            ep_reward += reward
            obs = next_obs

        ep_rewards.append(ep_reward)
        if print_progress:
            env.render()

    # TODO: statistics (reward, accuracy; figures)
    print("test case: %d, avg reward: %f" % (N, np.mean(ep_rewards)))