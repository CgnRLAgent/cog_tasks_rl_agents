import numpy as np
import matplotlib
import random
from time import time


def train(env, agent, N, print_progress=True, seed=None, timing_interval=10):
    """
    record every episode's reward, action accuracy (if target action is given),
    and accumulative time cost.
    :param env:
    :param agent:
    :param N:
    :param print_progress:
    :param seed:
    :param timing_interval: timing every k iterations
    :return: reward[], accuracy[], timing[]
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)

    agent.train()

    tr_rewards = []
    tr_accs = []
    tr_timings = []

    start = time()
    for i in range(1, N + 1):
        if print_progress:
            print('\ntraining case [%d]' % i)
        obs = env.reset()
        agent.reset()
        done = False

        ep_reward = 0.
        ep_acc = 0.
        ep_output_len = 0

        while not done:
            action = agent.respond(obs)
            next_obs, reward, done, info = env.step(action)
            target_act = info["target_act"] if "target_act" in info else None
            agent.learn(obs, action, reward, done, target_act)
            obs = next_obs
            # record
            ep_reward += reward
            ep_output_len += 1
            if target_act is not None:
                ep_acc += 1 if action == target_act else 0

        ep_acc /= ep_output_len
        tr_rewards.append(ep_reward)
        tr_accs.append(ep_acc)

        if print_progress:
            env.render()

        if i % timing_interval == 0:
            tr_timings.append((i, time()-start))

    if print_progress:
        print('\ntraining end. time elapsed: %.2f seconds' % (time()-start))

    return tr_rewards, tr_accs, tr_timings


def test(env, agent, N, print_progress=True, seed=None):
    """
    :param env:
    :param agent:
    :param N:
    :param print_progress:
    :return: avg_reward, avg_accuracy, time_cost
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)

    agent.eval()

    rewards = []
    accs = []

    start = time()
    for i in range(1, N + 1):
        if print_progress:
            print('test case [%d]' % i)
        obs = env.reset()
        agent.reset()
        done = False

        ep_reward = 0.
        ep_acc = 0.
        ep_output_len = 0

        while not done:
            action = agent.respond(obs)
            next_obs, reward, done, info = env.step(action)
            target_act = info["target_act"] if "target_act" in info else None
            obs = next_obs
            # record
            ep_reward += reward
            ep_output_len += 1
            if target_act is not None:
                ep_acc += 1 if action == target_act else 0

        ep_acc /= ep_output_len
        rewards.append(ep_reward)
        accs.append(ep_acc)

        if print_progress:
            env.render()

    time_cost = time() - start
    avg_reward = np.mean(rewards)
    avg_acc = np.mean(accs)
    if print_progress:
        print('\ntest end. time elapsed: %.2f seconds' % time_cost)
        print('avg reward: %.2f, avg accuracy: %.4f' % (avg_reward, avg_acc))

    return avg_reward, avg_acc, time_cost
