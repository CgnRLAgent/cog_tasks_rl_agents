import numpy as np
import matplotlib.pyplot as plt
import random
from time import time
import os


def train(env, agent, N, custom_reward=None, print_progress=True, seed=None, timing_interval=10):
    """
    record every episode's reward, action accuracy (if target action is given),
    and accumulative time cost.
    :param env:
    :param agent:
    :param N:
    :param custom_reward: [number func(reward)] input the reward from env, output the modified reward
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
            if custom_reward is not None:
                reward = custom_reward(reward)
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

    print('\ntraining end. time elapsed: %.2f seconds' % (time()-start))

    return tr_rewards, tr_accs, tr_timings


def save_train_res(path, results):
    """
    :param results: return of train()
    """
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.mkdir(dir)
    np.save(path, np.array(results), allow_pickle=True)


def load_train_res(path):
    r = np.load(path, allow_pickle=True)
    return r[0], r[1], r[2]


def train_results_plots(dir, figname, names, numbers):
    """
    plots training results with iterations: rewards, accuracies
    and timing (every n iterations)
    :param dir: save the figures to
    :param figname
    :param names: [str, ...] the names of the agents to be compared
    :param numbers: [(rewards, accs, timings), ...] the training results
    """
    if not os.path.exists(dir):
        os.mkdir(dir)
    figname = os.path.join(dir, figname)
    # plot rewards
    plt.figure(figsize=(14, 8))
    rewards = [x[0] for x in numbers]
    assert len(rewards) == len(names)
    plt.title('Rewards')
    for r in rewards:
        plt.plot(r)
    plt.legend(names, loc='lower right')
    plt.xlabel('iterations')
    plt.savefig(figname + '_rewards.jpg')
    # plot accuracies
    plt.figure(figsize=(14, 8))
    accs = [x[1] for x in numbers]
    assert len(accs) == len(names)
    plt.title('Accuracy')
    for a in accs:
        plt.plot(a)
    plt.legend(names, loc='lower right')
    plt.xlabel('iterations')
    plt.savefig(figname + '_accs.jpg')
    # plot timing
    plt.figure(figsize=(14, 8))
    timings = [x[2] for x in numbers]
    assert len(timings) == len(names)
    plt.title('Timing')
    for t in timings:
        x = [r[0] for r in t]
        y = [r[1] for r in t]
        plt.plot(x, y)
    plt.legend(names, loc='lower right')
    plt.ylabel('seconds')
    plt.xlabel('iterations')
    plt.savefig(figname + '_timing.jpg')


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

    print('\ntest end. time elapsed: %.2f seconds' % time_cost)
    print('avg reward: %.2f, avg accuracy: %.4f' % (avg_reward, avg_acc))

    return avg_reward, avg_acc, time_cost
