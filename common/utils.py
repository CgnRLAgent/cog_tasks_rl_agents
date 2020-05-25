import numpy as np
import matplotlib.pyplot as plt
import random
import os
from sklearn.metrics import balanced_accuracy_score, f1_score


def train(env, agent, N, custom_reward=None, print_progress=True, seed=None):
    """
    record every episode's reward, action accuracy and f1 over iteration.
    :param env:
    :param agent:
    :param N:
    :param custom_reward: [number func(reward)] input the reward from env, output the modified reward
    :param print_progress:
    :param seed:
    :return: reward[], accuracy[], f1[]
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)

    agent.train()

    tr_rewards = []
    tr_accs = []
    tr_f1 = []

    for i in range(1, N + 1):
        if print_progress:
            print('\ntraining case [%d]' % i)
        obs = env.reset()
        agent.reset()
        done = False

        ep_reward = 0.
        ep_act_target = []
        ep_act_agent = []

        while not done:
            action = agent.respond(obs)
            next_obs, reward, done, info = env.step(action)
            if custom_reward is not None:
                reward = custom_reward(reward)
            target_act = info["target_act"]
            agent.learn(obs, next_obs, action, reward, done, target_act)
            obs = next_obs
            # record
            ep_reward += reward
            ep_act_agent.append(action)
            ep_act_target.append(target_act)

        tr_rewards.append(ep_reward)
        ep_acc = balanced_accuracy_score(ep_act_target, ep_act_agent)
        tr_accs.append(ep_acc)
        ep_f1 = f1_score(ep_act_target, ep_act_agent, average='macro')
        tr_f1.append(ep_f1)

        if print_progress:
            env.render()

    print('\ntraining end.')
    return tr_rewards, tr_accs, tr_f1


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


def train_results_plots(dir, figname, names, numbers, smooth=51, xlim=None):
    """
    plots training results with iterations: rewards, accuracies, f1-score (every n iterations)
    :param dir: save the figures to
    :param figname
    :param names: [str, ...] the names of the agents to be compared
    :param numbers: [(rewards, accs), ...] the training results
    """
    if not os.path.exists(dir):
        os.mkdir(dir)
    figname = os.path.join(dir, figname)

    def _smooth(p):
        r = (smooth-1) // 2
        sp = p[:]
        size = len(p)
        for i in range(size):
            begin = np.max([0, i-r])
            end = np.min([size-1, i+r]) + 1
            sp[i] = np.mean(p[begin:end])
        return sp

    # plot rewards
    plt.figure(figsize=(14, 8))
    rewards = [x[0] for x in numbers]
    assert len(rewards) == len(names)
    plt.title('Rewards')
    for r in rewards:
        plt.plot(_smooth(r))
    plt.legend(names, loc='lower right')
    plt.xlabel('iterations')
    plt.savefig(figname + '_rewards.jpg')
    # plot accuracies
    plt.figure(figsize=(14, 8))
    accs = [x[1] for x in numbers]
    assert len(accs) == len(names)
    plt.title('Accuracy')
    for a in accs:
        plt.plot(_smooth(a))
    plt.legend(names, loc='lower right')
    plt.xlabel('iterations')
    plt.savefig(figname + '_accs.jpg')
    # plot f1
    plt.figure(figsize=(14, 8))
    f1 = [x[2] for x in numbers]
    assert len(f1) == len(names)
    plt.title('F1-score')
    for f in f1:
        if xlim is not None:
            f = f[:xlim]
        plt.plot(_smooth(f))
    plt.legend(names, loc='lower right')
    plt.xlabel('iterations')
    plt.savefig(figname + '_f1.jpg')


def test(env, agent, N, print_progress=True, seed=None):
    """
    :param env:
    :param agent:
    :param N:
    :param print_progress:
    :return: avg_reward, avg_accuracy, avg_f1, eps_acc
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)

    agent.eval()

    rewards = []
    accs = []
    f1 = []
    eps_acc = 0  # accuracy of episodes

    for i in range(1, N + 1):
        if print_progress:
            print('test case [%d]' % i)
        obs = env.reset()
        agent.reset()
        done = False

        ep_reward = 0.
        ep_act_target = []
        ep_act_agent = []
        ep_correct = True

        while not done:
            action = agent.respond(obs)
            next_obs, reward, done, info = env.step(action)
            target_act = info["target_act"]
            obs = next_obs
            # record
            ep_reward += reward
            ep_act_agent.append(action)
            ep_act_target.append(target_act)
            if target_act != action:
                ep_correct = False

        rewards.append(ep_reward)
        ep_acc = balanced_accuracy_score(ep_act_target, ep_act_agent)
        accs.append(ep_acc)
        ep_f1 = f1_score(ep_act_target, ep_act_agent, average='macro')
        f1.append(ep_f1)
        if ep_correct:
            eps_acc += 1

        if print_progress:
            env.render()

    avg_reward = np.mean(rewards)
    avg_acc = np.mean(accs)
    avg_f1 = np.mean(f1)
    eps_acc /= N

    print('\ntest end.')
    print('episode accuracy: %.3f, avg reward: %.3f, avg accuracy: %.4f, avg f1: %.4f' % (eps_acc, avg_reward, avg_acc, avg_f1))

    return avg_reward, avg_acc, avg_f1
