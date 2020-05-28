import gym
import gym_cog_ml_tasks
from LSTM.LSTM_model import Agent_LSTM
import numpy as np
import random
from sklearn.metrics import f1_score
import torch


def train_and_val(agent, env_tr, env_val_full, env_val_minor, N_tr, N_val, n_val_iter=1000, seed=123):
    np.random.seed(seed)
    random.seed(seed)
    env_tr.seed(seed)

    tmp_ep_acc = []
    tmp_f1 = []
    # records
    tr_ep_acc = []
    tr_f1 = []
    val_full_ep_acc = []
    val_full_f1 = []
    val_minor_ep_acc = []
    val_minor_f1 = []

    # start training
    for i in range(N_tr):
        agent.train()
        agent.reset()
        obs = env_tr.reset()
        done = False
        ep_act_target = []
        ep_act_agent = []

        while not done:
            action = agent.respond(obs)
            next_obs, reward, done, info = env_tr.step(action)
            target_act = info["target_act"]
            ep_act_agent.append(action)
            ep_act_target.append(target_act)
            agent.learn(obs, next_obs, action, reward, done, target_act)
            obs = next_obs

        tmp_ep_acc.append(1 if ep_act_target == ep_act_agent else 0)
        tmp_f1.append(f1_score(ep_act_target, ep_act_agent, average='macro'))

        # validation
        if (i+1) % n_val_iter == 0:
            # first, save the training metrics averaged over the last n_val_iter iters.
            assert len(tmp_ep_acc) == len(tmp_f1) == n_val_iter
            tr_ep_acc.append(np.mean(tmp_ep_acc))
            tr_f1.append(np.mean(tmp_f1))
            tmp_ep_acc = []
            tmp_f1 = []
            # save the current random states
            _rdst = random.getstate()
            _nprdst = np.random.get_state()
            # validation over the full mode
            val_full_res = validation(agent, env_val_full, N_val, seed)
            val_full_ep_acc.append(val_full_res[0])
            val_full_f1.append(val_full_res[1])
            # validation over the minor mode
            val_minor_res = validation(agent, env_val_minor, N_val, seed)
            val_minor_ep_acc.append(val_minor_res[0])
            val_minor_f1.append(val_minor_res[1])
            # recover the training states
            random.setstate(_rdst)
            np.random.set_state(_nprdst)
            # print
            assert len(tr_ep_acc) == len(tr_f1) == len(val_full_ep_acc) == len(val_full_f1) == len(val_minor_ep_acc) == len(val_minor_f1)
            _last = len(tmp_ep_acc) - 1
            print("iteration %d: train ep_acc=%f, f1=%f; validation(full) ep_acc=%f, f1=%f; validation(minor) ep_acc=%f, f1=%f" % \
                  (i+1, tr_ep_acc[_last], tr_f1[_last], val_full_ep_acc[_last], val_full_f1[_last], val_minor_ep_acc[_last], val_minor_f1[_last]))

    return tr_ep_acc, tr_f1, val_full_ep_acc, val_full_f1, val_minor_ep_acc, val_minor_f1


def validation(agent, env, N_val, seed=123):
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    agent.eval()
    tmp_ep_acc = []
    tmp_f1 = []
    for j in range(N_val):
        obs = env.reset()
        agent.reset()
        done = False
        ep_act_target = []
        ep_act_agent = []
        while not done:
            action = agent.respond(obs)
            next_obs, reward, done, info = env.step(action)
            target_act = info["target_act"]
            obs = next_obs
            ep_act_agent.append(action)
            ep_act_target.append(target_act)
        tmp_ep_acc.append(1 if ep_act_target == ep_act_agent else 0)
        tmp_f1.append(f1_score(ep_act_target, ep_act_agent, average='macro'))

    return np.mean(tmp_ep_acc), np.mean(tmp_f1)


# main
N_tr = 15000
N_val = 1000

env_tr = gym.make('Simple_Copy_Repeat-v1', n_char=10, size=3, repeat=2)
env_val_full = gym.make('Simple_Copy_Repeat-v1', n_char=10, size=3, repeat=2)
env_val_full.setMode('full')
env_val_minor = gym.make('Simple_Copy_Repeat-v1', n_char=10, size=3, repeat=2)
env_val_minor.setMode('minor', n_exclude=1)

n_hidden = 50
n_layers = 1
lr = 0.001

# train on the full mode
print("-"*10 + "\nFULL MODE:")
env_tr.setMode('full')
torch.manual_seed(123)
agent = Agent_LSTM(env_tr.observation_space.n, env_tr.action_space.n, n_hidden, lr, n_layers)
res_full = train_and_val(agent, env_tr, env_val_full, env_val_minor, N_tr, N_val)

# train on the major mode with n_exclude=1 (9*9*9 = 729)
print("-"*10 + "\nMAJOR MODE (1):")
env_tr.setMode('major', n_exclude=1)
torch.manual_seed(123)
agent = Agent_LSTM(env_tr.observation_space.n, env_tr.action_space.n, n_hidden, lr, n_layers)
res_major_729 = train_and_val(agent, env_tr, env_val_full, env_val_minor, N_tr, N_val)

# train on the major mode with n_exclude=2 (8*8*8 = 512)
print("-"*10 + "\nMAJOR MODE (1):")
env_tr.setMode('major', n_exclude=2)
torch.manual_seed(123)
agent = Agent_LSTM(env_tr.observation_space.n, env_tr.action_space.n, n_hidden, lr, n_layers)
res_major_512 = train_and_val(agent, env_tr, env_val_full, env_val_minor, N_tr, N_val)


# save results
np.save('./res_full', np.array(res_full), allow_pickle=True)
np.save('./res_major_729', np.array(res_major_729), allow_pickle=True)
np.save('./res_major_512', np.array(res_major_512), allow_pickle=True)