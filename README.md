# Cog_tasks_RL_agents
Create cognitive tasks for Reinforcement Learning agents and benchmark them

## Background
Cognitive neuroscientists run a number of experiments in the lab to probe animal and human behaviour. But, machine learning / reinforcement learning (RL) researchers use very different benchmarks to evaluate their learning agents.To make it easier to compare the behavior of animals / humans with these agents, we need to implement the cognitive neuroscience tasks in environments that are accessible to artificial reinforcement learning agents.

What is known:
* The performance of machine learning agent on machine learning task
* The performance of cognitive agent on cognitive task

What is unknown:
* The performance of machine learning agent on cognitive task
* The performance of the cognitive agent on machine learningtask.

## Usage
All agents inherit from the basic `Agent` class in [agent.py](common/agent.py). If you want to use the agents to train on any of the gym environment, please see the [example.py](example.py).


## Agents
5 agents are implemented in this project:
* AuGMEnT
* LSTM
* DQN
* HER
* Monte Carlo

## Tasks
Implemented in the OpenAI gym style. They are put in a independent repo [here](https://github.com/CgnRLAgent/cog_ml_tasks).
* 12_AX
* 12_AX_S
* AX_CPT
* 12_AX_CPT
* Copy
* Copy_repeat
* Saccades
* Sequential Prediction

## Benchmark
Every agent is trained and evaluated on each of the tasks.
