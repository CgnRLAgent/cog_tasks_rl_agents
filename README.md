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

## Agents
* AuGMEnT
* LSTM
* DQN
* HER
* Monte Carlo

## Tasks
Implemented in the OpenAI gym style. They are put in a independent repo [here](https://github.com/CgnRLAgent/cog_ml_tasks).
* 1_2AX (custom)
* 1_2AX_S (custom)
* AX_CPT (custom)
* Copy (gym)
* Copy_repeat (gym)
* Saccades (custom)
* Seq_prediction

## Benchmark
Every agent is trained and evaluated on each of the tasks.
