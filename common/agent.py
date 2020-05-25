
class Agent:
    """
    Base class for agents.
    The main API methods that users of this class need to know are:
        train
        eval
        reset
        respond
        learn
        save
        load
    These methods must be implemented(except train, eval, save, load)
    and can be publicly accessed.
    """
    def __init__(self, n_obs, n_act):
        """
        :param n_obs:  the size of observation space
        :param n_act:  the size of action space
        """
        self.n_obs = n_obs
        self.n_act = n_act

    def train(self):
        """
        indicating that the agent should work on training mode.
        """
        return

    def eval(self):
        """
        indicating that the agent should work on eval mode, not caring about training
        """
        return

    def reset(self):
        """
        reset the agent, forget memory if it has.
        """
        raise NotImplementedError

    def respond(self, obs):
        """
        respond an action, based on the observation and agent's memory(if has)
        :param obs: (int)
        :return: action (int)
        """
        raise NotImplementedError

    def learn(self, obs, next_obs, action, reward, done, target_act=None):
        """
        after responding to an observation, agent can learn from the rewards or golden action(if has).
        :param obs: (int)
        :param next_obs: (int) after the agent take the action, the next obs that the env returns
        :param action: the agent's action responding to the observation. (int)
        :param reward:
        :param done: if the episode is end after taking the action.
        :param target_act: the golden action (if has) responding to the observation.
        """
        raise NotImplementedError

    def save(self, dir, name):
        """
        save the agent's parameters and configs to the specified path
        2 saved files, ${name}_params, ${name}_configs
        :param dir: (str)
        :param name: agent's name
        """
        return

    def load(self, path):
        """
        load pre-trained parameters of the agent
        :param path: (str)
        """
        return