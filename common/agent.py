
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
    These methods must be implemented, and can be publicly accessed.
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
        raise NotImplementedError

    def eval(self):
        """
        indicating that the agent should work on eval mode, not caring about training
        """
        raise NotImplementedError

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

    def learn(self, obs, action, reward, done, target_act=None):
        """
        after responding to an observation, agent can learn from the rewards or golden action(if has).
        :param obs: (int)
        :param action: the agent's action reponding to the observation. (int)
        :param reward:
        :param done: if the episode is end after taking the action.
        :param target_act: the golden action responding to the observation.
        """
        raise NotImplementedError

    def save(self, path):
        """
        save the agent's parameters to the specified path
        :param path: (str)
        """
        raise NotImplementedError

    def load(self, path):
        """
        load pre-trained parameters of the agent
        :param path: (str)
        """
        raise NotImplementedError