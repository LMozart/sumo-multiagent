from AgentSet.DDPG import DDPG
from AgentSet.DQN import MLPLight
from AgentSet.A2C import A2C
from AgentSet.LDQN import LDQN


class AgentFactory:
    def __init__(self, agent_config, numb_a, action_size, state_size):
        self.agent_config = agent_config
        self.numb_a = numb_a
        self.action_size = action_size
        self.state_size = state_size

    def get_agent(self):
        if self.agent_config.algo == "DDPG":
            return DDPG(self.agent_config, self.numb_a, self.action_size, self.state_size)
        elif self.agent_config.algo == "A2C":
            return A2C(self.agent_config, self.numb_a, self.action_size, self.state_size)
        elif self.agent_config.algo == "MLPLight":
            return MLPLight(self.agent_config, self.numb_a, self.action_size, self.state_size)
        elif self.agent_config.algo == "LDQN":
            return LDQN(self.agent_config, self.numb_a, self.action_size, self.state_size)
        else:
            print("Sorry but we do not implement this algorithm")
            raise NotImplementedError
