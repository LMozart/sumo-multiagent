from PoolFactory.PoolFactory import PoolFactory
from AgentFactory.AgentFactory import AgentFactory
from Utils.ConfigAllocator import Allocator
from EnvironmentFacade.SUMO import SUMO
import tensorflow as tf
from Utils.networkdata import NetworkData


class Test:
    def __init__(self, args):
        allocator = Allocator(args)
        env_config, controller_config, pool_config, agent_config, trainer_config = allocator.get_config()
        # controllers relevant
        self.traffic_light_ids = []
        # traffic environment relevant
        self.net_data = NetworkData(args.net_fp).get_net_data()

        self.env = SUMO(env_config, controller_config, self.net_data)

        self.numb_a = self.env.numb_ctrl

        pool_factory = PoolFactory(pool_config, self.numb_a)
        agent_factory = AgentFactory(agent_config, self.numb_a, self.env.action_numb, self.env.input_numb)

        self.pool = pool_factory.get_pool()
        self.agents = agent_factory.get_agent()

        self.n_batch = self.pool.batch_size
        self.gamma = trainer_config.gamma
        self.epoch = trainer_config.epoch
        self.learn_mark = trainer_config.learn_mark

    def run(self):
        raise NotImplementedError

    def next_state_bootstrap(self, next_states, terminals, index):
        raise NotImplementedError()

    def load(self, path):
        raise NotImplementedError()
