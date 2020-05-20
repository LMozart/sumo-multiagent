from PoolFactory.PoolFactory import PoolFactory
from AgentFactory.AgentFactory import AgentFactory
from Utils.ConfigAllocator import Allocator
from EnvironmentFacade.SUMO import SUMO
from datetime import datetime
import tensorflow as tf
import os, sys
from Utils.networkdata import NetworkData

os.environ['SUMO_HOME'] = "/usr/share/sumo"
# os.environ['SUMO_HOME'] = "/usr/local/opt/sumo/share/sumo/"
try:
    import traci
    import traci.constants as tc
except ImportError:
    if "SUMO_HOME" in os.environ:
        sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
        import traci
        import traci.constants as tc
    else:
        raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")


class Trainer:
    def __init__(self, args):
        date = datetime.now().strftime('%Y%m%d_%H%M%S')
        # make save model
        model_dir = "/media/dennis/DATA/ModelpLog/model/{}_{}".format(args.trainer_type, date)
        log_dir = "/media/dennis/DATA/ModelpLog/log/{}_{}".format(args.trainer_type, date)
        # model_dir = "model/{}_{}".format(args.trainer_type, date)
        # log_dir = "log/{}_{}".format(args.trainer_type, date)
        if not os.path.exists("model"):
            os.makedirs("model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists("log"):
            os.makedirs("log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.model_dir = model_dir
        self.log_dir = log_dir

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

        self.tf_reward = tf.Variable(0, dtype=tf.float32)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.tf_reward_op = tf.summary.scalar('reward', self.tf_reward)
        self.reward_writer = tf.summary.FileWriter(log_dir, self.sess.graph)

    def run(self):
        raise NotImplementedError

    def train(self, t):
        raise NotImplementedError()

    def next_state_bootstrap(self, next_states, terminals, index):
        raise NotImplementedError()

    def save(self, path, t):
        raise NotImplementedError()

    def load(self, path, t):
        raise NotImplementedError()
