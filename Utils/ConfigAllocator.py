class Allocator:
    def __init__(self, args):
        self.__pool_config = PoolConfig(args.max_size, args.pool_type, args.batch)
        self.__env_config = EnvConfig(args.cfg_fp, args.net_fp, args.nogui)
        self.__agent_config = AgentConfig(args.lrc, args.lra, args.TAU, args.eps, args.trainer_type, args.batch)
        self.__controller_config = ControllerConfig(args.metric_mode, args.controller_type,
                                                    args.y, args.g_min, args.g_max, args.r, args.max_step)
        self.__trainer_config = TrainerConfig(args.epoch, args.learn_mark, args.gamma)

    def get_config(self):
        return self.__env_config, self.__controller_config, self.__pool_config, self.__agent_config, \
               self.__trainer_config


class AgentConfig:
    def __init__(self, lrc, lra, TAU, eps, trainer_type, batch):
        self.lrc = lrc
        self.lra = lra
        self.TAU = TAU
        self.eps = eps
        self.algo = trainer_type
        self.batch = batch


class ControllerConfig:
    def __init__(self, metric_mode, controller_type, yellow_t, green_t, g_max, red_t, max_step):
        self.metric_mode = metric_mode
        self.controller_type = controller_type
        self.yellow_t = yellow_t
        self.green_t = green_t
        self.red_t = red_t
        self.g_max = g_max
        self.g_min = green_t
        self.max_step = max_step


class EnvConfig:
    def __init__(self, cfg_fp, net_fp, nogui):
        self.cfg_fp = cfg_fp
        self.net_fp = net_fp
        self.nogui = nogui


class PoolConfig:
    def __init__(self, max_size, pool_type, batch_size):
        self.max_size = max_size
        self.pool_type = pool_type
        self.batch_size = batch_size


class TrainerConfig:
    def __init__(self, epoch, learn_mark, gamma):
        self.epoch = epoch
        self.gamma = gamma
        self.learn_mark = learn_mark


