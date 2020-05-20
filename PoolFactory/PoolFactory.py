from PoolSet.MultiAgentPool import MultiAgentPool
from PoolSet.SerialPool import SerialPool
from PoolSet.SinglePool import SinglePool
from PoolSet.LSinglePool import LSinglePool
from PoolSet.PERSinglePool import PERSinglePool


class PoolFactory:
    def __init__(self, pool_config, numb_a):
        self.pool_type = pool_config.pool_type
        self.pool_size = pool_config.max_size
        self.batch_size = pool_config.batch_size
        self.numb_a = numb_a

    def get_pool(self):
        if self.pool_type == "multi":
            return MultiAgentPool(self.numb_a, self.batch_size, self.pool_size)
        if self.pool_type == "serial":
            return SerialPool(self.numb_a, self.batch_size, self.pool_size)
        if self.pool_type == "single":
            return SinglePool(self.numb_a, self.batch_size, self.pool_size)
        if self.pool_type == "per":
            return PERSinglePool(self.numb_a, self.batch_size, self.pool_size)
        if self.pool_type == "latency":
            return LSinglePool(self.numb_a, self.batch_size, self.pool_size)
        else:
            print("Error: No Such A Pool Type")
            raise NotImplementedError()
