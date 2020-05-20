import argparse


def parse_cl_args():
    parser = argparse.ArgumentParser()

    # env params
    parser.add_argument("-netfp", type=str, default='data/double/double.net.xml', dest='net_fp',
                        help='path to desired simulation network file, default: networks/double.net.xml')
    parser.add_argument("-sumocfg", type=str, default='data/double/double.sumocfg', dest='cfg_fp',
                        help='path to desired simulation configuration file, default: networks/double.sumocfg')
    parser.add_argument("-nogui", default=False, action='store_true', dest='nogui', help='disable gui, default: False')

    # controller params
    parser.add_argument("-controller_type", type=str, default='duration control')
    parser.add_argument("-metric_mode", type=str, default='queue')
    parser.add_argument("-max_step", type=int, default=500)
    parser.add_argument("-gmin", type=int, default=5, dest='g_min', help='minimum green phase time (s), default: 5')
    parser.add_argument("-gmax", type=int, default=30, dest='g_max', help='maximum green phase time (s), default: 30')
    parser.add_argument("-y", type=int, default=2, dest='y', help='yellow change phase time (s), default: 2')
    parser.add_argument("-r", type=int, default=3, dest='r', help='all red stop phase time (s), default: 3')
    # # websters params
    parser.add_argument("-cmin", type=int, default=60, dest='c_min', help='minimum cycle time (s), default: 60')
    parser.add_argument("-cmax", type=int, default=180, dest='c_max', help='maximum cycle time (s), default: 180')
    parser.add_argument("-satflow", type=float, default=0.38, dest='sat_flow',
                        help='lane vehicle saturation rate (veh/s), default: 0.38')
    parser.add_argument("-f", type=int, default=900, dest='update_freq',
                        help='interval over which websters timing are computed (s), default: 900')

    # Agent params
    parser.add_argument("-eps", type=float, default=0.9, dest='eps',
                        help='reinforcement learning explortation rate, default: 0.01')
    parser.add_argument("-lra", type=float, default=0.0001, dest='lra',
                        help='ddpg actor/dqn neural network learning rate, default: 0.0001')
    parser.add_argument("-lrc", type=float, default=0.001, dest='lrc',
                        help='ddpg critic neural network learning rate, default: 0.001')
    parser.add_argument("-TAU", type=float, default=0.005, dest='TAU',
                        help='ddpg online/target weight shifting tau, default: 0.005')

    # Trainer params
    parser.add_argument("-trainer_type", type=str, dest="trainer_type")
    parser.add_argument("-epoch", type=int, default=15000, dest='epoch', help='times of the system to train')
    parser.add_argument("-learn_mark", type=int, default=500, help='time for system to start train')
    parser.add_argument("-gamma", type=float, default=0.99, dest='gamma', help='reward discount factor, default: 0.99')

    # Pool Params
    parser.add_argument("-pool_type", default='multi', type=str, dest='pool_type')
    parser.add_argument("-max_size", default=10000, type=int, dest='max_size')
    # parser.add_argument("-max_size", default=5000, type=int, dest='max_size')
    parser.add_argument("-batch", type=int, default=64, dest='batch',
                        help='batch size to sample from replay to train neural net, default: 32')

    args = parser.parse_args()
    return args
