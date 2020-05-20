import os
import sys
from itertools import cycle
from collections import deque
from ControllerFactory.Controller import Controller
os.environ['SUMO_HOME'] = "/usr/share/sumo"
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


class DiscreteDurationController(Controller):
    def __init__(self, traffic_id, net_data, controller_config):
        Controller.__init__(self, traffic_id, net_data, controller_config)
        # cycle
        self.cycle = cycle(self.green_phases)
        self.gmax = controller_config.g_max
        self.gmin = controller_config.g_min
        self.mid = ((self.gmax - self.gmin) / 2.0) + self.gmin
        self.interval = self.gmax - self.mid
        self.action_numb = 9

    def step(self, action):
        if action == 0:
            trans_a = -1
        elif action == 1:
            trans_a = -0.75
        elif action == 2:
            trans_a = -0.5
        elif action == 3:
            trans_a = -0.25
        elif action == 4:
            trans_a = 0
        elif action == 5:
            trans_a = 0.25
        elif action == 6:
            trans_a = 0.5
        elif action == 7:
            trans_a = 0.75
        elif action == 8:
            trans_a = 1
        self.data = self.get_vehicle_subscription_data()
        if self.phase_time == 0:
            # get new phase and duration
            next_phase = self.next_phase()
            traci.trafficlight.setRedYellowGreenState(self.id, next_phase)
            self.phase = next_phase
            self.phase_time = self.next_phase_duration(trans_a)
        self.phase_time -= 1
        self.t += 1

    def next_phase_duration(self, action):
        if self.phase in self.green_phases:
            t = int((action * self.interval) + self.mid)
            return t
        elif 'y' in self.phase:
            return self.yellow_t
        else:
            return self.red_t

    # most important phase
    def next_phase(self):
        # if there are no phase waiting to execute in the queue, add more
        # else do nothing
        if len(self.phase_deque) == 0:
            next_phase = next(self.cycle)
            phases = self.get_intermediate_phases(self.phase, next_phase)
            self.phase_deque.extend(phases+[next_phase])
        return self.phase_deque.popleft()

    def reset(self):
        self.t = 0
        self.phase_deque = deque()
        self.cycle = cycle(self.green_phases)
        traci.junction.subscribeContext(self.id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 150,
                                        [traci.constants.VAR_LANEPOSITION,
                                         traci.constants.VAR_SPEED,
                                         traci.constants.VAR_LANE_ID])
        return self.get_state()
