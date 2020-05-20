import os
import sys
import numpy as np
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


class PressurePhaseController(Controller):
    """Abstract base class for all traffic signal controller.

    Build your own traffic signal controller by implementing the follow methods.
    """

    def __init__(self, traffic_id, net_data, controller_config):
        Controller.__init__(self, traffic_id, net_data, controller_config)
        # cycle
        self.cycle = cycle(self.green_phases)
        self.green_t = controller_config.g_min
        self.int_to_phase = self.int_to_input(self.green_phases)
        self.phase_lanes = self.get_phase_lanes(self.green_phases)
        # self.input_numb = len(self.green_phases)
        self.input_numb = self.phase_numb + \
                          (len(self.net_data['inter'][self.id]['incoming_lanes']) * 2) + \
                          (len(self.net_data['inter'][self.id]['inter_phases'])) + 1
        self.action_numb = len(self.green_phases)

    # Part S : initialize
    def int_to_input(self, phases):
        return {p: phases[p] for p in range(len(phases))}

    def reset(self):
        traci.junction.subscribeContext(self.id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 150,
                                        [traci.constants.VAR_LANEPOSITION,
                                         traci.constants.VAR_SPEED,
                                         traci.constants.VAR_LANE_ID])
        # clean up the action space
        self.phase_lanes = self.get_phase_lanes(self.green_phases)
        self.cycle = cycle(self.green_phases)
        self.phase_deque = deque()
        self.phase = self.all_red
        self.phase_time = 0
        self.t = 0
        return self.get_state()
        # clean up the end

    # def get_state(self):
    #     phase_pressure = []
    #     for g in self.green_phases:
    #         inc_lanes = self.max_pressure_lanes[g]['inc']
    #         out_lanes = self.max_pressure_lanes[g]['out']
    #         # pressure is defined as the number of vehicles in a lane
    #         inc_pressure = sum([len(self.data[l]) if l in self.data else 0 for l in inc_lanes])
    #         out_pressure = sum([len(self.data[l]) if l in self.data else 0 for l in out_lanes])
    #         phase_pressure.append(inc_pressure - out_pressure)
    #     return np.array(phase_pressure)

    def phase_lanes(self, actions):
        phase_lanes = {a: [] for a in actions}
        for a in actions:
            green_lanes = set()
            red_lanes = set()
            for s in range(len(a)):
                if a[s] == 'g' or a[s] == 'G':
                    green_lanes.add(self.net_data['inter'][self.id]['tlsindex'][s])
                elif a[s] == 'r':
                    red_lanes.add(self.net_data['inter'][self.id]['tlsindex'][s])
            # some movements are on the same lane, removes duplicate lanes
            pure_green = [l for l in green_lanes if l not in red_lanes]
            if len(pure_green) == 0:
                phase_lanes[a] = list(set(green_lanes))
            else:
                phase_lanes[a] = list(set(pure_green))
        return phase_lanes

    def step(self, action):
        self.data = self.get_vehicle_subscription_data()
        if self.phase_time == 0:
            # get new phase and duration
            next_phase = self.next_phase(action)
            traci.trafficlight.setRedYellowGreenState(self.id, next_phase)
            self.phase = next_phase
            self.phase_time = self.next_phase_duration()
        self.phase_time -= 1

        # if self.phase in self.green_phases:
        #     print(self.phase_time, self.phase, "green time")
        # elif 'y' in self.phase:
        #     print(self.phase_time, self.phase, "yellow time")
        # else:
        #     print(self.phase_time, self.phase, "red time")
        # check if reach ends
        if self.phase_time == 0 and self.phase in self.green_phases:
            end = True
        else:
            end = False
        self.t += 1
        return end

    def next_phase_duration(self):
        if self.phase in self.green_phases:
            return self.green_t
        elif 'y' in self.phase:
            return self.yellow_t
        else:
            return self.red_t

    # most important phase
    def next_phase(self, action):
        # if there are no phase waiting to execute in the queue, add more
        # else do nothing
        if len(self.phase_deque) == 0:
            next_phase = self.get_next_phase(action)
            phases = self.get_intermediate_phases(self.phase, next_phase)
            self.phase_deque.extend(phases+[next_phase])
        return self.phase_deque.popleft()

    def get_next_phase(self, action):
        # check which action lane groups have vehicles
        # check if any vehicles at intersection, if yes
        next_phase = self.int_to_phase[action]
        return next_phase
