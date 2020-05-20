import os
import sys
import numpy as np
from collections import deque

# os.environ['SUMO_HOME'] = "/usr/share/sumo"
os.environ['SUMO_HOME'] = "/usr/local/opt/sumo/share/sumo/"
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


class Controller:
    def __init__(self, traffic_id, net_data, controller_config):
        self.max_speed = 1
        self.max_press = 1
        # phase
        self.id = traffic_id
        self.net_data = net_data
        self.phase_deque = deque()
        self.green_phases, self.inter_phases = self.get_tl_green_phases()
        self.all_red = len((self.green_phases[0])) * 'r'
        self.red_t = controller_config.red_t
        self.yellow_t = controller_config.yellow_t
        self.green_t = controller_config.green_t
        self.phase = self.all_red
        self.phase_lanes = self.get_phase_lanes(self.green_phases)
        self.phase_time = 0
        # traffic parse
        self.incoming_lanes = set()
        self.lane_lengths = None
        self.lane_speeds = None
        self.lane_travel_times = None
        self.v_info = {}
        self.t = 0
        self.set_incoming_lanes()
        self.max_pressure_lanes = {}

        self.max_speed_per_lane = {}
        for lane in self.incoming_lanes:
            self.max_speed_per_lane[lane] = 1

        # basic
        self.net_data['inter'][self.id]['incoming_lanes'] = self.incoming_lanes
        self.phase_to_one_hot = self.input_to_one_hot(self.green_phases + self.inter_phases + [self.all_red])
        self.lane_capacity = np.array([float(self.net_data['lane'][lane]['length']) / 7.5
                                       for lane in self.incoming_lanes])
        # for RL learning
        self.experience = {}
        self.max_step = controller_config.max_step
        self.metric_mode = controller_config.metric_mode
        self.phase_numb = len(self.net_data['inter'][self.id]['green_phases'])
        # self.input_numb = 2 + phase_numb + len(self.inter_phases) + 1
        # self.input_numb = len(self.net_data['inter'][self.id]['incoming_lanes'])
        self.max_pressure_lanes = self.get_max_pressure_lanes()
        # for metric
        if self.metric_mode == 'delay' or self.metric_mode == 'speed':
            self.old_v = set()
        elif self.metric_mode == 'queue':
            self.lane_queues = {lane: 0 for lane in self.incoming_lanes}
            self.stop_speed = 0.3

        # create subscription for this traffic signal junction to gather
        # vehicle information efficiently
        traci.junction.subscribeContext(self.id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 150,
                                        [traci.constants.VAR_LANEPOSITION,
                                         traci.constants.VAR_SPEED,
                                         traci.constants.VAR_LANE_ID])
        self.data = self.get_vehicle_subscription_data()

    def get_intermediate_phases(self, phase, next_phase):
        # phase is current phase
        # next phase is next phase
        if phase == next_phase or phase == self.all_red:
            return []
        else:
            yellow_phase = ''.join([p if p == 'r' else 'y' for p in phase])
            return [yellow_phase, self.all_red]

    def get_tl_green_phases(self):
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0]
        # get only the green phases
        green_phases = [p.state for p in logic.getPhases()
                        if 'y' not in p.state
                        and ('G' in p.state or 'g' in p.state)]
        # sort to ensure parity between sims (for RL actions)
        inter_phases = [p.state for p in logic.getPhases()
                        if 'y' in p.state]
        return sorted(green_phases), inter_phases

    def input_to_one_hot(self, phases):
        # first, create a identity matrix with a size of len(phases) x len(phases)
        # when one phase works, the other should be set to 0, means 'deactivate'.
        identity = np.identity(len(phases))
        one_hots = {phases[i]: identity[i, :] for i in range(len(phases))}
        return one_hots

    def set_incoming_lanes(self):
        for p in self.phase_lanes:
            for l in self.phase_lanes[p]:
                self.incoming_lanes.add(l)
        self.incoming_lanes = sorted(list(self.incoming_lanes))
        self.lane_lengths = {lane: self.net_data['lane'][lane]['length'] for lane in self.incoming_lanes}
        self.lane_speeds = {lane: self.net_data['lane'][lane]['speed'] for lane in self.incoming_lanes}
        self.lane_travel_times = {lane: self.lane_lengths[lane] / float(self.lane_speeds[lane])
                                  for lane in self.incoming_lanes}

    def get_vehicle_subscription_data(self):
        # use SUMO subscription to retrieve vehicle info in batches
        # around the traffic signal controller
        tl_data = traci.junction.getContextSubscriptionResults(self.id)
        # create empty incoming lanes for use else where
        lane_vehicles = {l: {} for l in self.incoming_lanes}
        if tl_data is not None:
            for v in tl_data:
                lane = tl_data[v][traci.constants.VAR_LANE_ID]
                if lane not in lane_vehicles:
                    lane_vehicles[lane] = {}
                lane_vehicles[lane][v] = tl_data[v]
        return lane_vehicles

    def get_reward(self):
        # calculate delay of vehicles on incoming lanes
        reward = 0
        if self.metric_mode == 'delay':
            for v in self.old_v:
                # calculate individual vehicle delay
                v_delay = (self.t - self.v_info[v]['t']) - self.lane_travel_times[self.v_info[v]['lane']]
                if v_delay > 0:
                    reward -= v_delay
            new_v = set()
            # record start time and lane of new_vehicles
            for lane in self.incoming_lanes:
                for v in self.data[lane]:
                    if v not in self.old_v:
                        self.v_info[v] = {}
                        self.v_info[v]['t'] = self.t
                        self.v_info[v]['lane'] = lane
                        self.v_info[v]['speed'] = self.data[lane][v][traci.constants.VAR_SPEED]
                new_v.update(set(self.data[lane].keys()))
            # remove vehicles that have left incoming lanes
            remove_vehicles = self.old_v - new_v
            for v in remove_vehicles:
                del self.v_info[v]
            self.old_v = new_v
        elif self.metric_mode == 'speed':
            rewards = []
            for v in self.old_v:
                rewards.append(self.v_info[v]['speed'])
            if rewards:
                reward = np.min(rewards)
            else:
                reward = 0
        elif self.metric_mode == 'queue':
            lane_queues = {}
            for lane in self.incoming_lanes:
                lane_queues[lane] = 0
                for v in self.data[lane]:
                    if self.data[lane][v][traci.constants.VAR_SPEED] < self.stop_speed:
                        lane_queues[lane] += 1
            self.lane_queues = lane_queues
            reward = -np.max([self.lane_queues[lane] for lane in self.lane_queues])
        elif self.metric_mode == 'pressure':
            if self.phase in self.green_phases:
                # compute pressure for all green movements
                inc_lanes = self.max_pressure_lanes[self.phase]['inc']
                out_lanes = self.max_pressure_lanes[self.phase]['out']
                # pressure is defined as the number of vehicles in a lane
                inc_pressure = sum([len(self.data[l]) if l in self.data else 0 for l in inc_lanes])
                out_pressure = sum([len(self.data[l]) if l in self.data else 0 for l in out_lanes])
                reward = inc_pressure - out_pressure
            else:
                reward = 0
        else:
            print('No such a metric mode')
            assert 0
        return reward

    def get_max_pressure_lanes(self):
        """for each green phase, get all incoming
        and outgoing lanes for that phase, store
        in dict for max pressure calculation
        """
        max_pressure_lanes = {}
        for g in self.green_phases:
            inc_lanes = set()
            out_lanes = set()
            for l in self.phase_lanes[g]:
                inc_lanes.add(l)
                for ol in self.net_data['lane'][l]['outgoing']:
                    out_lanes.add(ol)
            max_pressure_lanes[g] = {'inc': inc_lanes, 'out': out_lanes}
        return max_pressure_lanes

    def get_state(self):
        density = np.array([len(self.data[lane]) for lane in self.incoming_lanes]) / self.lane_capacity
        # density = np.average(np.array([len(self.data[lane]) for lane in self.incoming_lanes]) / self.lane_capacity)
        # calculation of the queue length
        lane_queues = []
        for lane in self.incoming_lanes:
            q = 0
            for v in self.data[lane]:
                if self.data[lane][v][traci.constants.VAR_SPEED] < 0.3:
                    q += 1
            lane_queues.append(q)
        queue = np.array(lane_queues) / self.lane_capacity
        # queue = np.average(np.array(lane_queues) / self.lane_capacity)
        max_pressure = []
        # compute pressure for all green movements
        for g in self.green_phases:
            inc_lanes = self.max_pressure_lanes[g]['inc']
            out_lanes = self.max_pressure_lanes[g]['out']
            # pressure is defined as the number of vehicles in a lane
            inc_pressure = sum([len(self.data[l]) if l in self.data else 0 for l in inc_lanes])
            out_pressure = sum([len(self.data[l]) if l in self.data else 0 for l in out_lanes])
            max_pressure.append(inc_pressure - out_pressure)
        if self.max_press < abs(max(max_pressure)):
            self.max_press = abs(max(max_pressure))
        elif self.max_press < abs(min(max_pressure)):
            self.max_press = abs(min(max_pressure))
        max_pressure = np.array(max_pressure) / self.max_press

        # SPEED STATE
        # SINGLE
        speed = []
        for lane in self.incoming_lanes:
            for v in self.data[lane]:
                speed.append(self.data[lane][v][traci.constants.VAR_SPEED])
        if speed:
            if self.max_speed < np.max(speed):
                self.max_speed = np.max(speed)
            s = np.array([np.average(speed) / self.max_speed])
        else:
            s = np.array([0])

        # MULTI
        # speed = []
        # for lane in self.incoming_lanes:
        #     temp_speed = []
        #     for v in self.data[lane]:
        #         temp_speed.append(self.data[lane][v][traci.constants.VAR_SPEED])
        #     if temp_speed:
        #         if self.max_speed_per_lane[lane] < np.max(temp_speed):
        #             self.max_speed_per_lane[lane] = np.max(temp_speed)
        #         speed.append(np.average(temp_speed) / self.max_speed_per_lane[lane])
        #     else:
        #         speed.append(np.array(0))

        # SPEED
        # return np.array(speed)
        # return np.concatenate([s, self.phase_to_one_hot[self.phase]])
        # return np.concatenate([[s, density], self.phase_to_one_hot[self.phase]])
        # return np.concatenate([[s, queue], self.phase_to_one_hot[self.phase]])
        # return np.concatenate([[s, queue, density], self.phase_to_one_hot[self.phase]])
        # return np.array([s, queue])
        # return np.concatenate([np.array(speed), self.phase_to_one_hot[self.phase]])
        # DENSITY & QUEUE
        # return queue
        # return density
        return max_pressure
        # return np.concatenate(max_pressure, density]
        # return np.concatenate([max_pressure, self.phase_to_one_hot[self.phase]])
        # return np.array([density, queue])
        # return np.concatenate([density, queue])
        # return np.concatenate([queue, self.phase_to_one_hot[self.phase]])
        # return np.concatenate([density, self.phase_to_one_hot[self.phase]])
        # return np.concatenate([density, queue, self.phase_to_one_hot[self.phase]])
        # return np.concatenate([[density, queue], self.phase_to_one_hot[self.phase]])

    def get_phase_lanes(self, actions):
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

    def get_feedback(self):
        if self.max_step <= self.t:
            done = True
        else:
            done = False
        return self.get_state(), self.get_reward(), done

    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
