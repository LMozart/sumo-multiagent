import sys
import os
import numpy as np
from ControllerFactory.ControllerFactory import ControllerFactory

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


class SUMO:
    def __init__(self, env_args, controller_config, net_data):
        self.temp = []
        # sumo stimulator relevant
        cfg_fp = env_args.cfg_fp
        self.traffic_light_ids = []
        # self.sumo_cmd = r'/usr/bin/sumo' if env_args.nogui else r'/usr/bin/sumo-gui'
        self.sumo_cmd = r'/usr/local/opt/sumo/bin/sumo' if env_args.nogui else r'/usr/local/opt/sumo/bin/sumo-gui'
        self.initialize = True
        self.sumo_cmd = [self.sumo_cmd, "-c",
                         cfg_fp, "--no-warnings",
                         "--no-step-log", "--random"]
        traci.start(self.sumo_cmd)

        # Get Traffic ID
        junction_ids = traci.junction.getIDList()
        traffic_light_and_junction_ids = set(traci.trafficlight.getIDList()).intersection(junction_ids)
        for tl in traffic_light_and_junction_ids:
            # get the data in traffic light
            traci.trafficlight.subscribe(tl, [traci.constants.TL_COMPLETE_DEFINITION_RYG])
            tl_data = traci.trafficlight.getAllSubscriptionResults()
            logic = tl_data[tl][traci.constants.TL_COMPLETE_DEFINITION_RYG][0]
            green_phases = [p.state for p in logic.getPhases()
                            if 'y' not in p.state
                            and ('G' in p.state or 'g' in p.state)]
            if len(green_phases) > 1:
                self.traffic_light_ids.append(tl)
            net_data['inter'][tl]['green_phases'] = green_phases
            net_data['inter'][tl]['inter_phases'] = [p.state for p in logic.getPhases() if 'y' in p.state]
        controller_factory = ControllerFactory(controller_config, net_data, self.traffic_light_ids)
        self.controllers = controller_factory.get_controller()
        self.numb_ctrl = len(self.controllers)

    def step(self, actions):
        for j in range(self.numb_ctrl):
            temp_e = self.controllers[j].step([])

        traci.simulationStep()

        rewards = []
        done = False

        for j in range(self.numb_ctrl):
            _, r, d = self.controllers[j].get_feedback()
            rewards.append(r)
            done = done or d
        return rewards, done

    def reset(self):
        if not self.initialize:
            traci.close()
            print('traci disconnected')
            traci.start(self.sumo_cmd)
            print('traci start')
        self.initialize = False
        state = [self.controllers[i].reset() for i in range(self.numb_ctrl)]
        return state
