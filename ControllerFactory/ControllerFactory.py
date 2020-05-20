from ControllerSet.DurationController import DurationController
from ControllerSet.DiscreteDurationController import DiscreteDurationController
from ControllerSet.MaxPressure import MaxPressureController
from ControllerSet.PhaseController import PhaseController
from ControllerSet.PressurePhaseController import PressurePhaseController
from ControllerSet.SOTL import SOTL


class ControllerFactory:
    def __init__(self, controller_config, net_data, traffic_light_ids):
        self.metric_mode = controller_config.metric_mode
        self.controller_type = controller_config.controller_type
        self.controller_config = controller_config
        self.net_data = net_data
        self.traffic_light_ids = traffic_light_ids

    def get_controller(self):
        if self.controller_type == "duration control":
            return [DurationController(traffic_id, self.net_data, self.controller_config)
                    for traffic_id in self.traffic_light_ids]
        elif self.controller_type == "discrete_duration_control":
            return [DiscreteDurationController(traffic_id, self.net_data, self.controller_config)
                    for traffic_id in self.traffic_light_ids]
        elif self.controller_type == "max_pressure":
            return [MaxPressureController(traffic_id, self.net_data, self.controller_config)
                    for traffic_id in self.traffic_light_ids]
        elif self.controller_type == "SOTL":
            return [SOTL(traffic_id, self.net_data, self.controller_config)
                    for traffic_id in self.traffic_light_ids]
        elif self.controller_type == "phase_control":
            return [PhaseController(traffic_id, self.net_data, self.controller_config)
                    for traffic_id in self.traffic_light_ids]
        elif self.controller_type == "pressure_phase_control":
            return [PressurePhaseController(traffic_id, self.net_data, self.controller_config)
                    for traffic_id in self.traffic_light_ids]
        else:
            print('This Controller does not exist')
            raise NotImplementedError()
