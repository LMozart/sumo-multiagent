#!/usr/bin/env bash
# python test.py -netfp data/double/2x2.net.xml -sumocfg data/double/2x2.sumocfg -epoch 500 -batch 64 -trainer_type Pressure -gmin 5 -metric_mode queue -nogui -controller_type max_pressure
# python test.py -netfp data/quodra/4x4.net.xml -sumocfg data/quodra/4x4.sumocfg -epoch 500 -batch 64 -trainer_type Pressure -gmin 5 -metric_mode queue -nogui -controller_type max_pressure
python test.py -netfp data/bolognaringway/bolognaringway.net.xml -sumocfg data/bolognaringway/bolognaringway.sumo.cfg -epoch 500 -batch 64 -trainer_type Pressure -gmin 5 -metric_mode queue -nogui -controller_type max_pressure

