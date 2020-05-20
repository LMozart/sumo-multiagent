#!/usr/bin/env bash
# MADDPG
python test.py -netfp data/quodra/4x4.net.xml -sumocfg data/quodra/4x4.sumocfg -epoch 150000 -batch 64 -trainer_type LDQN -gmin 5 -metric_mode queue -controller_type phase_control -pool_type multi -nogui
# python test.py -netfp data/double/2x2.net.xml -sumocfg data/double/2x2.sumocfg -epoch 150000 -batch 64 -trainer_type LDQN -gmin 5 -metric_mode queue -controller_type phase_control -pool_type multi -nogui
# python test.py -netfp data/bolognaringway/bolognaringway.net.xml -sumocfg data/bolognaringway/bolognaringway.sumo.cfg -nogui -lra 0.0001 -lrc 0.001 -epoch 150000 -batch 64 -trainer_type MADDPG -learn_mark 5000 -gmin 5 -metric_mode queue -pool_type multi
