#!/usr/bin/env bash
# Parameter Sharing
python main.py -netfp data/quodra/4x4.net.xml -sumocfg data/quodra/4x4.sumocfg -nogui -lrc 0.00001 -epoch 150000 -batch 128 -trainer_type MLPLight -learn_mark 5000 -gmin 5 -metric_mode queue -pool_type single -controller_type phase_control
# python main.py -netfp data/double/2x2.net.xml -sumocfg data/double/2x2.sumocfg -nogui -lrc 0.000001 -epoch 150000 -batch 128 -trainer_type MLPLight -learn_mark 5000 -gmin 5 -metric_mode queue -pool_type single -controller_type phase_control
# python main.py -netfp data/bolognaringway/bolognaringway.net.xml -sumocfg data/bolognaringway/bolognaringway.sumo.cfg -nogui -lrc 0.0005 -epoch 150000 -batch 128 -trainer_type MLPLight -learn_mark 10000 -gmin 5 -metric_mode queue -pool_type single -controller_type phase_control
# python main.py -netfp data/bolognaringway/bolognaringway.net.xml -sumocfg data/bolognaringway/bolognaringway.sumo.cfg -nogui -lrc 0.0001 -epoch 150000 -batch 128 -trainer_type MLPLight -learn_mark 10000 -gmin 5 -metric_mode queue -pool_type single -controller_type phase_control
# python main.py -netfp data/bolognaringway/bolognaringway.net.xml -sumocfg data/bolognaringway/bolognaringway.sumo.cfg -nogui -lrc 0.000001 -epoch 150000 -batch 128 -trainer_type MLPLight -learn_mark 10000 -gmin 5 -metric_mode queue -pool_type per -controller_type phase_control
# python main.py -netfp data/bolognaringway/bolognaringway.net.xml -sumocfg data/bolognaringway/bolognaringway.sumo.cfg -nogui -lrc 0.00001 -epoch 150000 -batch 128 -trainer_type MLPLight -learn_mark 10000 -gmin 5 -metric_mode queue -pool_type per -controller_type phase_control

# PressLight
# python main.py -netfp data/quodra/4x4.net.xml -sumocfg data/quodra/4x4.sumocfg -nogui -lrc 0.0001 -epoch 150000 -batch 128 -trainer_type MLPLight -learn_mark 1000 -gmin 5 -metric_mode pressure -pool_type single -controller_type pressure_phase_control
# python main.py -netfp data/double/2x2.net.xml -sumocfg data/double/2x2.sumocfg -nogui -lrc 0.00001 -epoch 150000 -batch 128 -trainer_type MLPLight -learn_mark 1000 -gmin 5 -metric_mode pressure -pool_type single -controller_type pressure_phase_control

# Multi
# python main.py -netfp data/quodra/4x4.net.xml -sumocfg data/quodra/4x4.sumocfg -nogui -lrc 0.00001 -epoch 150000 -batch 128 -trainer_type MLPLightMulti -learn_mark 1000 -gmin 5 -metric_mode queue -pool_type single -controller_type phase_control
# python main.py -netfp data/double/2x2.net.xml -sumocfg data/double/2x2.sumocfg -nogui -lrc 0.00001 -epoch 150000 -batch 128 -trainer_type MLPLightMulti -learn_mark 5000 -gmin 5 -metric_mode queue -pool_type single -controller_type phase_control
# python main.py -netfp data/bolognaringway/bolognaringway.net.xml -sumocfg data/bolognaringway/bolognaringway.sumo.cfg -nogui -lrc 0.00001 -epoch 150000 -batch 128 -trainer_type MLPLightMulti -learn_mark 1000 -gmin 5 -metric_mode queue -pool_type single -controller_type phase_control

# FP
# python main.py -netfp data/bolognaringway/bolognaringway.net.xml -sumocfg data/bolognaringway/bolognaringway.sumo.cfg -nogui -lrc 0.00001 -epoch 150000 -batch 128 -trainer_type FPMLPLight -learn_mark 5000 -gmin 5 -metric_mode queue -pool_type fp -controller_type phase_control