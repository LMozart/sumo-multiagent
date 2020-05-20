#!/usr/bin/env bash
# Parameter Sharing
python main.py -netfp data/double/2x2.net.xml -sumocfg data/double/2x2.sumocfg -nogui -lra 0.0005 -lrc 0.0005 -epoch 150000 -batch 128 -trainer_type A2C -learn_mark 5000 -gmin 5 -metric_mode queue -pool_type single -controller_type phase_control