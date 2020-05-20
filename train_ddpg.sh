#!/usr/bin/env bash
# DDPG
# python main.py -netfp data/double/2x2.net.xml -sumocfg data/double/2x2.sumocfg -nogui -lra 0.0001 -lrc 0.00001 -epoch 150000 -batch 128 -trainer_type DDPG -learn_mark 5000 -gmin 5 -metric_mode queue -pool_type single
python main.py -netfp data/quodra/4x4.net.xml -sumocfg data/quodra/4x4.sumocfg -nogui -lra 0.0001 -lrc 0.00001 -epoch 150000 -batch 64 -trainer_type DDPG -learn_mark 5000 -gmin 5 -metric_mode queue -pool_type single
# python main.py -netfp data/bolognaringway/bolognaringway.net.xml -sumocfg data/bolognaringway/bolognaringway.sumo.cfg -nogui -lra 0.0005 -lrc 0.0005 -epoch 150000 -batch 128 -trainer_type DDPG -learn_mark 10000 -gmin 5 -metric_mode queue -pool_type single
# python main.py -netfp data/double/2x2.net.xml -sumocfg data/double/2x2.sumocfg -nogui -lra 0.0001 -lrc 0.001 -epoch 150000 -batch 64 -trainer_type DDPG -learn_mark 500 -gmin 5 -metric_mode queue -pool_type single
# python main.py -netfp data/quodra/4x4.net.xml -sumocfg data/quodra/4x4.sumocfg -nogui -lra 0.0001 -lrc 0.001 -epoch 150000 -batch 64 -trainer_type DDPG -learn_mark 500 -gmin 5 -metric_mode queue -pool_type single
