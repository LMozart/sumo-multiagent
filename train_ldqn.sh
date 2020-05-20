#!/usr/bin/env bash
# EFFECTIVE
# python main.py -netfp data/quodra/4x4.net.xml -sumocfg data/quodra/4x4.sumocfg -nogui -lrc 0.0001 -epoch 150000 -batch 128 -trainer_type LDQN -learn_mark 5000 -gmin 5 -metric_mode queue -pool_type latency -controller_type phase_control
# python main.py -netfp data/double/2x2.net.xml -sumocfg data/double/2x2.sumocfg -nogui -lrc 0.000001 -epoch 150000 -batch 128 -trainer_type LDQN -learn_mark 5000 -gmin 5 -metric_mode queue -pool_type latency -controller_type phase_control

# A LITTLE BIT TRY
# python main.py -netfp data/bolognaringway/bolognaringway.net.xml -sumocfg data/bolognaringway/bolognaringway.sumo.cfg -nogui -lrc 0.00001 -epoch 150000 -batch 128 -trainer_type LDQN -learn_mark 10000 -gmin 5 -metric_mode queue -pool_type latency -controller_type phase_control -max_size 10000
# python main.py -netfp data/bolognaringway/bolognaringway.net.xml -sumocfg data/bolognaringway/bolognaringway.sumo.cfg -nogui -lrc 0.000001 -epoch 150000 -batch 128 -trainer_type LDQN -learn_mark 10000 -gmin 5 -metric_mode queue -pool_type latency -controller_type phase_control -max_size 10000

# PRESS BEST
# python main.py -netfp data/bolognaringway/bolognaringway.net.xml -sumocfg data/bolognaringway/bolognaringway.sumo.cfg -nogui -lrc 0.0005 -epoch 150000 -batch 128 -trainer_type LDQN -learn_mark 10000 -gmin 5 -metric_mode queue -pool_type latency -controller_type phase_control -max_size 10000
python main.py -netfp data/quodra/4x4.net.xml -sumocfg data/quodra/4x4.sumocfg -nogui -lrc 0.00001 -epoch 150000 -batch 128 -trainer_type LDQN -learn_mark 10000 -gmin 5 -metric_mode queue -pool_type latency -controller_type phase_control -max_size 10000
# python main.py -netfp data/double/2x2.net.xml -sumocfg data/double/2x2.sumocfg -nogui -lrc 0.00001 -epoch 150000 -batch 128 -trainer_type LDQN -learn_mark 10000 -gmin 5 -metric_mode queue -pool_type latency -controller_type phase_control

# FAILED TEST
# python main.py -netfp data/double/2x2.net.xml -sumocfg data/double/2x2.sumocfg -nogui -lrc 0.000001 -epoch 150000 -batch 128 -trainer_type LDQN -learn_mark 5000 -gmin 5 -metric_mode queue -pool_type latency -controller_type phase_control
# python main.py -netfp data/double/2x2.net.xml -sumocfg data/double/2x2.sumocfg -nogui -lrc 0.00001 -epoch 150000 -batch 128 -trainer_type LDQN -learn_mark 10000 -gmin 5 -metric_mode queue -pool_type latency -controller_type phase_control
# python main.py -netfp data/double/2x2.net.xml -sumocfg data/double/2x2.sumocfg -nogui -lrc 0.000001 -epoch 150000 -batch 128 -trainer_type LDQN -learn_mark 5000 -gmin 5 -metric_mode queue -pool_type latency -controller_type phase_control

# 4X4 ROU 2
# python main.py -netfp data/quodra/4x4.net.xml -sumocfg data/quodra/4x4.sumocfg -nogui -lrc 0.001 -epoch 150000 -batch 128 -trainer_type LDQN -learn_mark 5000 -gmin 5 -metric_mode queue -pool_type latency -controller_type phase_control -max_size 10000

# 在输入向量是phase + density + incoming lanes(加上每个道都算)的前提下才会把交叉口路网的延时降到一个较低点，以下是比较好的一次训练结果
# python main.py -netfp data/quodra/4x4.net.xml -sumocfg data/quodra/4x4.sumocfg -nogui -lrc 0.0001 -epoch 150000 -batch 128 -trainer_type LDQN -learn_mark 5000 -gmin 5 -metric_mode queue -pool_type latency -controller_type phase_control -max_size 10000

# MULTI
# python main.py -netfp data/bolognaringway/bolognaringway.net.xml -sumocfg data/bolognaringway/bolognaringway.sumo.cfg -nogui -lrc 0.0005 -epoch 150000 -batch 128 -trainer_type LDQNMulti -learn_mark 500 -gmin 5 -metric_mode queue -pool_type latency -controller_type phase_control

# NO SEPERATOR
# OKAY
# python main.py -netfp data/bolognaringway/bolognaringway.net.xml -sumocfg data/bolognaringway/bolognaringway.sumo.cfg -nogui -lrc 0.000001 -epoch 150000 -batch 128 -trainer_type LDQN -learn_mark 10000 -gmin 5 -metric_mode queue -pool_type latency -controller_type phase_control
# python main.py -netfp data/bolognaringway/bolognaringway.net.xml -sumocfg data/bolognaringway/bolognaringway.sumo.cfg -nogui -lrc 0.000001 -epoch 150000 -batch 256 -trainer_type LDQN -learn_mark 50000 -gmin 5 -metric_mode queue -pool_type latency -controller_type phase_control -max_size 50000