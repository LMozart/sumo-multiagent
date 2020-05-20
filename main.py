from Utils import config
from TrainFacade.TrainerSet.DdpgTrainer import DdpgTrainer
from TrainFacade.TrainerSet.DQNTrainer import DQNTrainer
from TrainFacade.TrainerSet.LDQNTrainer import LDQNTrainer


args = config.parse_cl_args()
if args.trainer_type == "DDPG":
    trainer = DdpgTrainer(args)
elif args.trainer_type == "DQN":
    trainer = DQNTrainer(args)
elif args.trainer_type == "LDQN":
    trainer = LDQNTrainer(args)
else:
    raise NotImplementedError
trainer.run()
