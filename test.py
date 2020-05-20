from Utils import config
from TestFacade.TestSet.MaxPressureTest import PRESSURE
from TestFacade.TestSet.DQNTest import MLPTest
from TestFacade.TestSet.LDQNTest import LDQNTest
from TestFacade.TestSet.SOTL import SOTL

args = config.parse_cl_args()
if args.trainer_type == "Pressure":
    Test = PRESSURE(args)
elif args.trainer_type == "SOTL":
    Test = SOTL(args)
elif args.trainer_type == "MLPLight":
    Test = MLPTest(args)
elif args.trainer_type == "LDQN":
    Test = LDQNTest(args)
else:
    raise NotImplementedError
Test.run()
