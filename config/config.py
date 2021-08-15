# third-party imports
from yacs.config import CfgNode as ConfigurationNode


# YACS overwrite these settings using YAML, all YAML variables must be defined here first.
_C = ConfigurationNode()

# Dataset config
_C.DATASET = ConfigurationNode()
_C.DATASET.NAME = "C.elegans"
_C.DATASET.DATA_PATH = "../data/"
_C.DATASET.RANDOM_SEED = 0

# Paths
_C.LOG_PATH = "./logs/"
_C.RESULT_PATH = "./results/"

# Training config
_C.TRAIN = ConfigurationNode()
_C.TRAIN.EPOCH = 500
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.USE_SCHEDULAR = True
_C.TRAIN.USE_EARLY_STOPPING = False
# _C.TRAIN.TAKE_SNAPSHOT = False
_C.TRAIN.K_FOLD_TRAINING_TIMES = 10

# Model config
_C.MODEL = ConfigurationNode()
_C.MODEL.NAME = "Taeyeon_Net"
_C.MODEL.INPUT_CHANNEL = 4
_C.MODEL.INPUT_LENGTH = 41

# Optimizer config
_C.OPTIMIZER = ConfigurationNode()
_C.OPTIMIZER.NAME = "SGD"

# Optimizer SGD config
_C.OPTIMIZER.SGD = ConfigurationNode()
_C.OPTIMIZER.SGD.MOMENTUM = 0.9
_C.OPTIMIZER.SGD.BASE_LR = 0.01
_C.OPTIMIZER.SGD.WEIGHT_DECAY = 0.0
_C.OPTIMIZER.SGD.NESTEROV = False

# Optimizer Adam conifig
_C.OPTIMIZER.ADAM = ConfigurationNode()
_C.OPTIMIZER.ADAM.BASE_LR = 1e-3
_C.OPTIMIZER.ADAM.WEIGHT_DECAY = 0

# Optimizer AdaBelief
_C.OPTIMIZER.ADABELIEF = ConfigurationNode()
_C.OPTIMIZER.ADABELIEF.BASE_LR = 2e-3
_C.OPTIMIZER.ADABELIEF.EPS = 1e-16

# Schedular config
_C.SCHEDULAR = ConfigurationNode()
_C.SCHEDULAR.NAME = "Reduce_on_plateau"

# Schedular ReduceLR on Plateau
_C.SCHEDULAR.REDUCELR = ConfigurationNode()
_C.SCHEDULAR.REDUCELR.PATIENCE = 7
_C.SCHEDULAR.REDUCELR.FACTOR = 0.75
_C.SCHEDULAR.REDUCELR.MIN_LR = 1e-8
_C.SCHEDULAR.REDUCELR.COOL_DOWN = 1

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()