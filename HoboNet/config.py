import os


ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

BATCH_SIZE = 128
NUM_EPOCHS = 150

MAX_SEQ_LENGTH = 133
EDIT_SPACE = 5

SAVE_CHECKPOINT_STEP = 100

MODEL_PATH = os.path.join(ROOT_PATH, "model/hobonet")
CHECKPOINT_DIR = os.path.join(ROOT_PATH, 'checkpoints/')
LOG_DIR = os.path.join(ROOT_PATH, 'logs/')

TRAIN_FILE = os.path.join(ROOT_PATH, 'data/train.txt')
