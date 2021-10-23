DATA_DIR = ''
IMAGE_DIR = ''
# Model Buidling Static Configurations
BATCH_SIZE = 64
IMG_DIM = (224, 224, 3)
IMG_SIZE = (224, 224)
NUM_EPOCHS = 100 #Num Epochs
CHECKPOINT_DIR = '' #Checkpoint
N_TRAIN = train.shape[0] # Num of training examples
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE # Steps for scheduler
LR = 0.01 # Learning rate
model_histories = {}
