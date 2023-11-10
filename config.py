EPOCHS = 150  # Number of the epochs
SAVE_EVERY = 10  # after how many epochs to save a checkpoint
LOG_EVERY = 2  # log training and validation metrics every `LOG_EVERY` epochs
BATCH_SIZE = 8
DEVICE = 'cuda'  
LR = 0.001
ROOT_PATH = 'C://imaadproj//Drivable-Road-Region-Detection-and-Steering-Angle-Estimation-Method//CARL-DATASET'

# the classes that we want to train
CLASSES_TO_TRAIN = ['Road', 'Background']
# DEBUG for visualizations
DEBUG = True
