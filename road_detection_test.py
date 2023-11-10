import torch
import numpy as np
import cv2
import argparse
import config
import albumentations
import matplotlib.pyplot as plt
from PIL import Image
from model import model
from utils.helpers import draw_test_segmentation_map, image_overlay
######################################################
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
#########################################################
# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model-path', dest='model_path', required=True,
                    help='path to the trained model weights')
parser.add_argument('-i', '--input', required=True, 
                    help='path to input image')
args = vars(parser.parse_args())

# define the image transforms
transform = albumentations.Compose([
    albumentations.Normalize(
            mean=[0.45734706, 0.43338275, 0.40058118],
            std=[0.23965294, 0.23532275, 0.2398498],
            always_apply=True)
])

# initialize the model
model = model
# load the model checkpoint
checkpoint = torch.load(args['model_path'])
# load the trained weights
model.load_state_dict(checkpoint['model_state_dict'])
# load the model on to the computation device
model.eval().to(config.DEVICE)

image = np.array(Image.open(args['input']).convert('RGB'))
# make a copy of the image
orig_image = image.copy()
# apply transforms
image = transform(image=image)['image']
# tranpose dimensions
image = np.transpose(image, (2, 0, 1))
# convert to torch tensors
image = torch.tensor(image, dtype=torch.float)
# add batch dimension
image = image.unsqueeze(0).to(DEVICE)

# forward pass through the model
outputs = model(image)
outputs = outputs['out']
# get the segmentation map
segmented_image = draw_test_segmentation_map(outputs)
# image overlay
result = image_overlay(orig_image, segmented_image)

# Convert the result image from OpenCV BGR format to PIL RGB format
result_pil = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
result_pil = Image.fromarray(result_pil)

# Display the result image using PIL
result_pil.show()
'''
# visualize result
cv2.imshow('Result', result)
cv2.waitKey(0)
'''
save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
cv2.imwrite(f"outputs/{save_name}.jpg", result)