import config
import argparse
import sys
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

from engine import Trainer
from dataset import train_dataset, train_data_loader
from dataset import valid_dataset, valid_data_loader
from model import model

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--resume-training', dest='resume_training',
                    required=True, help='whether to resume training or not',
                    choices=['yes', 'no'])
parser.add_argument('-p', '--model-path', dest='model_path',
                    help='path to trained model for resuming training') 
args = vars(parser.parse_args())

print(f"\nSAVING CHECKPOINT EVERY {SAVE_EVERY} EPOCHS\n")
print(f"LOGGING EVERY {LOG_EVERY} EPOCHS\n")

epochs = EPOCHS

model.to(DEVICE)

# initialie `Trainer` if resuming training
if args['resume_training'] == 'yes':
    if args['model_path'] == None:
        sys.exit('\nPLEASE PROVIDE A MODEL TO RESUME TRAINING FROM!')
    trainer = Trainer( 
    model, 
    train_data_loader, 
    train_dataset,
    valid_data_loader,
    valid_dataset,
    CLASSES_TO_TRAIN,
    epochs,
    DEVICE, 
    LR,
    args['resume_training'],
    model_path=args['model_path']
)

# initialie `Trainer` if training from beginning
else:
    trainer = Trainer( 
        model, 
        train_data_loader, 
        train_dataset,
        valid_data_loader,
        valid_dataset,
        CLASSES_TO_TRAIN,
        epochs,
        DEVICE, 
        LR,
        args['resume_training']
    )

trained_epochs = trainer.get_num_epochs()
epochs_to_train = epochs - trained_epochs

train_loss , train_mIoU, train_pix_acc = [], [], []
valid_loss , valid_mIoU, valid_pix_acc = [], [], []
for epoch in range(epochs_to_train):
    epoch = epoch+1+trained_epochs
    print(f"Epoch {epoch} of {epochs}")
    train_epoch_loss, train_epoch_mIoU, train_epoch_pixacc = trainer.fit()
    valid_epoch_loss, valid_epoch_mIoU, valid_epoch_pixacc = trainer.validate(epoch)
    train_loss.append(train_epoch_loss)
    train_mIoU.append(train_epoch_mIoU)
    train_pix_acc.append(train_epoch_pixacc)
    valid_loss.append(valid_epoch_loss)
    valid_mIoU.append(valid_epoch_mIoU)
    valid_pix_acc.append(valid_epoch_pixacc)

    if epoch % LOG_EVERY == 0: 
        print(f"Train Epoch Loss: {train_epoch_loss:.4f}, Train Epoch mIoU: {train_epoch_mIoU:.4f}, Train Epoch PixAcc: {train_epoch_pixacc:.4f}")
        print(f"Valid Epoch Loss: {valid_epoch_loss:.4f}, Valid Epoch mIoU: {valid_epoch_mIoU:.4f}, Valid Epoch PixAcc: {valid_epoch_pixacc:.4f}")


    # save model every 5 epochs
    if epoch % SAVE_EVERY == 0:
        print('SAVING MODEL')
        trainer.save_model(epoch)
        print('SAVING COMPLETE')

print('TRAINING COMPLETE')