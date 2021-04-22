# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 12:41:40 2021

@author: MEvans
"""

from utils import model_tools, processing
from utils.prediction_tools import makePredDataset, callback_predictions, plot_to_image
from matplotlib import pyplot as plt
import argparse
import os
import glob
import json
import math
import tensorflow as tf
from datetime import datetime
from azureml.core import Run

# Set Global variables

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type = str, required = True, help = 'Training datasets')
parser.add_argument('--eval_data', type = str, required = True, help = 'Evaluation datasets')
parser.add_argument('--test_data', type = str, default = None, help = 'directory containing test image(s) and mixer')
parser.add_argument('-lr', '--learning_rate', type = float, default = 0.001, help = 'Initial learning rate')
parser.add_argument('-w', '--weight', type = float, default = 1.0, help = 'Positive sample weight for iou, bce, etc.')
parser.add_argument('--bias', type = float, default = None, help = 'bias value for keras output layer initializer')
parser.add_argument('-e', '--epochs', type = int, default = 10, help = 'Number of epochs to train the model for')
parser.add_argument('-b', '--batch', type = int, default = 16, help = 'Training batch size')
parser.add_argument('--size', type = int, default = 3000, help = 'Size of training dataset')
parser.add_argument('--kernel_size', type = int, default = 256, dest = 'kernel_size', help = 'Size in pixels of incoming patches')
parser.add_argument('--response', type = str, required = True, help = 'Name of the response variable in tfrecords')
parser.add_argument('--bands', type = str, nargs = '+', required = False, default = ['B2', 'B3', 'B4', 'B8_1', 'B2_1', 'B3_1', 'B4_1', 'B8_1'])
parser.add_argument('--splits', type = int, nargs = '+', required = False, default = [4,4] )
args = parser.parse_args()

SPLITS = [4,4]#args.splits
TRAIN_SIZE = args.size
BATCH = args.batch
EPOCHS = args.epochs
BIAS = args.bias
WEIGHT = args.weight
LR = args.learning_rate
BANDS = ['B2', 'B3', 'B4', 'B8', 'B2_1', 'B3_1', 'B4_1', 'B8_1']#args.bands
RESPONSE = args.response
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999)

METRICS = [tf.keras.metrics.categorical_accuracy, tf.keras.metrics.MeanIoU(num_classes=2, name = 'mean_iou')]

FEATURES = BANDS + [RESPONSE]

# round the training data size up to nearest 100 to define buffer
BUFFER = math.ceil(args.size/100)*100

# Specify the size and shape of patches expected by the model.
KERNEL_SIZE = args.kernel_size
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
COLUMNS = [
  tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in FEATURES
]
FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

# create special folders './outputs' and './logs' which automatically get saved
os.makedirs('outputs', exist_ok = True)
os.makedirs('logs', exist_ok = True)
out_dir = './outputs'
log_dir = './logs'

# create training dataset
#train_files = []
#for path in args.train_data:
#    train_files += glob.glob(os.path.join(args.train_data, 'UNET_256_*.gz'))
#    
#eval_files = []
#for path in args.eval_data:
#    eval_files += glob.glob(os.path.join(args.eval_data, 'UNET_256_*.gz'))
#    
# train_files = glob.glob(os.path.join(args.data_folder, 'training', 'UNET_256_[A-Z]*.gz'))
# eval_files =  glob.glob(os.path.join(args.data_folder, 'eval', 'UNET_256_[A-Z]*.gz'))

train_files = glob.glob(os.path.join(args.train_data, 'UNET_256_[A-Z]*.gz'))
eval_files =  glob.glob(os.path.join(args.eval_data, 'UNET_256_[A-Z]*.gz'))

print(len(train_files))

training = processing.get_training_dataset(
        files = train_files,
        ftDict = FEATURES_DICT,
        features = BANDS,
        response = RESPONSE,
        buff = BUFFER,
        batch = BATCH,
        splits = SPLITS,
        repeat = True)

evaluation = processing.get_eval_dataset(
        files = eval_files,
        ftDict = FEATURES_DICT,
        features = BANDS,
        response = RESPONSE,
        splits = SPLITS)

def get_weighted_bce(y_true, y_pred):
    return model_tools.weighted_bce(y_true, y_pred, WEIGHT)

# get the run context
run = Run.get_context()

# build the model
m = model_tools.get_model(depth = len(BANDS), optim = OPTIMIZER, loss = get_weighted_bce, mets = METRICS, bias = BIAS)

# compile the model with our loss fxn, metrics, and optimizer
#m.compile(
#        optimizer = OPTIMIZER,
#        loss = LOSS,
#        metrics = METRICS
#        )


# get the current time
now = datetime.now() 
date = now.strftime("%d%b%y")
date

# define a checkpoint callback to save best models during training
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(out_dir, 'best_weights_' + date + '.hdf5'),
    monitor='val_mean_iou',
    verbose=1,
    save_best_only=True,
    mode='max'
    )

# define a tensorboard callback to write training logs
tensorboard = tf.keras.callbacks.TensorBoard(log_dir = log_dir)

# if test images provided, define an image saving checkpoint

if args.test_data:
    
    test_files = glob.glob(os.path.join(args.test_data, '*.gz'))
    mixer_file = glob.glob(os.path.join(args.test_data, '*.json'))
    
    # run predictions on a test image and log so we can see what the model is doing at each epoch
    jsonFile = mixer_file[0]
    with open(jsonFile,) as file:
        mixer = json.load(file)
        
    pred_data = makePredDataset(test_files, BANDS)
    file_writer = tf.summary.create_file_writer(log_dir + '/preds')

    def log_pred_image(epoch, logs):
      out_image = callback_predictions(pred_data, m, mixer)
      prob = out_image[:, :, 0]
      figure = plt.figure(figsize=(10, 10))
      plt.imshow(prob)
      image = plot_to_image(figure)
    
      with file_writer.as_default():
        tf.summary.image("Predicted Image", image, step=epoch)
    
    pred_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end = log_pred_image)
    
    callbacks = [checkpoint, tensorboard, pred_callback]
else:
    callbacks = [checkpoint, tensorboard]

# train the model
m.fit(
        x = training,
        epochs = args.epochs,
        steps_per_epoch = int(TRAIN_SIZE//BATCH),
        validation_data = evaluation,
        callbacks = callbacks
        )

m.save(os.path.join(out_dir, 'unet256.h5'))

