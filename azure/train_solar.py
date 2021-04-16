# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 17:10:33 2021

@author: MEvans
"""

from utils import model, processing
import argparse
import os
import glob
import math
import tensorflow as tf
from datetime import datetime
from azureml.core import Run

# Set Global variables

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', dest = 'data_folder', type = str, required = True, help = 'Training dataset')
#parser.add_argument('--eval_data', type = str, required = True, help = 'Evaluation dataset')
parser.add_argument('-lr', '--learning_rate', type = float, default = 0.0001, help = 'Initial learning rate')
parser.add_argument('-w', '--weight', type = float, default = 1.0, help = 'Positive sample weight for iou, bce, etc.')
parser.add_argument('-e', '--epochs', type = int, default = 10, help = 'Number of epochs to train the model for')
parser.add_argument('-b', '--batch', type = int, default = 16, help = 'Training batch size')
parser.add_argument('--size', type = int, default = 3000, help = 'Size of training data')
parser.add_argument('--kernel-size', type = int, default = 256, dest = 'kernel_size', help = 'Size in pixels of incoming patches')
parser.add_argument('--response', type = str, required = True, help = 'Name of the response variable in tfrecords')
args = parser.parse_args()

LR = args.learning_rate
WEIGHT = args.weight
BANDS = ['B2', 'B3', 'B4', 'B8', 'B2_1', 'B3_1', 'B4_1', 'B8_1']
RESPONSE = args.response
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999)
LOSS = model.weighted_bce(WEIGHT)
METRICS = [tf.keras.metrics.categorical_accuracy, tf.keras.metrics.MeanIoU(num_classes=2, name = 'mean_iou')]

FEATURES = BANDS + [RESPONSE]

# Round buffer up to nearest 100 based on training data size

BUFFER = int(math.ceil(args.size/100)) * 100

# Specify the size and shape of patches expected by the model.
KERNEL_SIZE = args.kernel_size
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
COLUMNS = [
  tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in FEATURES
]
FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

# create training dataset
train_files = glob.glob(os.path.join(args.data_folder, 'training', 'UNET_256_[A-Z]*.gz'))
eval_files =  glob.glob(os.path.join(args.data_folder, 'eval', 'UNET_256_[A-Z]*.gz'))

print(len(train_files))

training = processing.get_training_dataset(train_files, ftDict = FEATURES_DICT, buff = BUFFER, batch = args.batch)
evaluation = processing.get_eval_dataset(eval_files, ftDict = FEATURES_DICT)

# get the run context
run = Run.get_context()

# build the model
m = model.get_model(depth = len(BANDS), optim = OPTIMIZER, loss = LOSS, mets = METRICS)

# compile the model with our loss fxn, metrics, and optimizer
m.compile(
        optimizer = OPTIMIZER,
        loss = LOSS,
        metrics = METRICS
        )

# create special folders './outputs' and './logs' which automatically get saved
os.makedirs('outputs', exist_ok = True)
os.makedirs('logs', exist_ok = True)
out_dir = './outputs'
log_dir = './logs'

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

# train the model
m.fit(
        x = training,
        epochs = args.epochs,
        #TODO: make command line argument for size
        steps_per_epoch = 63,
        validation_data = evaluation,
        callbacks = [checkpoint, tensorboard]
        )

m.save(os.path.join(out_dir, 'unet256.h5'))