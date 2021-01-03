# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 12:41:40 2021

@author: MEvans
"""

import processing
import model
import argparse
import os
import glob
import tensorflow as tf
from datetime import datetime

# Set Global variables

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', dest = 'data_folder', type = str, required = True, help = 'Training dataset')
#parser.add_argument('--eval_data', type = str, required = True, help = 'Evaluation dataset')
parser.add_argument('-lr', '--learning_rate', type = float, default = 0.0001, help = 'Initial learning rate')
parser.add_argument('-w', '--weight', type = float, default = 1.0, help = 'Positive sample weight for iou, bce, etc.')
parser.add_argument('-e', '--epochs', type = int, default = 10, help = 'Number of epochs to train the model for')
args = parser.parse_args()

LR = args.lr
WEIGHT = args.w
BANDS = ['B2', 'B3', 'B3', 'B8', 'B2_1', 'B3_1', 'B4_1', 'B8_1']
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999)
LOSS = model.weighted_bce(WEIGHT)
METRICS = [tf.keras.metrics.categorical_accuracy, tf.keras.metrics.MeanIoU(num_classes=2, name = 'mean_iou')]

# create training dataset
train_files = glob.glob(os.path.join(args.data_folder, 'train'))
eval_files =  glob.glob(os.path.join(args.data_folder, 'eval'))

training = processing.get_training_dataset(train_files)
evaluation = processing.get_eval_dataset(eval_files)

# build the model
m = model.get_model(depth = len(BANDS), optim = OPTIMIZER, loss = LOSS, mets = METRICS)

# compile the model with our loss fxn, metrics, and optimizer
m.compile(
        optimizer = OPTIMIZER,
        loss = LOSS,
        metrics = METRICS
        )

# create a checkpoint to save model weights
os.makedirs('outputs', exists_OK = True)
out_folder = '/outputs'

now = datetime.now() 
date = now.strftime("%d%b%y")
date

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(out_folder, 'best_weights_' + date + '.hdf5'),
    monitor='val_mean_iou',
    verbose=1,
    save_best_only=True,
    mode='max'
    )

# train the model
m.train(
        x = training,
        epochs = args.e,
        validation_data = evaluation,
        callbacks = [checkpoint, tensorboard])

joblib.dump(value = m, filename = 'outputs/unet256.h5)

