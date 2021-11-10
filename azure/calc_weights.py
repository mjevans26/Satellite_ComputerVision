# -*- coding: utf-8 -*-
"""
Created on Tue Nov 9 12:13:11 2021

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
from azureml.core import Run, Workspace, Model


# Set Global variables

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type = str, required = True, help = 'Training datasets')
parser.add_argument('--model_id', type = str, required = False, default = None, help = 'model id for continued training')
parser.add_argument('--kernel_size', type = int, default = 256, dest = 'kernel_size', help = 'Size in pixels of incoming patches')
parser.add_argument('--response', type = str, required = True, default = 'landcover', help = 'Name of the response variable in tfrecords')
parser.add_argument('--nclasses', type = int, required = True, default = 10, help = 'Number of response classes')
parser.add_argument('--bands', type = str, nargs = '+', required = False, default = ['B2_spring', 'B2_summer', 'B2_fall', 'B3_spring', 'B3_summer', 'B3_fall', 'B4_spring', 'B4_summer', 'B4_fall', 'B5_spring', 'B5_summer', 'B5_fall', 'B6_spring', 'B6_summer', 'B6_fall', 'B7_spring', 'B7_summer', 'B7_fall', 'B8_spring', 'B8_summer', 'B8_fall', 'B8A_spring', 'B8A_summer', 'B8A_fall', 'B11_spring', 'B11_summer', 'B11_fall', 'B12_spring', 'B12_summer', 'B12_fall', 'R', 'G', 'B', 'N'])
parser.add_argument('--splits', type = int, nargs = '+', required = False, default = None )
parser.add_argument('--one_hot_levels', type = int, nargs = '+', required = False, default = [10])
parser.add_argument('--one_hot_names', type = str, nargs = '+', required = False, default = ['landcover'])
args = parser.parse_args()

ONE_HOT = dict(zip(args.one_hot_names, args.one_hot_levels))
SPLITS = args.splits
BANDS = args.bands
RESPONSE = args.response
NCLASSES = args.nclasses

FEATURES = BANDS + [RESPONSE]

# if the response is one-hot convert it to a dictionary
# this will trigger correct processing by processing.to_tuple
if RESPONSE in ONE_HOT.keys():
    # RESPONSE = {key:ONE_HOT[key] for key in [RESPONSE]}
    RESPONSE = {RESPONSE:ONE_HOT.pop(RESPONSE)}
    if len(ONE_HOT) < 1:
        ONE_HOT = None
    
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999)
DEPTH = 4
print(BANDS)

METRICS = METRICS = [tf.keras.metrics.categorical_accuracy,
                     tf.keras.metrics.MeanIoU(num_classes=list(RESPONSE.values())[0], name = 'mean_iou')]

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

# train_files = glob.glob(os.path.join(args.data_folder, 'training', 'UNET_256_[A-Z]*.gz'))
# eval_files =  glob.glob(os.path.join(args.data_folder, 'eval', 'UNET_256_[A-Z]*.gz'))
i = 1
train_files = []
for root, dirs, files in os.walk(args.train_data):
    for f in files:
        train_files.append(os.path.join(root, f))
        i+=1

# eval_files = []
# for root, dirs, files in os.walk(args.eval_data):
#     for f in files:
#         if i%2==0:
#             eval_files.append(os.path.join(root, f))
#         i+=1
        
# train_files = glob.glob(os.path.join(args.train_data, 'UNET_256_[A-Z]*.gz'))
# eval_files =  glob.glob(os.path.join(args.eval_data, 'UNET_256_[A-Z]*.gz'))

training = processing.get_training_dataset(
        files = train_files,
        ftDict = FEATURES_DICT,
        features = BANDS,
        response = RESPONSE,
        buff = BUFFER,
        batch = BATCH,
        repeat = False,
        splits = SPLITS,
        one_hot = ONE_HOT)

# evaluation = processing.get_eval_dataset(
#         files = eval_files,
#         ftDict = FEATURES_DICT,
#         features = BANDS,
#         response = RESPONSE,
#         splits = SPLITS,
#         one_hot = ONE_HOT)

## DEFINE CALLBACKS

m = get_multiclass_model(
    depth = DEPTH,
    nclasses = NCLASSES,
    optim = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999),
    loss = 'mse', 
    mets = [tf.keras.metrics.categorical_accuracy],
    bias = None)
    
train_con_mat = make_confusion_matrix(training, m, True)

classums = train_con_mat.sum(axis = 1)
BIAS = np.log(classums[1]/classums[0])
WEIGHT = classums[0]/classums[1]
TRAIN_SIZE = train_con_mat.sum()//(256*256)

print('size = ', TRAIN_SIZE)
print(f'bias = {BIAS}')
print(f'weight = {WEIGHT}')

np.save(join(out_dir, 'con_mat'), train_con_mat)
