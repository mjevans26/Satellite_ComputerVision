# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:51:32 2021

@author: MEvans
"""

from utils import model_tools, processing
from utils.prediction_tools import makePredDataset, write_tfrecord_predictions
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

parser.add_argument('--pred_data', type = str, default = True, help = 'directory containing test image(s) and mixer')
parser.add_argument('--model_id', type = str, required = True, default = None, help = 'model id for continued training')
parser.add_argument('--kernel_size', type = int, default = 256, dest = 'kernel_size', help = 'Size in pixels of incoming patches')
parser.add_argument('--bands', type = str, nargs = '+', required = False, default = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])

args = parser.parse_args()

# get the run context
run = Run.get_context()
exp = run.experiment
ws = exp.workspace

BANDS = args.bands
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)

METRICS = {
        'logits':[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
        'classes':[tf.keras.metrics.MeanIoU(num_classes=2, name = 'mean_iou')]
        }

WEIGHTS = args.weights

def get_weighted_bce(y_true, y_pred):
    return model_tools.weighted_bce(y_true, y_pred, 1)

# if a model directory provided we will reload previously trained model and weights
# we will package the 'models' directory within the 'azure' dirrectory submitted with experiment run
model_dir = Model.get_model_path(args.model_id, _workspace = ws)
#    model_dir = os.path.join('./models', args.model_id, '1', 'outputs')

# load our previously trained model and weights
model_file = glob.glob(os.path.join(model_dir, '*.h5'))[0]
weights_file = glob.glob(os.path.join(model_dir, '*.hdf5'))[0]
m = model_tools.get_binary_model(depth = len(BANDS), optim = OPTIMIZER, loss = get_weighted_bce, mets = METRICS, bias = None)
m.load_weights(weights_file)

# Specify the size and shape of patches expected by the model.
KERNEL_SIZE = args.kernel_size
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]


# create special folders './outputs' and './logs' which automatically get saved
os.makedirs('outputs', exist_ok = True)
os.makedirs('logs', exist_ok = True)
out_dir = './outputs'
log_dir = './logs'

testFiles = []

for root, dirs, files in os.walk(args.pred_data):
    for f in files:
        testFiles.append(os.path.join(root, f))


predFiles = [x for x in testFiles if '.gz' in x]
jsonFiles = [x for x in testFiles if '.json' in x]
jsonFile = jsonFiles[0]
predData = makePredDataset(predFiles, BANDS, one_hot = None)

write_tfrecord_predictions(
    imageDataset = predData,
    model = m, 
    pred_path = out_dir, 
    out_image_base = jsonFile[:-10], 
    kernel_shape = [256, 256],
    kernel_buffer = [128,128])

# get the current time
now = datetime.now() 
date = now.strftime("%d%b%y")
date


