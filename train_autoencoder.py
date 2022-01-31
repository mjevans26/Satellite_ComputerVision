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
from azureml.core import Run, Workspace, Model


# Set Global variables

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type = str, required = True, help = 'Training datasets')
parser.add_argument('--eval_data', type = str, required = True, help = 'Evaluation datasets')
parser.add_argument('--test_data', type = str, default = None, help = 'directory containing test image(s) and mixer')
parser.add_argument('--model_id', type = str, required = False, default = None, help = 'model id for continued training')
parser.add_argument('-lr', '--learning_rate', type = float, default = 0.001, help = 'Initial learning rate')
parser.add_argument('-e', '--epochs', type = int, default = 10, help = 'Number of epochs to train the model for')
parser.add_argument('-b', '--batch', type = int, default = 16, help = 'Training batch size')
parser.add_argument('--size', type = int, default = 3000, help = 'Size of training dataset')
parser.add_argument('--kernel_size', type = int, default = 256, dest = 'kernel_size', help = 'Size in pixels of incoming patches')
parser.add_argument('--bands', type = str, nargs = '+', required = False, default = ['B2', 'B3', 'B4', 'B8'])
args = parser.parse_args()

TRAIN_SIZE = args.size
BATCH = args.batch
EPOCHS = args.epochs
LR = args.learning_rate
BANDS = args.bands
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999)
LOSS = 'mean_squared_error'
METRICS = [tf.keras.metrics.MeanSquaredError(name = 'mse')]

# round the training data size up to nearest 100 to define buffer
BUFFER = math.ceil(args.size//4/100)*100

# Specify the size and shape of patches expected by the model.
KERNEL_SIZE = args.kernel_size
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]

FEATURES = [tf.io.FixedLenFeature(shape = KERNEL_SHAPE, dtype = tf.float32) for band in BANDS]
FEATURES_DICT = dict(zip(BANDS, FEATURES))

# create special folders './outputs' and './logs' which automatically get saved
os.makedirs('outputs', exist_ok = True)
os.makedirs('logs', exist_ok = True)
out_dir = './outputs'
log_dir = './logs'

# create training dataset

# train_files = glob.glob(os.path.join(args.data_folder, 'training', 'UNET_256_[A-Z]*.gz'))
# eval_files =  glob.glob(os.path.join(args.data_folder, 'eval', 'UNET_256_[A-Z]*.gz'))

train_files = []
for root, dirs, files in os.walk(args.train_data):
    for f in files:
        train_files.append(os.path.join(root, f))

eval_files = []
for root, dirs, files in os.walk(args.eval_data):
    for f in files:
        eval_files.append(os.path.join(root, f))
print(f'number of train files = {len(train_files)}')
print(f'first train file is {train_files[0]}')
def to_tuple(inputs):
  """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
  Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
  Args:
    inputs: A dictionary of tensors, keyed by feature name.
  Returns: 
    A dtuple of (inputs, outputs).
  """
  # double up our bands to match the structure of before/after data
  inputsList = [inputs.get(key) for key in BANDS]
  stacked = tf.stack(inputsList, axis=0)
  # Convert from CHW to HWC
  stacked = tf.transpose(stacked, [1, 2, 0])
  # Perform image augmentation
  stacked = processing.aug_img(stacked)
  normalized = processing.normalize(stacked, [2])
  # do color augmentation on input features
  before = processing.aug_color(normalized)
  after = processing.aug_color(normalized)
  # standardize each patch of bands
  bands = tf.concat([before, after], axis = -1)
  response = bands
  return bands, response 

def get_dataset(files, ftDict, axes = [2], splits = None, one_hot = None, moments = None, **kwargs):
  """Function to read, parse and format to tuple a set of input tfrecord files.
  Get all the files matching the pattern, parse and convert to tuple.
  Args:
    files (list): A list of filenames storing tfrecords
    FtDict (dic): Dictionary of input features in tfrecords
    features (list): List of input feature names
    respones (str): response name(s)
    axes (list): axes along which to calculate moments for rescaling
    one_hot (dict): key:value pairs for name of one-hot variable and desired one-hot depth
    splits (list): size(s) of groups of features to be kept together
    moments (list<tpl>): list of [mean, var] tuples for standardization
  Returns: 
    A tf.data.Dataset
  """

  def parse_tfrecord(example_proto):
      return tf.io.parse_single_example(example_proto, ftDict)
  
  def tupelize(ftDict):
      return to_tuple(ftDict)
  
  dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
  dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
  dataset = dataset.map(tupelize, num_parallel_calls=5)
  return dataset

def get_training_dataset(files, ftDict, buff, batch = 16, repeat = True, axes = [2], splits = None, one_hot = None, moments = None, **kwargs):
    """
    Get the preprocessed training dataset
    Args:
        files (list): list of tfrecord files to be used for training
        FtDict (dic): Dictionary of input features in tfrecords
        features (list): List of input feature names
        respones (str): response name(s)
        axes (list): axes along which to calculate moments for rescaling
        buffer (int): buffer size for shuffle
        batch (int): batch size for training
        repeat (bool): should the dataset be repeated
    Returns: 
      A tf.data.Dataset of training data.
    """
    dataset = get_dataset(files, ftDict, axes, splits, one_hot, moments, **kwargs)
    if repeat:
        dataset = dataset.shuffle(buff).batch(batch).repeat()
    else:
        dataset = dataset.shuffle(buff).batch(batch)
    return dataset

def get_eval_dataset(files, ftDict, axes = [2], splits = None, one_hot = None, moments = None, **kwargs):
	"""
    Get the preprocessed evaluation dataset
    Args:
        files (list): list of tfrecords to be used for evaluation
    Returns: 
      A tf.data.Dataset of evaluation data.
    """

	dataset = get_dataset(files, ftDict, axes, splits, one_hot, moments, **kwargs)
	dataset = dataset.batch(1)
	return dataset

training = get_training_dataset(
        files = train_files[:len(train_files)//2],
        ftDict = FEATURES_DICT,
        buff = BUFFER,
        batch = BATCH,
        repeat = True)

evaluation = get_eval_dataset(
        files = eval_files[:len(eval_files)//2],
        ftDict = FEATURES_DICT,
        features = BANDS)

## DEFINE CALLBACKS

# get the current time
now = datetime.now() 
date = now.strftime("%d%b%y")
date

# define a checkpoint callback to save best models during training
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(out_dir, 'best_weights'+date+'.hdf5'),
    monitor='val_mse',
    verbose=1,
    save_best_only=True,
    mode='min'
    )

# define a tensorboard callback to write training logs
tensorboard = tf.keras.callbacks.TensorBoard(log_dir = log_dir)

callbacks = [checkpoint, tensorboard]

# get the run context
run = Run.get_context()
exp = run.experiment
ws = exp.workspace

# # Create a MirroredStrategy.
# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# ## BUILD THE MODEL
# with strategy.scope():
#     METRICS = [tf.keras.metrics.MeanSquaredError(name = 'mse')]

#     OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999)
m = model_tools.get_autoencoder(depth = len(BANDS)*2, optim = OPTIMIZER, loss = LOSS, mets = METRICS)
# # if a model directory provided we will reload previously trained model and weights
# if args.model_id:
#     # we will package the 'models' directory within the 'azure' dirrectory submitted with experiment run
#     model_dir = Model.get_model_path(args.model_id, _workspace = ws)
# #    model_dir = os.path.join('./models', args.model_id, '1', 'outputs')
    
#     # load our previously trained model and weights
#     model_file = glob.glob(os.path.join(model_dir, '*.h5'))[0]
#     weights_file = glob.glob(os.path.join(model_dir, '*.hdf5'))[0]
#     m, checkpoint = model_tools.retrain_model(model_file, checkpoint, evaluation, 'classes_mean_iou', weights_file, weight = WEIGHT, lr = LR)
#     # TODO: make this dynamic
#     initial_epoch = 100
# # otherwise build a model from scratch with provided specs
# else:
#     m = model_tools.get_autoencoder(depth = len(BANDS)*2, optim = OPTIMIZER, loss = LOSS, mets = METRICS)
#     initial_epoch = 0

# train the model
steps_per_epoch = int(TRAIN_SIZE//BATCH//4)
print('steps per epoch', steps_per_epoch)
m.fit(
        x = training,
        epochs = EPOCHS,
        steps_per_epoch = steps_per_epoch,
        validation_data = evaluation,
        callbacks = callbacks
        )

m.save(os.path.join(out_dir, 'unet256_autoencoder_8band.h5'))