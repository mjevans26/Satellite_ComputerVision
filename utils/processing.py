# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:50:44 2020

@author: MEvans
"""
import tensorflow as tf
import numpy as np
import math
import os
import sys
import requests
import io
from random import shuffle, randint, uniform
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
DIR = Path(os.path.relpath(ROOT, Path.cwd()))

if str(DIR) not in sys.path:
    sys.path.append(str(DIR)) 

from array_tools import merge_classes, normalize_array, rescale_array, aug_array_color, aug_array_morph, rearrange_timeseries, normalize_timeseries, make_harmonics

def split_files(files, labels = ['label', 'lu', 'lidar', 's2', 'naip']):
  """Divide list of .npy arrays into separate lists by source data (e.g. NAIP, S2, etc.)

  Params
  ---
  files: list(str)
    list of files to be split
  labels: list(str)
    list of prefixes identifying subsets of files to return
  
  Return
  ---
  list, list, list: tuple of lists per file subset
  """

#   tuples = [tuple(f.split('_')[1::4]) for f in basenames]
  indices = [set([tuple(os.path.basename(f).split('_')[1::4]) for f in files if label in os.path.dirname(f)]) for label in labels]

  intersection = set.intersection(*indices)

  out_files = [[f for f in files if label in os.path.dirname(f) and tuple(os.path.basename(f).split('_')[1::4]) in intersection] for label in labels]

  return tuple(out_files)
  
def calc_ndvi(input):
  """Caclulate NDVI from Sentinel-2 data
  Parameters:
    input (dict): dictionary of incoming tensors
  Returns:
    tensor
  """
  epsilon = 1e-8
  nir = input.get('B8')
  red = input.get('B4')
  ndvi = tf.divide(tf.subtract(nir, red), tf.add(epsilon, tf.add(nir,red)))
  return ndvi

def aug_tensor_color(img):
    n_ch = tf.shape(img)[-1]
    contra_adj = 0.05
    bright_adj = 0.05

    ch_mean = tf.math.reduce_mean(img, axis = (0,1), keepdims = True)
    #ch_mean = np.mean(img, axis=(0, 1), keepdims=True).astype(np.float32)

    contra_mul = tf.random.uniform(shape = (1, 1, n_ch),
                                   minval = 1-contra_adj,
                                   maxval = 1+contra_adj)
    # contra_mul = np.random.uniform(1 - contra_adj, 1 + contra_adj, (1, 1, n_ch)).astype(
    #     np.float32
    # )

    bright_mul = tf.random.uniform(shape = (1, 1, n_ch),
                                   minval = 1 - bright_adj,
                                   maxval = 1+bright_adj)
    # bright_mul = np.random.uniform(1 - bright_adj, 1 + bright_adj, (1, 1, n_ch)).astype(
    #     np.float32
    # )

    recolored = (img - ch_mean) * contra_mul + ch_mean * bright_mul
    return recolored

def augColor(x, contra_adj = 0.05, bright_adj = 0.05):
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.05)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x
  
def aug_tensor_morph(img):
    """
    Perform image augmentation on tfRecords
    Parameters:
        img (TFRecord): 4D tensor
    Returns:
        3D tensor: 
    """
    outDims = tf.shape(img)[0:1]
    x = tf.image.random_flip_left_right(img)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
      #x = zoom(x, outDims)
      #since were gonna map_fn this on a 4d image, output must be 3d, so squeeze the artificial 'sample' dimension
    return tf.squeeze(x)

def normalize_timeseries(arr, maxval = 10000, axis = -1, e = 0.00001):
  # normalize band values across timesteps
  normalized = arr/maxval
#   mn = np.nanmean(arr, axis = axis, keepdims = True)
#   std = np.nanstd(arr, axis = axis, keepdims = True)
#   normalized = (arr - mn)/(std+e)
  # replace nans with zeros?
  finite = np.where(np.isnan(normalized), 0.0, normalized)
  return finite

def rearrange_timeseries(arr, nbands):
  # the number of time steps is in the 1st dimension if our data is (B, T, H, W, C)
  timesteps = arr.shape[1]
  # randomly pick one of the timesteps as the starting time
  starttime = randint(0, timesteps-1)
  # print('start', starttime)
  # grab all timesteps leading up to the timestep corresponding to our random first
  last = arr[:,0:starttime,:,:,:]
  print('last shape', last.shape)
  first = arr[:,starttime:timesteps,:,:,:]
  print('start shape', first.shape)
  rearranged = np.concatenate([first, last], axis = 1)
  rearranged.shape == arr.shape

  feats = rearranged[:,0:-1,:,:,:]
  labels = rearranged[:,-1,:,:,0:nbands]

  # confirm there are no all-nan images in labels
  batch_sums = np.sum(labels, axis = (1,2,3))
  if 0.0 in batch_sums:
    print('all nan labels, reshuffling')
    feats, labels, starttime = rearrange_timeseries(arr, nbands)

  return(feats, labels, starttime)

def sin_cos(t:int, freq:int = 6) -> tuple:
    x = t/freq
    theta = 2*math.pi * x
    return (math.sin(theta), math.cos(theta))
    
def normalize_tensor(x, axes=[2], epsilon=1e-8, moments = None, splits = None):
    """
    Standardize incoming image patches by mean and variance.

    Moments can be calculated based on patch data by providing axes:      
    To standardize each pixel use axes = [2]
    To standardize each channel use axes = [0, 1]
    To standardize globally use axes = [0, 1, 2]

    To standardize by global, or per-channel moments supply a list of [mean, variance] tuples.
    To standardize groups of channels separately, identify the size of each group. Groups of
    channels must be stacked contiguously and group sizes must sum to the total # of channels
    
    Parameters:
        x (tensor): nD image tensor
        axes (array): Array of ints. Axes along which to compute mean and variance, usually length n-1
        epsilon (float): small number to avoid dividing by zero
        moments (list<tpl>): list of global mean, variance tuples for standardization
        splits (list): size(s) of groups of features to be kept together
    Return:
        tensor: nD image tensor normalized by channels
    """
    
    # define a basic function to normalize a 3d tensor
    def normalize(x):
#        shape = tf.shape(x).numpy()
        # if we've defined global or per-channel moments...
        if moments:
            # cast moments to arrays for mean and variance
            mean = np.array([tpl[0] for tpl in moments], dtype = 'float32')
            variance = np.array([tpl[1] for tpl in moments], dtype = 'float32')
        # otherwise, calculate moments along provided axes
        else:
            mean, variance = tf.nn.moments(x, axes, keepdims = True)
            # keepdims = True to ensure compatibility with input tensor

        # normalize the input tensor
        normed = (x - mean)/tf.sqrt(variance + epsilon)
        return normed
    

    # if splits are given, apply tensor normalization to each split
    if splits:
        splitLen = sum(splits)
        toNorm = x[:,:,0:splitLen]
        dontNorm = x[:,:,splitLen:]
        tensors = tf.split(toNorm, splits, axis = 2)
        normed = [normalize(tensor) for tensor in tensors]
        normed.append(dontNorm)
        # gather normalized splits into single tensor
        x_normed = tf.concat(normed, axis = 2)
    else:
        x_normed = normalize(x)

    return x_normed 

def rescale_tensor(img, axes = [2], epsilon=1e-8, moments = None, splits = None):
    """
    Rescale incoming image patch to [0,1] based on min and max values
    
    Min, max can be calculated based on patch data by providing axes:      
    To rescale each pixel use axes = [2]
    To rescale each channel use axes = [0, 1]
    To rescale globally use axes = [0, 1, 2]

    To rescale by global, or per-channel moments supply a list of [mean, variance] tuples.
    To rescale groups of channels separately, identify the size of each group. Groups of
    channels must be stacked contiguously and group sizes must sum to the total # of channels
    
    Args:
        img (tensor): 3D (H,W,C) image tensor
        axes (list): axes along which to calculate min/max for rescaling
        moments (list<tpl>): list of [min, max] tuples for standardization
        splits (list): size(s) of groups of features to be kept together
    Return:
        tensor: 3D tensor of same shape as input, with values [0,1]
    """
    def rescale(img):
        if moments:
            minimum = np.array([tpl[0] for tpl in moments], dtype = 'float32')
            maximum = np.array([tpl[1] for tpl in moments], dtype = 'float32')
        else:
            minimum = tf.math.reduce_min(img, axis = axes, keepdims = True)
            maximum = tf.math.reduce_max(img, axis = axes, keepdims = True)
        scaled = (img - minimum)/((maximum - minimum) + epsilon)
#        scaled = tf.divide(tf.subtract(img, minimum), tf.add(tf.subtract(maximum, minimum))
        return scaled
    
    # if splits are given, apply tensor normalization to each split
    if splits:
        tensors = tf.split(img, splits, axis = 2)
        rescaled = [rescale(tensor) for tensor in tensors]
        # gather normalized splits into single tensor
        img_rescaled = tf.concat(rescaled, axis = 2)
    else:
        img_rescaled = rescale(img)
        
    return img_rescaled

#def parse_tfrecord(example_proto, ftDict):
#    """The parsing function.
#    Read a serialized example into the structure defined by FEATURES_DICT.
#    Args:
#      example_proto: a serialized Example.
#    Returns: 
#      A dictionary of tensors, keyed by feature name.
#    """
#    return tf.io.parse_single_example(example_proto, ftDict)

          
def to_tuple(inputs, features, response, axes = [2], splits = None, one_hot = None, moments = None, **kwargs):
    """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
    Args:
      inputs (dict): A dictionary of tensors, keyed by feature name. Response
      variable must be the last item.
      features (list): List of input feature names
      respones (str): response name(s)
      axes (list): axes along which to calculate moments for rescaling
      one_hot (dict): key:value pairs for name of one-hot variable and desired one-hot depth
      splits (list): size(s) of groups of features to be kept together
      moments (list<tpl>): list of [mean, var] tuples for standardization
    Returns: 
      A dtuple of (inputs, outputs).
    """
#    one_hot = kwargs.get('one_hot')
#    splits = kwargs.get('splits')
#    moments = kwargs.get('moments')

    # If custom preprocessing functions are specified add respective bands
    for fxn in kwargs.values():
        der = fxn(inputs)
        inputs = der

#    inputsList = [inputs.get(key) for key in features + [response]]
    if type(response) == dict:
        depth = list(response.values())[0]
        key = list(response.keys())[0]
        res = tf.squeeze(tf.one_hot(tf.cast(inputs.get(key), tf.uint8), depth = depth))
    else:
        res = tf.expand_dims(inputs.get(response), axis = 2)
    
    # stack the augmented bands, optional one-hot tensors, and response variable
    if one_hot:
        featList = [inputs.get(key) for key in features if key not in one_hot.keys()]
        hotList= [tf.one_hot(tf.cast(inputs.get(key), tf.uint8), val, axis = 2) for key, val in one_hot.items() if key in features]
        # hotList = [tf.one_hot(tf.cast(inputs.get(key), tf.uint8), val, axis = 2) for key, val in one_hot.items()]
    else:
        featList = [inputs.get(key) for key in features]

    # stack, transpose, augment, and normalize continuous bands
    bands = tf.transpose(tf.stack(featList, axis = 0), [1,2,0])
    bands = aug_tensor_color(bands)
    bands = rescale_tensor(bands, axes = axes, moments = moments, splits = splits)
    
    if one_hot:
      hotStack = tf.concat(hotList, axis = 2)
      stacked = tf.concat([bands, hotStack, res], axis =2)
    else:
      stacked = tf.concat([bands, res], axis = 2)
    
    # perform morphological augmentation
    stacked = aug_tensor_morph(stacked)
    
    feats = stacked[:, :, :-res.shape[2]]
    labels = stacked[:, :, -res.shape[2]:]
    labels = tf.where(tf.greater(labels, 1.0), 1.0, labels)
    return feats, labels

def get_dataset(files, ftDict, features, response, axes = [2], splits = None, one_hot = None, moments = None, **kwargs):
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
      return to_tuple(ftDict, features, response, axes, splits, one_hot, moments, **kwargs)
  
  dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
  dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
  dataset = dataset.map(tupelize, num_parallel_calls=5)
  return dataset

def get_training_dataset(files, ftDict, features, response, buff, batch = 16, repeat = True, axes = [2], splits = None, one_hot = None, moments = None, **kwargs):
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
    dataset = get_dataset(files, ftDict, features, response, axes, splits, one_hot, moments, **kwargs)
    if repeat:
        dataset = dataset.shuffle(buff).batch(batch).repeat()
    else:
        dataset = dataset.shuffle(buff).batch(batch)
    return dataset
    
def get_eval_dataset(files, ftDict, features, response, axes = [2], splits = None, one_hot = None, moments = None, **kwargs):
	"""
    Get the preprocessed evaluation dataset
    Args:
        files (list): list of tfrecords to be used for evaluation
    Returns: 
      A tf.data.Dataset of evaluation data.
    """

	dataset = get_dataset(files, ftDict, features, response, axes, splits, one_hot, moments, **kwargs)
	dataset = dataset.batch(1)
	return dataset

class UNETDataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, labelfiles = None, s2files = None, naipfiles = None, lidarfiles = None, lufiles = None,
                 to_fit=True, batch_size=32, dim=(256, 256),
                 n_channels=4, n_classes = 8, shuffle=True,
                 splits = None, moments = None, translations = None):
        """Initialization

        :param files: list of all files to use in the generator
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param n_timesteps: number of multi-channel images
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.s2files = s2files
        self.naipfiles = naipfiles
        self.lidarfiles = lidarfiles
        self.labelfiles = labelfiles
        self.lufiles = lufiles
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.splits = splits
        self.moments = moments
        self.trans = translations
        self.indexes = np.arange(len(self.labelfiles))
        self.on_epoch_end()

        # do an initial shuffle for cases where the generator is called fresh at the start of each epoch
        if self.shuffle == True:
            print('shuffling')
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """
        return int(np.floor(len(self.indexes) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        print('the generator knows the epoch ended')
        self.indexes = np.arange(len(self.indexes))
        if self.shuffle == True:
            print('shuffling')
            np.random.shuffle(self.indexes)
      
    def load_numpy_url(self, url):
        response = requests.get(url)
        response.raise_for_status()
        data = np.load(io.BytesIO(response.content))
        return(data)

    def load_numpy_data(self, files_temp):
        arrays = [self.load_numpy_url(f) if f.startswith('http') else np.load(f) for f in files_temp]
        return(arrays)

    def _get_x_data(self, files_temp):
        # arrays come from PC in (C, H, W) format
        arrays = self.load_numpy_data(files_temp)
        array_shapes = [x.shape for x in arrays]
        try:
            assert len(arrays) == self.batch_size
            # make sure everything is 3D
            assert all([len(x) == 3 for x in array_shapes])
            # ensure all arrays are channels first
            arrays = [np.moveaxis(x, source = -1, destination = 0) if x.shape[-1] < x.shape[0] else x for x in arrays]
            # creat a single (B, C, H, W) array per batch
            batch = np.stack(arrays, axis = 0)
            in_shape = batch.shape
            # in case our incoming data is of different size than we want, define a trim amount
            trim = ((in_shape[2] - self.dim[0])//2, (in_shape[3] - self.dim[1])//2) 
            # If necessary, trim data to (-1, dims[0], dims[1])
            array = batch[:,:,trim[0]:self.dim[0]+trim[0], trim[1]:self.dim[1]+trim[1]]
            # rearrange arrays from (B, C, H, W) -> (B, H, W, C) expected by model
            reshaped = np.moveaxis(array, source = 1, destination = 3)
            return reshaped
        except AssertionError:
            return None
    
    def _get_naip_data(self, indexes):
        files_temp = [self.naipfiles[k] for k in indexes]
        naip = self._get_x_data(files_temp)
        if type(naip) == np.ndarray:
            rescaled = naip/255.0
            recolored = aug_array_color(rescaled)
            return recolored
        else:
            return naip
    
    def _get_s2_data(self, indexes):
        files_temp = [self.s2files[k] for k in indexes]
        s2 = self._get_x_data(files_temp)
        if type(s2) == np.ndarray:
            rescaled = s2/10000.0
            recolored = aug_array_color(rescaled)
            return recolored
        else:
            return s2
    
    def _get_lidar_data(self, indexes):
        files_temp = [self.lidarfiles[k] for k in indexes]
        lidar = self._get_x_data(files_temp)
        if type(lidar) == np.ndarray:
            rescaled = lidar/100
            return rescaled  
        else:
            return lidar      

    def _process_y(self, indexes):
        # get label files for current batch
        lc_files = [self.labelfiles[k] for k in indexes]
        lc_arrays = self.load_numpy_data(lc_files)
        lc = np.stack(lc_arrays, axis = 0) #(B, C, H, W)
        int_labels = lc.astype(int)

        # reduce the number of classes 
        merged_labels = merge_classes(cond_array = int_labels, trans = self.trans, out_array = int_labels)
        
        if self.lufiles:
            lu_files = [self.lufiles[k] for k in indexes]
            lu_arrays = [np.load(file) for file in lu_files]
            lu = np.stack(lu_arrays, axis = 0) #(B, C, H, W)
            merged_labels = merge_classes(cond_array = lu, trans = [(82,9), (84,10)], out_array = merged_labels)

        # If necessary, trim data to (-1, dims[0], dims[1])
        in_shape = merged_labels.shape
        trim = ((in_shape[2] - self.dim[0])//2, (in_shape[3] - self.dim[1])//2) 
        array = merged_labels[:,:,trim[0]:self.dim[0]+trim[0], trim[1]:self.dim[1]+trim[1]]

        # shift range of categorical labels from [1, n_classes] to [0, n_classes]
        zeroed = array - 1
        # create one-hot representation of classes
        one_hot = tf.one_hot(zeroed, self.n_classes)
        # one_hot = to_one_hot(zeroed, self.n_classes)
        return tf.squeeze(one_hot)
        
    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        if self.s2files:
            s2Data = self._get_s2_data(indexes)

        if self.naipfiles:
            naipData = self._get_naip_data(indexes)

        if self.lidarfiles:
            lidarData = self._get_lidar_data(indexes)
            xData = np.concatenate([naipData, lidarData], axis = -1)
        else:
            xData = naipData

        labels = self._process_y(indexes)
        
        # perform morphological augmentation - expects a 3D (H, W, C) image array
        stacked = np.concatenate([xData, labels], axis = -1)
        morphed = aug_array_morph(stacked)
        # print('augmented max', np.nanmax(augmented, axis = (0,1,2)))

        feats = morphed[:,:,:,0:self.n_channels]
        labels = morphed[:,:,:,self.n_channels:]

        if self.to_fit:
            return feats, labels
        else:
            return xData

class SiameseDataGenerator(UNETDataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # do an initial shuffle for cases where the generator is called fresh at the start of each epoch
        if self.shuffle == True:
            print('shuffling')
            np.random.shuffle(self.indexes)
        print(self.batch_size)
    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """
        return UNETDataGenerator.__len__(self)

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        UNETDataGenerator.on_epoch_end(self)

    def _process_y(self, indexes):
        # get label files for current batch
        files_temp = [self.labelfiles[k] for k in indexes]
        lc_files = UNETDataGenerator.load_numpy_data(self, files_temp)
        lc_arrays = [np.squeeze(np.load(file)) for file in lc_files] # make all labels 2D to start
        try:
            assert len(lc_arrays) == self.batch_size
            lc = np.stack(lc_arrays, axis = 0) #(B, H, W)
            int_labels = lc.astype(int)
            binary = np.where(int_labels > 1, 1, int_labels)
            # If necessary, trim data to (-1, dims[0], dims[1])
            in_shape = binary.shape # -> (B, H, W)
            trim = ((in_shape[1] - self.dim[0])//2, (in_shape[2] - self.dim[1])//2) 
            array = binary[:,trim[0]:self.dim[0]+trim[0], trim[1]:self.dim[1]+trim[1]]

            # add channel dimension (B, H, W) -> (B, H, W, C) expected by model
            reshaped = np.expand_dims(array, -1)
            return reshaped
        except AssertionError:
            return None
        
    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        s2Data = UNETDataGenerator._get_s2_data(self, indexes)

        labels = self._process_y(indexes)
        
        # perform morphological augmentation - expects a 3D (H, W, C) image array
        stacked = np.concatenate([s2Data, labels], axis = -1)
        morphed = aug_array_morph(stacked)
        # print('augmented max', np.nanmax(augmented, axis = (0,1,2)))

        feats_b = morphed[:,:,:,0:self.n_channels]
        feats_a = morphed[:,:,:,self.n_channels:-1]
        labels = morphed[:,:,:,-1:]

        if self.to_fit:
            return [feats_b, feats_a], labels
        else:
            return [feats_b, feats_a]
    

class LSTMDataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, files = None,
                 to_fit=True, batch_size=32, dim=(256, 256),
                 n_channels=4, n_timesteps = 6, shuffle=True):
        """Initialization

        :param files: list of all files to use in the generator
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param n_timesteps: number of multi-channel images
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.files = files
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """
        return int(np.floor(len(self.files) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Find list of IDs
        files_temp = [self.files[k] for k in indexes]
        # arrays come from PC in (T, C, H, W) format
        arrays = [np.load(file) for file in files_temp]
        trim = ((arrays[0].shape[2] - self.dim[0])//2, (arrays[0].shape[3] - self.dim[1])//2) 
        # TEMPORARY FIX: drop the last image to give us a sereis of 5
        array = [arr[0:self.n_timesteps,:,trim[0]:-trim[0],trim[1]:-trim[1]] for arr in arrays]

        # creat a single (B, T, C, H, W) array
        batch = np.stack(array, axis = 0)
        print('batch shape', batch.shape)
        # rearrange arrays from (B, T, C, H, W) -> (B, T, H, W, C) expected by model
        reshaped = np.moveaxis(batch, source = 2, destination = 4)
        print('reshaped shape', reshaped.shape)
        normalized = normalize_timeseries(reshaped, axis = 1)
        # harmonized = add_harmonic(normalized)
        if self.to_fit:
            rearranged = rearrange_timesereis(normalized, self.n_channels)
            feats, labels = split_timeseries(rearranged)
            # we can't have nans in label
            return feats, labels
        else:
            print('normalized dims', normalized.shape)
            return normalized

class LSTMAutoencoderGenerator(LSTMDataGenerator):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(
        self, harmonics = True, *args, **kwargs):
        """Initialization

        :param files: list of all files to use in the generator
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param n_timesteps: number of multi-channel images
        :param shuffle: True to shuffle label indexes after every epoch
        """
        super().__init__(*args, **kwargs)
        self.add_harmonics = harmonics
        self.on_epoch_end()

    def __len__(self):
        return LSTMDataGenerator.__len__(self)

    def on_epoch_end(self):
        LSTMDataGenerator.on_epoch_end(self)

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Find list of IDs
        files_temp = [self.files[k] for k in indexes]

        # arrays come from PC in (T, C, H, W) format
        arrays = [np.load(f) for f in files_temp]

        # creat a single (B, T, C, H, W) array
        batch = np.stack(arrays, axis = 0)

        # in case our incoming data is of different size than we want, define a trim amount
        trim = ((batch.shape[3] - self.dim[0])//2, (batch.shape[4] - self.dim[1])//2)

        # n_timesteps + 1 to account for the fact that the sequence includes the next image as target
        array = batch[:, 0:self.n_timesteps+1,:,trim[0]:self.dim[0]+trim[0],trim[1]:self.dim[1]+trim[1]]

        # rearrange arrays from (B, T, C, H, W) -> (B, T, H, W, C) expected by model
        reshaped = np.moveaxis(array, source = 2, destination = 4)

        normalized = normalize_timeseries(reshaped, axis = 1)
        
        # harmonized = add_harmonic(normalized)
        if self.add_harmonics:
            # get start dates for each file
            starts = [int(Path(f).stem.split('_')[2]) for f in files_temp]
        else:
            harmonics = None

        if self.to_fit:
            feats, y, start = rearrange_timeseries(normalized, self.n_channels)
            temporal_y = np.flip(feats, axis = 1) # reverse images along time dimension
            if self.add_harmonics:
                starts = [x + start - self.n_timesteps for x in starts]
                harmonics = make_harmonics(starts, self.n_timesteps, self.dim)
            return [feats, harmonics], [temporal_y, y]
        else:
            if self.add_harmonics:
                harmonics = make_harmonics(starts, self.n_timesteps, self.dim)
            return [normalized, harmonics]

class HybridDataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras model with U-Net and LSTM branches
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, s2files, naipfiles, labelfiles, lufiles, lidarfiles = None, n_classes = 8,
                 to_fit=True, batch_size=32, unet_dim=(320, 320, 4),
                 transitions = [(12,3), (11,3), (10,3), (9,8), (255, 0)],
                 lstm_dim = (6, 32, 32, 6), shuffle=True):
        """Class Initialization

        Params
        ---
        s2files: list
            numpy files containing sentinel-2 data to use in the generator
        naipfiles: list
            numpy files containing naip data to use in the generator
        lidarfiles: list
            numpy files containing lidar data to use in the generator
        labelfiles: list
            numpy files containing label data to use in the generator
        to_fit: bool
            True to return X and y, False to return X only
        batch_size: int
            batch size at each iteration
        dim: tuple 
            desired image H, W dimensions
        n_classes: int
            number of output masks
        n_timesteps: int
            number of multi-channel images in temporal s2 data
        shuffle: bool
            True to shuffle label indexes after every epoch
        
        Return
        ---
        tuple: three arrays containing batch of corresponding sentinel-2, naip, and label data
        """
        self.s2files = s2files
        self.naipfiles = naipfiles
        self.lidarfiles = lidarfiles
        self.labelfiles = labelfiles
        self.lufiles = lufiles
        self.trans = transitions
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.unet_dim = unet_dim
        self.lstm_dim = lstm_dim
        self.n_classes = n_classes
        self.n_timesteps = lstm_dim[0]
        self.shuffle = shuffle
        self.on_epoch_end()

        assert len(s2files) == len(naipfiles) == len(labelfiles), 'different number of files'

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """
        return int(np.floor(len(self.labelfiles) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        self.indexes = np.arange(len(self.labelfiles))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __iter__(self):
       for item in (self[i] for i in range(len(self))):
            yield item 
    
    def _get_s2_data(self, indexes):

        files_temp = [self.s2files[k] for k in indexes]
        # arrays come from PC in (T, C, H, W) format
        arrays = [np.load(f) for f in files_temp]

        try:
            assert len(arrays) > 0
            assert all([x.shape == (self.lstm_dim[0], self.lstm_dim[3], self.lstm_dim[1], self.lstm_dim[2]) for x in arrays])
        
            # creat a single (B, T, C, H, W) array
            batch = np.stack(arrays, axis = 0)
            # in case our incoming data is of different size than we want, define a trim amount
            trim = ((batch.shape[3] - self.lstm_dim[1])//2, (batch.shape[4] - self.lstm_dim[2])//2) 

            array = batch[:, 0:self.n_timesteps,:,trim[0]:self.lstm_dim[1]+trim[0],trim[1]:self.lstm_dim[2]+trim[1]]

            # rearrange arrays from (B, T, C, H, W) -> (B, T, H, W, C) expected by model
            reshaped = np.moveaxis(array, source = 2, destination = 4)
            normalized = normalize_timeseries(reshaped, axis = 1)
            recolored = aug_array_color(normalized)
            return recolored
        except AssertionError:
            return None

    def _get_naip_data(self, indexes):
        # Find list of IDs
        files_temp = [self.naipfiles[k] for k in indexes]
        # arrays come from PC in (C, H, W) format
        arrays = [np.load(f) for f in files_temp]

        try: 
            assert len(arrays) > 0
            assert all([x.shape == (4, self.unet_dim[0], self.unet_dim[1]) for x in arrays])
            # creat a single (B, C, H, W) array per batch
            batch = np.stack(arrays, axis = 0)
            in_shape = batch.shape
            # in case our incoming data is of different size than we want, define a trim amount
            trim = ((in_shape[2] - self.unet_dim[0])//2, (in_shape[3] - self.unet_dim[1])//2) 
            # If necessary, trim data to (-1, dims[0], dims[1])
            array = batch[:,:,trim[0]:self.unet_dim[0]+trim[0], trim[1]:self.unet_dim[1]+trim[1]]
            # rearrange arrays from (B, C, H, W) -> (B, H, W, C) expected by model
            reshaped = np.moveaxis(array, source = 1, destination = 3)
            normalized = reshaped/255.0
            recolored = aug_array_color(normalized)
            return recolored
        except AssertionError:
            return None
    
    def _get_lidar_data(self, indexes):
        if self.lidarfiles:

            files_temp = [self.lidarfiles[k] for k in indexes]
            arrays = [np.load(f) for f in files_temp]
            try:
                assert len(arrays) == self.batch_size
                assert all([x.shape == (1, self.unet_dim[0], self.unet_dim[1]) for x in arrays])
                batch = np.stack(arrays, axis = 0)
                in_shape = batch.shape
                trim = ((in_shape[2] - self.unet_dim[0])//2, (in_shape[3] - self.unet_dim[1])//2) 
                array = batch[:,:,trim[0]:self.unet_dim[0]+trim[0], trim[1]:self.unet_dim[1]+trim[1]]
                reshaped = np.moveaxis(array, source = 1, destination = 3)
                normalized = reshaped/100.0
                return normalized
            except AssertionError:
                return None
        else:
            return None

    def _process_y(self, indexes):
        # get label files for current batch
        lc_files = [self.labelfiles[k] for k in indexes]
        lc_arrays = [np.load(file) for file in lc_files]
        try:
            assert len(lc_arrays) == self.batch_size
            assert all([x.shape == (1, self.unet_dim[0], self.unet_dim[1]) for x in lc_arrays])
            lc = np.stack(lc_arrays, axis = 0) #(B, C, H, W)
            int_labels = lc.astype(int)

        # reduce the number of classes 
            merged_labels = merge_classes(cond_array = int_labels, trans = self.trans, out_array = int_labels)

            if self.lufiles:
                lu_files = [self.lufiles[k] for k in indexes]
                lu_arrays = [np.load(file) for file in lu_files]
                try:
                    assert len(lu_arrays) == self.batch_size
                    assert all([x.shape == (1, self.unet_dim[0], self.unet_dim[1]) for x in lu_arrays])                
                    lu = np.stack(lu_arrays, axis = 0) #(B, C, H, W)
                    y = merge_classes(cond_array = lu, trans = [(82,9), (84,10)], out_array = merged_labels)
                except AssertionError:
                    return None
            else:
                y = merged_labels

            # If necessary, trim data to (-1, dims[0], dims[1])
            in_shape = y.shape
            trim = ((in_shape[2] - self.unet_dim[0])//2, (in_shape[3] - self.unet_dim[1])//2) 
            array = y[:,:,trim[0]:self.unet_dim[0]+trim[0], trim[1]:self.unet_dim[1]+trim[1]]

            # shift range of categorical labels from [1, n_classes] to [0, n_classes]
            zeroed = array - 1
            # create one-hot representation of classes
            one_hot = tf.one_hot(zeroed, self.n_classes)
            # one_hot = to_one_hot(zeroed, self.n_classes)
            return tf.squeeze(one_hot)

        except AssertionError:
            return None

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        lstmData = self._get_s2_data(indexes)

        naipData = self._get_naip_data(indexes)

        lidarData = self._get_lidar_data(indexes)

        labels = self._process_y(indexes)

        if type(lidarData) == np.ndarray:
            unetData = np.concatenate([naipData, lidarData], axis = -1)
        else:
            unetData = naipData

        feats = [unetData, lstmData]
        if any([type(dat) == type(None) for dat in feats]):
            return self.__getitem__(randint(0, len(self.indexes) - self.batch_size))

        if self.to_fit:
            if type(labels) == type(None):
                return self.__getitem__(randint(0, len(self.indexes) - self.batch_size))
            # feats, labels = split_timeseries(rearranged)
            # we can't have nans in label
            else:
                return feats, labels
        else:
            return feats