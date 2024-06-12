"""
Created on Fri Mar 20 10:50:44 2020

@author: MEvans
"""
import tensorflow as tf
import numpy as np
import math
import os
import copy
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

def get_file_id(f:str, delim:str = '_', parts:slice = slice(3,5)):
    """Return a unique identifyier from a file name

    Params
    ---
    f: str
        file basename
    delim: str
        delimiter optionally splitting filename into parts
    parts: slice
        slice identifying the parts to return

    Returns
    ---
    tuple: tuple of filename pieces
    """
    stem = str(Path(f).stem)
    splits = stem.split(delim)
    ids = splits[parts]
    return tuple(ids)

def match_files(urls, vars, delim:str = '_', parts:slice = slice(3,5), subset: set = None):
    """Align files by unique id among variables
    Params
    ---
    urls: list:str
      unordered list of all filepaths to be sorted and aligned by variable
    vars: dict
      key, value pairs with variable names as keys (e.g., 'naip'). value = None will skip that variable
    delim: str
        delimiter optionally splitting filename into parts
    parts: slice
        slice identifying the parts to return
    subset: set
      optional. unique ids with which to further subset the returned files

    Returns
    ---
    dict: key, value pairs for each valid key in vars. variable names are key (e.g. 'naip') and values are corresponding list of files
    """

    #print(len(subset))
    vars_copy = copy.deepcopy(vars)

    files_dic = {key:[url for url in urls if f'/{key}/' in url] for key in vars_copy.keys() if vars_copy[key]['files'] is not None}

    ids = [set([get_file_id(f, delim, parts) for f in files]) for files in files_dic.values()] # list of sets per var

    intersection = set.intersection(*ids)

    if subset:
        intx = intersection.intersection(subset)
    else:
        intx = intersection
    for var, ls in files_dic.items():
       subset = [f for f in ls if get_file_id(f, delim, parts) in intx]
       subset.sort()
       vars_copy[var].update({"files": subset})

    return vars_copy

def split_files(files, labels = ['label', 'lu', 'naip', 'lidar', 's2'], delim = '_', parts = slice(3,5)):
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
  def get_file_id(f, parts):
    stem = str(Path(f).stem)
    splits = stem.split(delim)
    ids = splits[parts]
    return tuple(ids)

  indices = [set([get_file_id(f, parts) for f in files if label in Path(f).parts]) for label in labels]
  intersection = set.intersection(*indices)
  out_files = [[f for f in files if label in Path(f).parts and get_file_id(f, parts) in intersection] for label in labels]
  return out_files

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
    def __init__(self, labelfiles = None, s2files = None, naipfiles = None,
                 hagfiles = None, lidarfiles = None, lufiles = None,
                 demfiles = None, ssurgofiles = None,
                 to_fit=True, batch_size=32, unet_dim=(256, 256),
                 n_channels=4, n_classes = 8, shuffle=True,
                 splits = None, moments = None,
                lc_transitions = [(12,3), (11,3), (10,3), (9,8), (255, 0)],
                lu_transitions = [(82,9), (84,10)]):
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
        self.hagfiles = hagfiles
        self.demfiles = demfiles
        self.ssurgofiles = ssurgofiles
        self.lidarfiles = lidarfiles
        self.labelfiles = labelfiles
        self.lufiles = lufiles
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.unet_dim = unet_dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.splits = splits
        self.moments = moments
        self.lc_trans = lc_transitions
        self.lu_trans = lu_transitions
        self.indexes = np.arange(len(self.naipfiles))
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

    @staticmethod
    def load_numpy_url(url):

        if os.path.exists(url):
            data = np.load(url)
        else:
            response = requests.get(url)
            response.raise_for_status()
            data = np.load(io.BytesIO(response.content))

        return(data)

    def _load_numpy_data(self, files_temp):
        arrays = [UNETDataGenerator.load_numpy_url(f) for f in files_temp]
        return(arrays)

    def _get_unet_data(self, files_temp, add_nan_mask = False,rescale_val=False):
        # arrays come from PC in (C, H, W) format
        arrays = self._load_numpy_data(files_temp)
        try:
            assert len(arrays) > 0
            assert all([len(x.shape) == 3 for x in arrays]), 'all arrays not 3D'
            # ensure all arrays are C, H, W to start
            chw = [np.moveaxis(x, source = -1, destination = 0) if x.shape[-1] < x.shape[0] else x for x in arrays]
            if rescale_val is not False:
                chw = [x/rescale_val for x in chw]
            if add_nan_mask == True:
                chw_new = []
                for cur_array in chw:                 
                        
                    mask_channel = np.zeros([cur_array.shape[1], cur_array.shape[2]])
                    # Create a random array to be used to replace the original data
                    for arr_2d in cur_array:
                        nans = np.isnan(arr_2d)
                        bads = arr_2d < -5000
                        mask_channel[nans==True] = 1
                        mask_channel[bads==True] = 1
                        arr_2d[mask_channel==1] = np.random.randn((mask_channel==1).sum())
                        # arr_2d[nans==True] = np.random.uniform()
                        #arr_2d[np.isnan(arr_2d)] = np.random.randn(len(arr_2d[np.isnan(arr_2d)]))
                    #print("AFTER FIX:",np.isnan(cur_array).sum())
                    #cur_array = np.vstack((cur_array, mask[None,:,:]))


                    """randarr = np.random.uniform(size=cur_array.shape)*cur_array.max()
                    # Build a mask layer to use in the replacement
                    n_cols = cur_array.shape[2]
                    n_rows = cur_array.shape[1]
                    mask_channel = np.ones((n_rows, n_cols), dtype=np.int8)
                    np.any(cur_array == np.nan, axis=0, out=mask_channel)
                    # Replace the values in any of the channels where the mask_channel is 0 with the values from the random array
                    cur_array[:, mask_channel == 1] = randarr[:, mask_channel == 1]
                    cur_array[:, mask_channel == 1] = randarr[:, mask_channel == 1] """
                    cur_array = np.append(cur_array, mask_channel[np.newaxis, :, :], axis=0)
                    #print("AFTER:",np.isnan(cur_array).sum())
                    chw_new.append(cur_array)
                chw = chw_new
            batch = np.stack(chw, axis = 0)
            assert np.isnan(batch).sum() < 1, 'nans in batch, skipping'
            in_shape = batch.shape
            # in case our incoming data is of different size than we want, define a trim amount
            trim = ((in_shape[2] - self.unet_dim[0])//2, (in_shape[3] - self.unet_dim[1])//2)
            # If necessary, trim data to (-1, dims[0], dims[1])
            array = batch[:,:,trim[0]:self.unet_dim[0]+trim[0], trim[1]:self.unet_dim[1]+trim[1]]
            # rearrange arrays from (B, C, H, W) -> (B, H, W, C) expected by model

            reshaped = np.moveaxis(array, source = 1, destination = 3)
            return reshaped
        except AssertionError as msg:
          print(msg)
          return None
    def _get_naip_data(self, indexes):
        files_temp = [self.naipfiles[k] for k in indexes]
        naip = self._get_unet_data(files_temp,rescale_val=255.0)
        if type(naip) == np.ndarray:

            if self.to_fit:
                recolored = aug_array_color(naip)
                return recolored
            return naip
        #else:
            #return naip

    def _get_s2_data(self, indexes):
        files_temp = [self.s2files[k] for k in indexes]
        s2 = self._get_unet_data(files_temp,rescale_val=10000.0)
        if type(s2) == np.ndarray:
            if self.to_fit:
                recolored = aug_array_color(s2)
                return recolored
            else:
                return s2
        #else:
            #return s2

    def _get_lidar_data(self, indexes):
        files_temp = [self.lidarfiles[k] for k in indexes]
        lidar = self._get_unet_data(files_temp,True,rescale_val=100)
        if type(lidar) == np.ndarray:
            return lidar

    def _get_hag_data(self, indexes):
        files_temp = [self.hagfiles[k] for k in indexes]
        hag = self._get_unet_data(files_temp, True, rescale_val=100)
        if type(hag) == np.ndarray:
            return hag
        #else:
         #   return hag

    def _get_dem_data(self, indexes):
        files_temp = [self.demfiles[k] for k in indexes]
        dem = self._get_unet_data(files_temp,True,rescale_val=2000.0)
        if type(dem) == np.ndarray:
           # we are going to use the min and max elevations across the chesapeake
          return dem
        #else:
         # return dem

    def _get_ssurgo_data(self, indexes):
        files_temp = [self.ssurgofiles[k] for k in indexes]
        ssurgo = self._get_unet_data(files_temp)
        if type(ssurgo) == np.ndarray:
            return ssurgo

    def _process_y(self, indexes):
        # get label files for current batch
        lc_files = [self.labelfiles[k] for k in indexes]
        # lc_arrays = [np.load(file) for file in lc_files]
        lc_arrays = self._load_numpy_data(lc_files)
        
        try:
            assert len(lc_arrays) == self.batch_size
            assert all([x.shape == (1, self.unet_dim[0], self.unet_dim[1]) for x in lc_arrays])
            lc = np.stack(lc_arrays, axis = 0) #(B, C, H, W)
            int_labels = lc.astype(int)

            # optionally reduce the number of classes
            if self.lc_trans:
              merged_labels = merge_classes(cond_array = int_labels, trans = self.lc_trans, out_array = int_labels)
            else:
              merged_labels = int_labels

            if self.lufiles:
                lu_files = [self.lufiles[k] for k in indexes]
                # lu_arrays = [np.load(file) for file in lu_files]
                lu_arrays = self._load_numpy_data(lu_files)
                try:
                    assert len(lu_arrays) == self.batch_size
                    assert all([x.shape == (1, self.unet_dim[0], self.unet_dim[1]) for x in lu_arrays])
                    lu = np.stack(lu_arrays, axis = 0) #(B, C, H, W)
                    y = merge_classes(cond_array = lu, trans = self.lu_trans, out_array = merged_labels)
                except AssertionError:
                    return None
            else:
                y = merged_labels

            # If necessary, trim data to (-1, dims[0], dims[1])
            in_shape = y.shape
            trim = ((in_shape[2] - self.unet_dim[0])//2, (in_shape[3] - self.unet_dim[1])//2)
            array = y[:,:,trim[0]:self.unet_dim[0]+trim[0], trim[1]:self.unet_dim[1]+trim[1]]

            # shift range of categorical labels from [1, n_classes] to [0, n_classes]
            zeroed = array
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

        datasets = []

        if self.s2files:
            s2Data = self._get_s2_data(indexes)
            datasets.append(s2Data)

        if self.naipfiles:
            naipData = self._get_naip_data(indexes)
            #print("appending Naip data",type(naipData))
            datasets.append(naipData)

        if self.hagfiles:
            hagData = self._get_hag_data(indexes)
            datasets.append(hagData)

        if self.demfiles:
            demData = self._get_dem_data(indexes)
            # print('dem', demData.shape)
            #print("appening DEM data",type(demData))
            datasets.append(demData)

        if self.ssurgofiles:
            ssurgoData = self._get_ssurgo_data(indexes)
            # print('ssurgo', ssurgoData.shape
            #print("appending ssurgoData",type(ssurgoData))
            datasets.append(ssurgoData)

        if self.lidarfiles:
            lidarData = self._get_lidar_data(indexes)
            datasets.append(lidarData)

        if any([type(dat) != np.ndarray for dat in datasets]):
          pass
        else:
            xData = np.concatenate(datasets, axis = -1)

        if self.to_fit:
            labels = self._process_y(indexes)
            # perform morphological augmentation - expects a 3D (H, W, C) image array
            stacked = np.concatenate([xData, labels], axis = -1)
            morphed = aug_array_morph(stacked)
            # print('augmented max', np.nanmax(augmented, axis = (0,1,2)))

            feats = morphed[:,:,:,0:self.n_channels]
            labels = morphed[:,:,:,self.n_channels:]
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
        lc_files = self.load_numpy_data(files_temp)
        lc_arrays = [np.squeeze(f) for f in lc_files] # make all labels 2D to start
        try:
            assert len(lc_arrays) == self.batch_size
            lc = np.stack(lc_arrays, axis = 0) #(B, H, W)
            int_labels = lc.astype(int)
            binary = np.where(int_labels > 1, 1, int_labels)
            # If necessary, trim data to (-1, dims[0], dims[1])
            in_shape = binary.shape # -> (B, H, W)
            trim = ((in_shape[1] - self.unet_dim[0])//2, (in_shape[2] - self.unet_dim[1])//2)
            array = binary[:,trim[0]:self.unet_dim[0]+trim[0], trim[1]:self.unet_dim[1]+trim[1]]

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

        s2Data = self._get_s2_data(indexes)

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
        # rearrange arrays from (B, T, C, H, W) -> (B, T, H, W, C) expected by model
        reshaped = np.moveaxis(batch, source = 2, destination = 4)
        normalized = normalize_timeseries(reshaped, axis = 1)
        # harmonized = add_harmonic(normalized)
        if self.to_fit:
            rearranged = rearrange_timeseries(normalized, self.n_channels)
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

class HybridDataGenerator(UNETDataGenerator):
    """Generates data for Keras model with U-Net and LSTM branches
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, s1files,
                lstm_dim = (6, 32, 32, 6),
                lc_transitions = [(12,3), (11,3), (10,3), (9,8), (255, 0)],
                lu_transitions = [(82,9), (84,10)],
                 *args, **kwargs):
        """Class Initialization

        Params
        ---
        unet_dim: tuple
            desired unet image H, W, C dimensions
        lstm_dim: tuple
            desired lstm image T, H, W, C dimensions
        lc_transitions: list
            list of ('from', to') tuples defining optional categorical reclassifications for lc data
        lu_transitions: list
            list of ('from', 'to') tuples defining optional categorical reclassificaitons for lu data

        Return
        ---
        tuple: three arrays containing batch of corresponding sentinel-2, naip, and label data
        """
        super().__init__(*args, **kwargs)
        self.s1files = s1files
        self.lc_trans = lc_transitions
        self.lu_trans = lu_transitions
        self.lstm_dim = lstm_dim
        self.n_timesteps = lstm_dim[0]
        self.on_epoch_end()

    def _get_s2_data(self, indexes):

        files_temp = [self.s2files[k] for k in indexes]
        # arrays come from PC in (T, C, H, W) format
        # arrays = [np.load(f) for f in files_temp]
        arrays = self._load_numpy_data(files_temp)

        try:
            assert len(arrays) > 0, 'no arrays'
            assert all([x.shape == (self.lstm_dim[0], self.lstm_dim[3], self.lstm_dim[1], self.lstm_dim[2]) for x in arrays]), [x.shape for x in arrays]

            # creat a single (B, T, C, H, W) array
            batch = np.stack(arrays, axis = 0)
            # in case our incoming data is of different size than we want, define a trim amount
            trim = ((batch.shape[3] - self.lstm_dim[1])//2, (batch.shape[4] - self.lstm_dim[2])//2)

            array = batch[:, 0:self.n_timesteps,:,trim[0]:self.lstm_dim[1]+trim[0],trim[1]:self.lstm_dim[2]+trim[1]]

            # rearrange arrays from (B, T, C, H, W) -> (B, T, H, W, C) expected by model
            reshaped = np.moveaxis(array, source = 2, destination = 4)
            normalized = normalize_timeseries(reshaped, axis = 1)
            if self.to_fit:
                recolored = aug_array_color(normalized)
            else:
                recolored = normalized
            return recolored
        except AssertionError as msg:
            print(msg)
            return None
        
    def _get_s1_data(self, indexes):

        files_temp = [self.s1files[k] for k in indexes]
        # arrays come from PC in (T, C, H, W) format
        # arrays = [np.load(f) for f in files_temp]
        arrays = self._load_numpy_data(files_temp)

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
            normalized = normalize_timeseries(reshaped, maxval = 0.5, axis = 1)
            return normalized
        except AssertionError:
            return None

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        unetDatasets = []
        lstmDatasets = []
        if self.s2files:
            s2Data = self._get_s2_data(indexes)
            lstmDatasets.append(s2Data)
        if self.s1files:
            s1Data = self._get_s1_data(indexes)
            lstmDatasets.append(s1Data)
        if self.naipfiles:
            naipData = self._get_naip_data(indexes)
            unetDatasets.append(naipData)
        if self.demfiles:
            demData = self._get_dem_data(indexes)
            unetDatasets.append(demData)
        if self.hagfiles:
            hagData = self._get_hag_data(indexes)
            unetDatasets.append(hagData)
        if self.lidarfiles:
            lidarData = self._get_lidar_data(indexes)
            unetDatasets.append(lidarData)
        if self.ssurgofiles:
            ssurgoData = self._get_ssurgo_data(indexes)
            unetDatasets.append(ssurgoData)

        if any([type(dat) != np.ndarray for dat in unetDatasets + lstmDatasets]):
          pass
        else:
          unetData = np.concatenate(unetDatasets, axis = -1)
          lstmData = np.concatenate(lstmDatasets, axis = -1)
        feats = [unetData, lstmData]
        # if type(lidarData) == np.ndarray:
        #     unetData = np.concatenate([naipData, lidarData], axis = -1)
        # else:
        #     unetData = naipData

        # feats = [unetData, s2Data]
        # if any([type(dat) == type(None) for dat in feats]):
        #     return self.__getitem__(randint(0, len(self.indexes) - self.batch_size))

        if self.to_fit:
            labels = self._process_y(indexes)
            if type(labels) == type(None):
                pass
            # feats, labels = split_timeseries(rearranged)
            # we can't have nans in label
            else:
                return feats, labels
        else:
            return feats
