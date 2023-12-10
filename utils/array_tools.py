# -*- coding: utf-8 -*-
"""
Created on Fri Mar 226 10:50:44 2023

@author: MEvans
"""

import numpy as np
import math
from random import shuffle, randint, uniform

def make_harmonics(times: np.ndarray, timesteps, dims):
    """Create arrays of sin and cos representations of time
    Parameters:
        times (np.ndarray): 1D array of start times
        timesteps (int): number of annual timesteps
        dims (tpl): H, W dimensions of output data
    Returns:
        np.ndarray: 4D array (B, (dims), 2) with 
    """
    xys = [sin_cos(time, timesteps) for time in times] # use the T dimension to get number of intervals
    # r = deg_to_radians(lat) # convert latitude to radians
    out = np.stack([np.stack([np.full(dims, x), np.full(dims, y)], axis = -1) for x,y in xys], axis = 0)
    return out
    
def merge_classes(cond_array, trans, out_array):
    """Reclassify categorical array values
    Parameters
    ---
    cond_array: np.ndarray
      array with values to be evaluated by conditional expression
    trans: list[tpl]
      tuples containing condition and value to return where true
    array: np.ndarray
      array to be returned where condition false
    Returns
    ---
    np.darray
        reclassified array same shape and size as input
    """
    for x,y in trans:
      out_array[cond_array == x] = y
    return out_array
        
 
def normalize_array(img, axes=[2], epsilon=1e-8, moments = None, splits = None):
    """
    Standardize incoming image patches by mean and variance.

    Moments can be calculated based on patch data by providing axes:      
    To standardize each pixel use axes = [2]
    To standardize each channel use axes = [0, 1]
    To standardize globally use axes = [0, 1, 2]

    To standardize by global, or per-channel moments supply a list of [mean, variance] tuples.
    To standardize groups of channels separately, identify the size of each group. Groups of
    channels must be stacked contiguously and group sizes must sum to the total # of channels
    
    Parameters
    ---
        img: np.ndarray
            nD image (usually 3d) to be normalized
        axes: list: int
            Array of ints. Axes along which to compute mean and variance, usually length n-1
        epsilon: float
            small number to avoid dividing by zero
        moments: list:tpl:int
            list of global mean, std tuples for standardization
        splits: list:int
            size(s) of groups of features to be kept together
    Return:
        tensor: nD image tensor normalized by channels
    """
    
    # define a basic function to normalize a 3d tensor
    def normalize(img):
#        shape = tf.shape(x).numpy()
        # if we've defined global or per-channel moments...
        if moments:
            # cast moments to arrays for mean and variance
            mean = np.array([tpl[0] for tpl in moments], dtype = 'float32')
            std = np.array([tpl[1] for tpl in moments], dtype = 'float32')
        # otherwise, calculate moments along provided axes
        else:
            mean = np.nanmean(img, axes, keepdims = True)
            std = np.nanstd(img, axes, keepdims = True)
            # keepdims = True to ensure compatibility with input tensor

        # normalize the input tensor
        normed = (img - mean)/(std + epsilon)
        return normed
    
    # if splits are given, apply tensor normalization to each split
    if splits:
        splitLen = sum(splits)
        toNorm = img[:,:,0:splitLen]
        dontNorm = img[:,:,splitLen:]
        arrays = np.split(toNorm, splits, axis = -1)
        normed = [normalize(array) for array in arrays]
        normed.append(dontNorm)
        # gather normalized splits into single tensor
        img_normed = np.concatenate(normed, axis = -1)
    else:
        img_normed = normalize(img)

    return img_normed

def rescale_array(img, axes = -1, epsilon=1e-8, moments = None, splits = None):
    """
    Rescale incoming image patch to [0,1] based on min and max values
    
    Min, max can be calculated based on patch data by providing axes:      
    To rescale each pixel use axes = [2]
    To rescale each channel use axes = [0, 1]
    To rescale globally use axes = [0, 1, 2]

    To rescale by global, or per-channel moments supply a list of [mean, variance] tuples.
    To rescale groups of channels separately, identify the size of each group. Groups of
    channels must be stacked contiguously and group sizes must sum to the total # of channels
    
    Parameters
    ---
        img: np.ndarray
            array to be rescaled, usually 3D (H,W,C)
        axes: list: int
            Array of ints. Axes along which to compute mean and variance, usually length n-1
        epsilon: float
            small number to avoid dividing by zero
        moments: list:tpl:int
            optional, list of global mean, std tuples for standardization
        splits: list:int
            optional, size(s) of groups of features to be kept together
    Return:
        tensor: 3D tensor of same shape as input, with values [0,1]
    """
    def rescale(img):
        if moments:
            minimum = np.array([tpl[0] for tpl in moments], dtype = 'float32')
            maximum = np.array([tpl[1] for tpl in moments], dtype = 'float32')
        else:
            minimum = np.nanmin(img, axis = axes, keepdims = True)
            maximum = np.nanmax(img, axis = axes, keepdims = True)
        scaled = (img - minimum)/((maximum - minimum) + epsilon)
#        scaled = tf.divide(tf.subtract(img, minimum), tf.add(tf.subtract(maximum, minimum))
        return scaled
    
    # if splits are given, apply tensor normalization to each split
    if splits:
        arrays = np.split(img, splits, axis = -1)
        rescaled = [rescale(array) for array in arrays]
        # gather normalized splits into single tensor
        img_rescaled = np.concat(rescaled, axis = -1)
    else:
        img_rescaled = rescale(img)
        
    return img_rescaled

def aug_array_color(img: np.ndarray) -> np.ndarray:
    """Randomly change the brightness and contrast of an image
    Parameters
    ---
    img: np.ndarray
        image to be adjusted

    Return
    ---
    np.ndarray: input array with brightness and contrast adjusted 
    """
    dims = len(img.shape)
    n_ch = img.shape[-1]
    axes = (0,1) if dims == 3 else (1,2)

    contra_adj = 0.05
    bright_adj = 0.05

    ch_mean = np.nanmean(img, axis = axes, keepdims = True)
    # print('channel means', ch_mean)
    contra_mul = uniform(a = 1-contra_adj, b = 1+contra_adj)

    bright_mul = uniform(a = 1 - bright_adj, b = 1+bright_adj)

    recolored = (img - ch_mean) * contra_mul + (ch_mean * bright_mul)
    return recolored

def aug_array_morph(img: np.ndarray) -> np.ndarray:
    """
    Perform morphological image augmentation on image array
    Parameters:
        img (np.ndarray): 4D or 3D channels last image array
    Returns:
        np.ndarray: 3D channels last image array 
    """
    dims = len(img.shape)
    v_axis = 0 if dims == 3 else 1
    h_axis = 1 if dims == 3 else 2

    # flip array up/down
    x = np.flip(img, axis = v_axis)
    # flip array left_right
    x = np.flip(x, axis = h_axis)
    x = np.rot90(x, uniform(0,4), axes = (v_axis, h_axis))

    return x

def normalize_timeseries(arr, maxval = 10000, minval = 0, axis = -1, e = 0.00001):
  # normalize band values across timesteps
  normalized = (arr-minval)/(maxval-minval)
#   mn = np.nanmean(arr, axis = axis, keepdims = True)
#   std = np.nanstd(arr, axis = axis, keepdims = True)
#   normalized = (arr - mn)/(std+e)
  # replace nans with zeros?
  finite = np.where(np.isnan(normalized), 0.0, normalized)
  return finite

def rearrange_timeseries(arr: np.ndarray, nbands: int) -> np.ndarray:
  """ Randomly rearange 3d images in a timeseries

  Changes the startpoint of a temporal sequence of 3D images stored in a 4D array
  while maintaining relative order.
  
  Parameters
  ---
  arr: np.ndarray
    5D (B, T, H, W, C) array to be rearranged
  nbands: int
    size of the last array dimension corresponding to image bands/channels

  Returns
  ---
  np.ndarray
    5D array of same size/shape as input
  """

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
  return(rearranged)

def split_timeseries(arr: np.ndarray) -> tuple:
  """Divide a timeseries of 3D images into a series of images and labels

  Parameters
  ---
  arr: np.ndarray
    5D (B, T, H, W, C) array to be split

  Returns
  ---
  tuple
    two 5D timeseries arrays
  """

  feats = arr[:,0:-1,:,:,:]
  labels = arr[:,-1,:,:,0:nbands]

  # confirm there are no all-nan images in labels
  batch_sums = np.sum(labels, axis = (1,2,3))
  if 0.0 in batch_sums:
    print('all nan labels, reshuffling')
    feats, labels = rearrange_timeseries(arr, nbands)

  return(feats, labels)

def sin_cos(t:int, freq:int = 6) -> tuple:
    x = t/freq
    theta = 2*math.pi * x
    return (math.sin(theta), math.cos(theta))

def add_harmonic(timeseries: np.ndarray):
    """ add harmonic variables to an imagery timeseries. currently assumes first image is start of year
    B, T, H, W, C
    """
    in_shape = timeseries.shape
    timesteps = in_shape[1]
    tpls = [sin_cos(t, timesteps) for t in range(timesteps)]
    xys = [np.stack([np.full((in_shape[0], in_shape[2], in_shape[3]), x), np.full((in_shape[0], in_shape[2], in_shape[3]), y)], axis = -1) for x,y in tpls]
    harmonics = np.stack(xys, axis = 1)
    harmonic_timeseries = np.concatenate([timeseries, harmonics], axis = -1)
    return harmonic_timeseries
