# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:50:44 2020

@author: MEvans
"""
import tensorflow as tf

def augColor(x):
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x
  
  
def augImg(img):
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
    x = tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
      #x = zoom(x, outDims)
      #since were gonna map_fn this on a 4d image, output must be 3d, so squeeze the artificial 'sample' dimension
    return tf.squeeze(x)

def parse_tfrecord(example_proto, ftDict):
    """The parsing function.
    Read a serialized example into the structure defined by FEATURES_DICT.
    Args:
      example_proto: a serialized Example.
    Returns: 
      A dictionary of tensors, keyed by feature name.
    """
    return tf.io.parse_single_example(example_proto, ftDict)


def to_tuple(inputs, features):
    """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
    Args:
      inputs (dict): A dictionary of tensors, keyed by feature name. Response
      variable must be the last item.
      n (int): Number of input features (i.e. bands)
    Returns: 
      A dtuple of (inputs, outputs).
    """
    inputsList = [inputs.get(key) for key in dictionary.keys()]
    stacked = tf.stack(inputsList, axis=0)
    # Convert from CHW to HWC
    stacked = tf.transpose(stacked, [1, 2, 0])
    stacked = augImg(stacked)
    # return the features and labels
    return stacked[:,:,:n], stacked[:,:,n:]


def get_dataset(pattern):
    """Function to read, parse and format to tuple a set of input tfrecord files.
    Get all the files matching the pattern, parse and convert to tuple.
    Args:
      pattern: A file pattern to match in a Cloud Storage bucket.
    Returns: 
      A tf.data.Dataset
    """
    glob = tf.gfile.Glob(pattern)
    dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
    dataset = dataset.map(to_tuple, num_parallel_calls=5)
    return dataset

def get_training_dataset(path, batch):
	"""
    Get the preprocessed training dataset
    Args:
        path (str): file path to directory containing training tfrecords
        batch (int): batch size for training
    Returns: 
      A tf.data.Dataset of training data.
    """
	glob = path
	dataset = get_dataset(glob)
	dataset = dataset.shuffle(8000).batch(batch).repeat()
	return dataset

def get_eval_dataset(path):
	"""
    Get the preprocessed evaluation dataset
    Args:
        path (str): file path to directory containing evaluation tfrecords
    Returns: 
      A tf.data.Dataset of evaluation data.
    """
	glob = path
	dataset = get_dataset(glob)
	dataset = dataset.batch(1).repeat()
	return dataset