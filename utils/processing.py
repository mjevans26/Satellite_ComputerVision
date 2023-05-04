# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:50:44 2020

@author: MEvans
"""
import tensorflow as tf
import numpy as np

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

def aug_color(img):
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
  
def aug_img(img):
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
    
def normalize(x, axes=[2], epsilon=1e-8, moments = None, splits = None):
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
    def normalize_tensor(x):
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
        normed = [normalize_tensor(tensor) for tensor in tensors]
        normed.append(dontNorm)
        # gather normalized splits into single tensor
        x_normed = tf.concat(normed, axis = 2)
    else:
        x_normed = normalize_tensor(x)

    return x_normed 

def rescale(img, axes = [2], epsilon=1e-8, moments = None, splits = None):
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
    def rescale_tensor(img):
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
        rescaled = [rescale_tensor(tensor) for tensor in tensors]
        # gather normalized splits into single tensor
        img_rescaled = tf.concat(rescaled, axis = 2)
    else:
        img_rescaled = rescale_tensor(img)
        
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
    bands = aug_color(bands)
    bands = rescale(bands, axes = axes, moments = moments, splits = splits)
    
    if one_hot:
      hotStack = tf.concat(hotList, axis = 2)
      stacked = tf.concat([bands, hotStack, res], axis =2)
    else:
      stacked = tf.concat([bands, res], axis = 2)
    
    # perform morphological augmentation
    stacked = aug_img(stacked)
    
    feats = stacked[:, :, :-res.shape[2]]
    labels = stacked[:, :, -res.shape[2]:]
    labels = tf.where(tf.greater(labels, 1.0), 1.0, labels)
    return feats, labels
#    stacked = tf.stack(inputsList, axis=0)
#    # Convert from CHW to HWC
#    stacked = tf.transpose(stacked, [1, 2, 0])
#    stacked = aug_img(stacked)
#    #split input bands and labels
#    bands = stacked[:,:,:len(features)]
#    labels = stacked[:,:,len(features):]
#    # in case labels are >1
#    labels = tf.where(tf.greater(labels, 1.0), 1.0, labels)
#    # perform color augmentation on input features
#    bands = aug_color(bands)
#    # standardize each patch of bands
#    bands = normalize(bands, axes)
#    # return the features and labels
#    return bands, labels

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

#def tup(param = 0, extra = 10):
#    return param + extra
#
#def get_data(x, **kwargs):
#    return tup(**kwargs) + x^2
#
#def get_train_data(x, y, **kwargs):
#    return get_data(x, **kwargs) - y
#    
#get_train_data(2, 3, param = 1, extra = 3)

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

class UNETDataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, files,
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
        self.files = files
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.splits = splits
        self.moments = moments
        self.trans = translations
        self.indexes = np.arange(len(self.files))
        self.on_epoch_end()

        # do an initial shuffle for cases where the generator is called fresh at the start of each epoch
        if self.shuffle == True:
            print('shuffling')
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """
        return int(np.floor(len(self.files) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        print('the generator knows the epoch ended')
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            print('shuffling')
            np.random.shuffle(self.indexes)

    def _normalize(self, img, axes=[2], epsilon=1e-8):
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
            img (ndarray): nD image 
            axes (array): Array of ints. Axes along which to compute mean and variance, usually length n-1
            epsilon (float): small number to avoid dividing by zero
            moments (list<tpl>): list of global mean, std tuples for standardization
            splits (list): size(s) of groups of features to be kept together
        Return:
            tensor: nD image tensor normalized by channels
        """
        
        # define a basic function to normalize a 3d tensor
        def normalize_array(img):
    #        shape = tf.shape(x).numpy()
            # if we've defined global or per-channel moments...
            if self.moments:
                # cast moments to arrays for mean and variance
                mean = np.array([tpl[0] for tpl in self.moments], dtype = 'float32')
                std = np.array([tpl[1] for tpl in self.moments], dtype = 'float32')
            # otherwise, calculate moments along provided axes
            else:
                mean = np.nanmean(img, axes, keepdims = True)
                std = np.nanstd(img, axes, keepdims = True)
                # keepdims = True to ensure compatibility with input tensor

            # normalize the input tensor
            normed = (img - mean)/(std + epsilon)
            return normed
        

        # if splits are given, apply tensor normalization to each split
        if self.splits:
            splitLen = sum(self.splits)
            toNorm = img[:,:,0:splitLen]
            dontNorm = img[:,:,splitLen:]
            arrays = np.split(toNorm, self.splits, axis = -1)
            normed = [normalize_array(array) for array in arrays]
            normed.append(dontNorm)
            # gather normalized splits into single tensor
            img_normed = np.concatenate(normed, axis = -1)
        else:
            img_normed = normalize_array(img)

        return img_normed

    def _rescale(self, img, axes = -1, epsilon=1e-8):
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
        def rescale_array(img):
            if self.moments:
                minimum = np.array([tpl[0] for tpl in self.moments], dtype = 'float32')
                maximum = np.array([tpl[1] for tpl in self.moments], dtype = 'float32')
            else:
                minimum = np.nanmin(img, axis = axes, keepdims = True)
                maximum = np.nanmax(img, axis = axes, keepdims = True)
            scaled = (img - minimum)/((maximum - minimum) + epsilon)
    #        scaled = tf.divide(tf.subtract(img, minimum), tf.add(tf.subtract(maximum, minimum))
            return scaled
        
        # if splits are given, apply tensor normalization to each split
        if self.splits:
            arrays = np.split(img, self.splits, axis = -1)
            rescaled = [rescale_array(array) for array in arrays]
            # gather normalized splits into single tensor
            img_rescaled = np.concat(rescaled, axis = -1)
        else:
            img_rescaled = rescale_array(img)
            
        return img_rescaled

    def _aug_color(self, img: np.ndarray) -> np.ndarray:
        """Randomly change the brightness and contrast of an image
        ---Parameters---
            img np.ndarray:
        ---Return---
            np.ndarray: 
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

    def _aug_img(self, img: np.ndarray) -> np.ndarray:
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

    def _rescale_and_aug(self, x):
        recolored = self._aug_color(x)
        rescaled = self._rescale(recolored, axes = 2, moments = self.moments, splits = self.splits)
        return rescaled

    def _process_x(self, x):
        processed = np.array([self._rescale_and_aug(img) for img in x])
        return processed

    def _process_y(self, y):
        # cast the labels to int
        int_labels = y.astype(int)
        # reduce the number of classes from 12 to 8
        merged_labels = merge_classes(int_labels, self.trans)
        # shift range of categorical labels from [1, n_classes] to [0, n_classes]
        zeroed = merged_labels - 1
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
        
        # Find list of IDs
        files_temp = [self.files[k] for k in indexes]
        # arrays come from PC in (C, H, W) format
        arrays = [np.load(file) for file in files_temp]
        in_shape = arrays[0].shape
        print('in shape', in_shape)
        trim = ((in_shape[1] - self.dim[0])//2, (in_shape[2] - self.dim[1])//2) 
        print('trim', trim)
        # If necessary, trim data to (-1, dims[0], dims[1])
        array = [arr[:,trim[0]:trim[0]+self.dim[0], trim[1]:trim[1]+self.dim[1]] for arr in arrays]

        # creat a single (B, C, H, W) array per batch
        batch = np.stack(array, axis = 0)
        print('batch shape', batch.shape)
        # rearrange arrays from (B, C, H, W) -> (B, H, W, C) expected by model
        reshaped = np.moveaxis(batch, source = 1, destination = 3)
        # is 255 a nan value is landcover labels?
        reshaped[:,:,:,-1] = np.where(reshaped[:,:,:,-1] == 255, 0.0, reshaped[:,:,:,-1])

        # perform morphological augmentation - expects a 3D (H, W, C) image array
        augmented = np.array([aug_img(array) for array in reshaped])
        # print('augmented max', np.nanmax(augmented, axis = (0,1,2)))

        feats = augmented[:,:,:,0:self.n_channels]
        labels = augmented[:,:,:,self.n_channels:]
        print(np.sum(feats, axis = (1,2)))
        feats = self._process_x(feats)
        print('feat dims', feats.shape)        
        one_hot = self._process_y(labels)
        print('label dims', one_hot.shape)

        if self.to_fit:
            return feats, one_hot
        else:
            return feats

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

class LSTMDataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, files,
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
        self.n_timesteps = n_timesteps+1
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

class HybridDataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras model with U-Net and LSTM branches
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, s2files, naipfiles, labelfiles, lufiles, n_classes = 8,
                 to_fit=True, batch_size=32, unet_dim=(320, 320, 4), transitions = [(12,3), (11,3), (10,3), (9,8)],
                 lstm_dim = (6, 32, 32, 6), shuffle=True, lidarfiles = None):
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
    
    def _get_s2_data(self, indexes):

        files_temp = [self.s2files[k] for k in indexes]

        # arrays come from PC in (T, C, H, W) format
        arrays = [np.load(file) for file in files_temp]
        # creat a single (B, T, C, H, W) array
        batch = np.stack(arrays, axis = 0)
        # in case our incoming data is of different size than we want, define a trim amount
        trim = ((batch.shape[3] - self.lstm_dim[1])//2, (batch.shape[4] - self.lstm_dim[2])//2) 
        print(trim)
        array = batch[:, 0:self.n_timesteps,:,trim[0]:self.lstm_dim[1]+trim[0],trim[1]:self.lstm_dim[2]+trim[1]]

        print('batch shape', batch.shape)
        # rearrange arrays from (B, T, C, H, W) -> (B, T, H, W, C) expected by model
        reshaped = np.moveaxis(array, source = 2, destination = 4)
        print('reshaped shape', reshaped.shape)
        normalized = normalize_timeseries(reshaped, axis = 1)
        return normalized

    def _get_naip_data(self, indexes):
        # Find list of IDs
        files_temp = [self.naipfiles[k] for k in indexes]
        # arrays come from PC in (C, H, W) format
        arrays = [np.load(file) for file in files_temp]
        # creat a single (B, C, H, W) array per batch
        batch = np.stack(arrays, axis = 0)
        in_shape = batch.shape
        # in case our incoming data is of different size than we want, define a trim amount
        trim = ((in_shape[2] - self.unet_dim[0])//2, (in_shape[3] - self.unet_dim[1])//2) 
        # If necessary, trim data to (-1, dims[0], dims[1])
        array = batch[:,:,trim[0]:self.unet_dim[0]+trim[0], trim[1]:self.unet_dim[1]+trim[1]]
        # rearrange arrays from (B, C, H, W) -> (B, H, W, C) expected by model
        reshaped = np.moveaxis(array, source = 1, destination = 3)
        # is 255 a nan value is landcover labels?
        reshaped[:,:,:,-1] = np.where(reshaped[:,:,:,-1] == 255, 0.0, reshaped[:,:,:,-1])
        normalized = reshaped/255.0
        return normalized

    def _process_y(self, indexes):
        # get label files for current batch
        lc_files = [self.labelfiles[k] for k in indexes]
        lu_files = [self.lufiles[k] for k in indexes]
        lu_arrays = [np.load(file) for file in lu_files]
        lc_arrays = [np.load(file) for file in lc_files]
        
        
        # cast the labels to int
        lu = np.stack(lu_arrays, axis = 0) #(B, C, H, W)
        lc = np.stack(lc_arrays, axis = 0) #(B, C, H, W)
        
        int_labels = lc.astype(int)
        # reduce the number of classes 
        merged_labels = merge_classes(cond_array = int_labels, trans = self.trans, out_array = int_labels)
        y = merge_classes(cond_array = lu, trans = [(82,9), (84,10)], out_array = merged_labels)

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

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        lstmData = self._get_s2_data(indexes)

        unetData = self._get_naip_data(indexes)

        labels = self._process_y(indexes)

        feats = [unetData, lstmData]

        if self.to_fit:

            # feats, labels = split_timeseries(rearranged)
            # we can't have nans in label
            return feats, labels
        else:
            return feats