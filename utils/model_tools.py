# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:53:59 2020

@author: MEvans
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import models
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import backend
import numpy as np

def gen_dice(y_true, y_pred, eps=1e-6, global_weights = None):

    """both tensors are [b, h, w, classes] and y_pred is in logit form

		https://stackoverflow.com/questions/49012025/generalized-dice-loss-for-multi-class-segmentation-keras-implementation

		This implementation will calculate class weights per batch - could alternatively substitute contants based on overall
		training dataset proportions
		
		Parameters:
			y_true (array): 4-d tensor of one-hot labels
			y_pred (array): 4-d tensor of class predictions
			eps (float): fixed weight to prevent division by 
            global_weights (list): list of per-class weights
		"""

    # [b, h, w, classes]
    # pred_tensor = tf.nn.softmax(y_pred)
    pred_tensor = y_pred
    y_true_shape = tf.shape(y_true)

    # [b, h*w, classes]
    y_true = tf.reshape(y_true, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
    y_pred = tf.reshape(pred_tensor, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])


    # we can use the batchwise weights or global weights - select one of them
    # -----------------------------------------------------------------------
    if global_weights:
    # global weights
    # --------------
        weights = tf.constant(global_weights, shape = (1, len(global_weights)), dtype = tf.float32)
    else:
    # batchwise weight
    # ----------------
    # to use batch-wise weights, count how many of each class are present in each image, 
        counts = tf.reduce_sum(y_true, axis=-1)
        weights = 1. / (counts ** 2)
        # if there are zero, then assign them a fixed weight of eps
        weights = tf.where(tf.math.is_finite(weights), weights, eps)        
    
    #[b, classes]
    multed = tf.reduce_sum(y_true * y_pred, axis=1)
    summed = tf.reduce_sum(y_true + y_pred, axis=1)

    # [b]
    numerators = tf.reduce_sum(weights*multed, axis=-1)
    denom = tf.reduce_sum(weights*summed, axis=-1)
    dices = 1. - 2. * numerators / denom
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    return tf.reduce_mean(dices)

def weighted_bce(y_true, y_pred, weight):
    """
    Compute the weighted binary cross entropy between predictions and observations
    Parameters:
        y_true (): 2D tensor of labels
        y_pred (): 2D tensor of probabilities
        
    Returns:
        2D tensor
    """
    bce = tf.nn.weighted_cross_entropy_with_logits(labels = y_true, logits = y_pred, pos_weight = weight)
    return tf.reduce_mean(bce)

# def dice_coef(y_true, y_pred, smooth=1, weight=0.5):
#     """
#     https://github.com/daifeng2016/End-to-end-CD-for-VHR-satellite-image
#     """
#     # y_true = y_true[:, :, :, -1]  # y_true[:, :, :, :-1]=y_true[:, :, :, -1] if dim(3)=1 等效于[8,256,256,1]==>[8,256,256]
#     # y_pred = y_pred[:, :, :, -1]
#     intersection = K.sum(y_true * y_pred)
#     union = K.sum(y_true) + weight * K.sum(y_pred)
#     # K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
#     return ((2. * intersection + smooth) / (union + smooth))  # not working better using mean

# def dice_coef_loss(y_true, y_pred):
#     """
#     https://github.com/daifeng2016/End-to-end-CD-for-VHR-satellite-image
#     """
#     return 1 - dice_coef(y_true, y_pred)

def iou_loss(true, pred):
    """
    Calcaulate the intersection over union metric
    """
    intersection = true * pred

    notTrue = 1 - true
    union = true + (notTrue * pred)

    return tf.subtract(1.0, tf.reduce_sum(intersection)/tf.reduce_sum(union))

def conv_block(input_tensor, num_filters):
	encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
	encoder = layers.BatchNormalization()(encoder)
	encoder = layers.Activation('relu')(encoder)
	encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
	encoder = layers.BatchNormalization()(encoder)
	encoder = layers.Activation('relu')(encoder)
	return encoder

def encoder_block(input_tensor, num_filters):
	encoder = conv_block(input_tensor, num_filters)
	encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
	return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
	decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
	decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
	decoder = layers.BatchNormalization()(decoder)
	decoder = layers.Activation('relu')(decoder)
	decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
	decoder = layers.BatchNormalization()(decoder)
	decoder = layers.Activation('relu')(decoder)
	decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
	decoder = layers.BatchNormalization()(decoder)
	decoder = layers.Activation('relu')(decoder)
	return decoder

def binary_unet(input_tensor, bias = None):
    """
    Build a U-Net model
    Parameters:
        depth (int): number of training features (i.e. bands)
        optim (tf.keras.optimizer): keras optimizer
        loss (tf.keras.loss): keras or custom loss function
        mets (dict<tf.keras.metrics): dictionary of metrics for logits and classes. elements are lists of keras metrics
    Returns:
        tf.keras.model: compiled U-Net model
    """
    if bias is not None:
        bias = tf.keras.initializers.Constant(bias)
        
    # inputs = layers.Input(shape=[None, None, depth]) # 256
    encoder0_pool, encoder0 = encoder_block(input_tensor, 32) # 128
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64) # 64
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) # 32
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) # 16
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) # 8
    center = conv_block(encoder4_pool, 1024) # center
    decoder4 = decoder_block(center, encoder4, 512) # 16
    decoder3 = decoder_block(decoder4, encoder3, 256) # 32
    decoder2 = decoder_block(decoder3, encoder2, 128) # 64
    decoder1 = decoder_block(decoder2, encoder1, 64) # 128
    decoder0 = decoder_block(decoder1, encoder0, 32) # 256
    logits = layers.Conv2D(1, (1, 1), activation='sigmoid', bias_initializer = bias, name = 'logits')(decoder0)
    # logits is a probability and classes is binary. in solar, "tf.cast(tf.greater(x, 0.9)" was used  to avoid too many false positives
    classes = layers.Lambda(lambda x: tf.cast(tf.greater(x, 0.5), dtype = tf.int32), name = 'classes')(logits)
    # model = Model(inputs=[inputs], outputs=[logits, classes])

    # model.compile(
    #         optimizer=optim, 
    #         loss = {'logits': loss},
    #         #loss=losses.get(LOSS),
    #         metrics=mets)

    return [logits, classes]

def get_binary_model(depth, optim, loss, mets, bias = None):
    """
    Build a U-Net model
    Parameters:
        depth (int): number of training features (i.e. bands)
        optim (tf.keras.optimizer): keras optimizer
        loss (tf.keras.loss): keras or custom loss function
        mets (dict<tf.keras.metrics): dictionary of metrics for logits and classes. elements are lists of keras metrics
    Returns:
        tf.keras.model: compiled U-Net model
    """
    # if bias is not None:
    #     bias = tf.keras.initializers.Constant(bias)
        
    # inputs = layers.Input(shape=[None, None, depth]) # 256
    # encoder0_pool, encoder0 = encoder_block(inputs, 32) # 128
    # encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64) # 64
    # encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) # 32
    # encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) # 16
    # encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) # 8
    # center = conv_block(encoder4_pool, 1024) # center
    # decoder4 = decoder_block(center, encoder4, 512) # 16
    # decoder3 = decoder_block(decoder4, encoder3, 256) # 32
    # decoder2 = decoder_block(decoder3, encoder2, 128) # 64
    # decoder1 = decoder_block(decoder2, encoder1, 64) # 128
    # decoder0 = decoder_block(decoder1, encoder0, 32) # 256
    # logits = layers.Conv2D(1, (1, 1), activation='sigmoid', bias_initializer = bias, name = 'logits')(decoder0)
    # # logits is a probability and classes is binary. in solar, "tf.cast(tf.greater(x, 0.9)" was used  to avoid too many false positives
    # classes = layers.Lambda(lambda x: tf.cast(tf.greater(x, 0.5), dtype = tf.int32), name = 'classes')(logits)
    # model = models.Model(inputs=[inputs], outputs=[logits, classes])
    unet = binary_unet(inputs, bias)
    model = models.Model(inputs = inputs, outputs = unet)
    model.compile(
            optimizer=optim, 
            loss = {'logits': loss},
            #loss=losses.get(LOSS),
            metrics=mets)

    return model

def get_multiclass_model(depth, nclasses, optim, loss, mets, bias = None):
    """
    Build a U-Net model
    Parameters:
        depth (int): number of training features (i.e. bands)
        nclasses (int): number of output classes
        optim (tf.keras.optimizer): keras optimizer
        loss (tf.keras.loss): keras or custom loss function
        mets (dict<tf.keras.metrics): dictionary of metrics for logits and classes. elements are lists of keras metrics
    Returns:
        tf.keras.model: compiled U-Net model
    """
    if bias is not None:
        bias = tf.keras.initializers.Constant(bias)
        
    inputs = layers.Input(shape=[None, None, depth]) # 256
    encoder0_pool, encoder0 = encoder_block(inputs, 32) # 128
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64) # 64
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) # 32
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) # 16
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) # 8
    center = conv_block(encoder4_pool, 1024) # center
    decoder4 = decoder_block(center, encoder4, 512) # 16
    decoder3 = decoder_block(decoder4, encoder3, 256) # 32
    decoder2 = decoder_block(decoder3, encoder2, 128) # 64
    decoder1 = decoder_block(decoder2, encoder1, 64) # 128
    decoder0 = decoder_block(decoder1, encoder0, 32) # 256
    outputs = layers.Conv2D(nclasses, (1,1), activation = 'softmax', name = 'softmax')(decoder0)
    # logits = layers.Conv2D(1, (1, 1), activation='sigmoid', bias_initializer = bias, name = 'logits')(decoder0)
    # logits is a probability and classes is binary. in solar, "tf.cast(tf.greater(x, 0.9)" was used  to avoid too many false positives
    # classes = layers.Lambda(lambda x: tf.cast(tf.greater(x, 0.5), dtype = tf.int32), name = 'classes')(logits)
    # model = models.Model(inputs=[inputs], outputs=[logits, classes])
    model = models.Model(inputs = [inputs], outputs = [outputs])

    model.compile(
            optimizer=optim, 
            loss = {'softmax': loss},
            #loss=losses.get(LOSS),
            metrics=mets)

    return model

def get_autoencoder(depth, optim, loss, mets):
    """
    Build a U-Net model
    Parameters:
        depth (int): number of training features (i.e. bands)
        optim (tf.keras.optimizer): keras optimizer
        loss (tf.keras.loss): keras or custom loss function
        mets (dict<tf.keras.metrics): dictionary of metrics for logits and classes. elements are lists of keras metrics
    Returns:
        tf.keras.model: compiled U-Net model
    """
        
    inputs = layers.Input(shape=[None, None, depth]) # 256
    encoder0_pool, encoder0 = encoder_block(inputs, 32) # 128
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64) # 64
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) # 32
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) # 16
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) # 8
    center = conv_block(encoder4_pool, 1024) # center
    decoder4 = decoder_block(center, encoder4, 512) # 16
    decoder3 = decoder_block(decoder4, encoder3, 256) # 32
    decoder2 = decoder_block(decoder3, encoder2, 128) # 64
    decoder1 = decoder_block(decoder2, encoder1, 64) # 128
    decoder0 = decoder_block(decoder1, encoder0, 32) # 256
    preds = layers.Conv2D(1, (1, 1), name = 'continuous')(decoder0)
    # logits is a probability and classes is binary. in solar, "tf.cast(tf.greater(x, 0.9)" was used  to avoid too many false positives
    # classes = layers.Lambda(lambda x: tf.cast(tf.greater(x, 0.5), dtype = tf.int32), name = 'classes')(logits)
    model = models.Model(inputs=[inputs], outputs=[preds])

    model.compile(
            optimizer=optim, 
            loss = {'continuous': loss},
            #loss=losses.get(LOSS),
            metrics=mets)

    return model

def siamese_path(depth):
  # specify the inputs for the feature extractor network
  # define the first set of CONV => RELU => POOL => DROPOUT layers
  inputs = Input(shape = (None, None, depth))
  conv1 = Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
  conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv1)
  conv3 = Conv2D(64, (3, 3), padding ="same", activation="relu")(conv2)
  concat = Concatenate(axis = -1)([conv1, conv2, conv3])
  # build the model - returns a list of arrays
  model = Model(inputs = inputs, outputs = concat)
  # return the model to the calling function
  return model

def get_siamese_unet(depth, optim, loss, mets, bias = None):
  midpoint = depth//2
  input = Input((None, None, depth))
  temporal = tf.stack([input[:, :, :, :midpoint], input[:, :, :, midpoint:]], axis = 1)
  print(temporal.shape)
  branch = siamese_path(midpoint)
  bitemporal = TimeDistributed(branch)(temporal)
  difference = tf.subtract(bitemporal[:,0,:,:,:], bitemporal[:,1,:,:,:])
  unet_depth = difference.shape[-1]
  unet = binary_unet(difference, bias)
  model = Model(inputs = input, outputs = unet)
  model.compile(
        optimizer=optim, 
        loss = {'logits': loss},
        #loss=losses.get(LOSS),
        metrics=mets)
  return model  

def make_confusion_matrix_data(tpl, model, multiclass = False):
  """Create data needed to construct a confusion matrix on model predictions
  Functions takes a tfrecord dataset consisting of input features and lables
  and returns label and prediction vectors

  Parameters:
    dataset (tpl): features, labels tuple from tfDataset
    model (keras Model): model used to make predictions
    multiclass (bool): are labels multiclass or binary?

  Returns:
    tuple: 1D label and prediction arrays from the input datset
  """
  predicted = model.predict(tpl[0], verbose = 1)
  print(len(predicted))
      # some models will outputs probs and classes as a list
  print(type(predicted))
  if type(predicted) == list:
      print(predicted[0].shape)
      preds = predicted[0]
      # in this case, concatenate list elments into a single 4d array along last dimension
    #   preds = np.concatenate(preds, axis = 3)
  else:
      print(predicted.shape)
      preds = predicted[0,:,:,:]
  labs = tpl[1]

  if multiclass:
    labels = np.argmax(labs, axis = -1).flatten()
    predictions = np.argmax(preds, axis = -1).flatten()
    
  else:
    predictions = np.squeeze(np.greater(preds, 0.5)).flatten()
    labels = np.squeeze(labs).flatten()
    
  return labels, predictions

def make_confusion_matrix(dataset, model, multiclass = False):
  data = dataset.unbatch().batch(1) # batch data
  iterator = iter(data) # create a vector to iterate over so that you can call for loop on this object
  i = 0
  m = model

  # while the iterator still has unread batches of data...
  while True:
    # try to create a dataset from the current batch
    try:
      tpl = next(iterator)
    # if the iterator is out of data, break loop
    except StopIteration:
      break
    # else with no error...
    else:
      # make our confusion matrix data for current batch
      labels, preds = make_confusion_matrix_data(tpl, m, multiclass)

      # create confusion matrix containing raw counts from current data
      nclasses = tpl[1].shape[-1]
      con_mat_current = tf.math.confusion_matrix(labels = labels, predictions = preds, num_classes = nclasses).numpy()
      # get row sums
      rowsums_current = con_mat_current.sum(axis = 1)
      # if we're on the first batch
      if i == 0:
        con_mat = con_mat_current
      else:
        con_mat = con_mat + con_mat_current
      i += 1
  print(i)
  return con_mat

def retrain_model(model_file, checkpoint, eval_data, metric, weights_file = None, custom_objects = None, lr = None, freeze=False):
    """
    Load a previously trained model and continue training
    Parameters:
        model_file (str): path to model .h5 file
        lr (float): initial learning rate
        eval_data (tf.Dataset): data on which to calculate starting metrics
        metric (str): metric name for checkpoint logging
        weights_file (str): path to .hdf5 model weights file
    Return:
        keras.Model: 
    """
    
    # def get_weighted_bce(y_true, y_pred):
    #     return weighted_bce(y_true, y_pred, weight)
    
    # def get_gen_dice(y_true, y_pred):
    #     return gen_dice(y_true, y_pred, global_weights = weight)
    
    if custom_objects:
        # custom_objects = {'get_weighted_bce': get_weighted_bce}
        custom_objects = custom_objects
    else:
        custom_objects = {}
        
    # load our previously trained model and weights    
    m = models.load_model(model_file, custom_objects = custom_objects)
    if weights_file:
        m.load_weights(weights_file)
    # set the initial evaluation metric for saving checkpoints to the previous best value
    evalMetrics = m.evaluate(x = eval_data, verbose = 1)
    metrics = m.metrics_names
    print(metrics)
    index = metrics.index(metric)
    checkpoint.best = evalMetrics[index]
    # set the learning rate for re-training
    if not lr:
      lr = backend.eval(m.optimizer.learning_rate)
    backend.set_value(m.optimizer.learning_rate, lr)
    if freeze:
      for layer in m.layers[:-1]:
        layer.trainable = False
    return m, checkpoint