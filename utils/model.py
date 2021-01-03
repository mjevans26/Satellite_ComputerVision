# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:53:59 2020

@author: MEvans
"""

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import metrics
from tensorflow.python.keras import optimizers

def weighted_bce(weight = 0.8):
    """
    Compute the weighted binary cross entropy between predictions and observations
    Parameters:
        y_true (): nD tensor of labels
        y_pred (): nD tensor of probabilities
        weight (float): weighting factor for positive predictions. weight < 1 = reduce false positives
        
    Returns:
        nD tensor of same shape as y_pred
    """
    def bce_loss(y_true, y_pred):
        bce = tf.nn.weighted_cross_entropy_with_logits(labels = y_true, logits = y_pred, pos_weight = weight)
        return tf.reduce_mean(bce)
    
    return bce_loss

def dice_coef(y_true, y_pred, smooth=1, weight=0.5):
    """
    https://github.com/daifeng2016/End-to-end-CD-for-VHR-satellite-image
    """
    # y_true = y_true[:, :, :, -1]  # y_true[:, :, :, :-1]=y_true[:, :, :, -1] if dim(3)=1 等效于[8,256,256,1]==>[8,256,256]
    # y_pred = y_pred[:, :, :, -1]
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + weight * K.sum(y_pred)
    # K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    return ((2. * intersection + smooth) / (union + smooth))  # not working better using mean

def dice_coef_loss(y_true, y_pred):
    """
    https://github.com/daifeng2016/End-to-end-CD-for-VHR-satellite-image
    """
    return 1 - dice_coef(y_true, y_pred)

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

def get_model(depth, optim, loss, mets):
    """
    Build a U-Net model
    Parameters:
        depth (int): number of training features (i.e. bands)
        optim (tf.keras.optimizer): keras optimizer
        loss (tf.keras.loss): keras or custom loss function
        mets (list<tf.keras.metrics): list of keras metrics
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
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

    model = models.Model(inputs=[inputs], outputs=[outputs])

    model.compile(
            optimizer=optim, 
            loss = loss,
            #loss=losses.get(LOSS),
            metrics=[metrics.get(metric) for metric in mets])

    return model
