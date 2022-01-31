# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:24:42 2020

@author: MEvans
"""
import os
from os.path import join
from sys import path
path.append(os.getcwd())
# import ee
import json
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
#import gsutil
import rasterio as rio
from utils.processing import normalize
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.transform import array_bounds 
     

# TODO: automate spliting of full GEE path
# def doExport(image, features, scale, bucket, pred_base, pred_path, region, kernel_shape = [256, 256], kernel_buffer = [128,128]):
#   """
#   Run an image export task on which to run predictions.  Block until complete.
#   Parameters:
#     image (ee.Image): image to be exported for prediction
#     features (list): list of band names to include in export
#     scale (int): pixel scale
#     bucket (str): name of GCS bucket to write files
#     pred_path (str): relative google cloud directory path for export
#     pred_base (str): base filename of exported image
#     kernel_shape (array<int>): size of image patch in pixels
#     kernel_buffer (array<int>): pixels to buffer the prediction patch. half added to each side
#     region (ee.Geometry):
#   """
#   task = ee.batch.Export.image.toCloudStorage(
#     image = image.select(features), 
#     description = pred_base, 
#     bucket = bucket, 
#     fileNamePrefix = join(pred_path, pred_base),
#     region = region,#.getInfo()['coordinates'], 
#     scale = scale, 
#     fileFormat = 'TFRecord', 
#     maxPixels = 1e13,
#     formatOptions = { 
#       'patchDimensions': kernel_shape,
#       'kernelSize': kernel_buffer,
#       'compressed': True,
#       'maxFileSize': 104857600
#     }
#   )
#   task.start()

#   # Block until the task completes.
#   print('Running image export to Cloud Storage...')
#   import time
#   while task.active():
#     time.sleep(30)

#   # Error condition
#   if task.status()['state'] != 'COMPLETED':
#     print('Error with image export.')
#   else:
#     print('Image export completed.')

#   # Error condition
#   if task.status()['state'] != 'COMPLETED':
#     print('Error with image export.')
#   else:
#     print('Image export completed.')
    
  
#def makePredDataset(bucket, pred_path, pred_image_base, kernel_buffer, features, raw = None):
def makePredDataset(file_list, features, kernel_shape = [256, 256], kernel_buffer = [128, 128], axes = [2], splits = None, moments = None, one_hot = None, **kwargs):
    """ Make a TFRecord Dataset that can be used for predictions
    Parameters:
        file_list: list of complete pathnames for prediction data files
        pred_path (str): path to .tfrecord files
        pred_image_base (str): pattern matching basename of file(s)
        kernel_shape (tpl): size of image patch in pixels
        kernel_buffer (tpl): pixels to trim from H, W dimensions of prediction
        features (list): names of features in incoming data
        axes (list): axes for normalization
        one_hot (dict): key:value pairs for name of one-hot variable and desired one-hot depth
    Return:
        TFRecord Dataset
    """
    
      # Make sure the files are in the right order.
    file_list.sort()
    
      # Get set up for prediction.
    x_buffer = int(kernel_buffer[0] / 2)
    y_buffer = int(kernel_buffer[1] / 2)
    
    buffered_shape = [
        kernel_shape[0] + kernel_buffer[0],
        kernel_shape[1] + kernel_buffer[1]]
    
    imageColumns = [
      tf.io.FixedLenFeature(shape=buffered_shape, dtype=tf.float32) 
        for k in features
    ]
    
    imageFeaturesDict = dict(zip(features, imageColumns))
    
    def parse_image(example_proto):
      return tf.io.parse_single_example(example_proto, imageFeaturesDict)
    
    def toTupleImage(dic):
        
        # stack the augmented bands, optional one-hot tensors, and response variable
        if one_hot:
            featList = [dic.get(key) for key in features if key not in one_hot.keys()]
            hotList = [tf.one_hot(tf.cast(dic.get(key), tf.uint8), val, axis = 2) for key, val in one_hot.items()]
        else:
            featList = [dic.get(key) for key in features]
        
        bands = tf.transpose(tf.stack(featList, axis = 0), [1,2,0])
        bands = normalize(bands, axes = axes, moments = moments, splits = splits)
            # If custom preprocessing functions are specified add respective bands

        for fxn in kwargs.values():
            der = fxn(dic)
            der = tf.expand_dims(der, axis = 2)
            bands = tf.concat([bands, der], axis = 2)
        
        if one_hot:
          hotStack = tf.concat(hotList, axis = 2)
          stacked = tf.concat([bands, hotStack], axis =2)
        else:
          stacked = tf.concat([bands], axis = 2)
        
        return stacked
  
  # Create a dataset(s) from the TFRecord file(s) in Cloud Storage.
    
    imageDataset = tf.data.TFRecordDataset(file_list, compression_type='GZIP')
    imageDataset = imageDataset.map(parse_image, num_parallel_calls=5)
    imageDataset = imageDataset.map(toTupleImage).batch(1)
    return imageDataset

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def callback_predictions(imageDataset, model, mixer, kernel_shape = [256, 256], kernel_buffer = [128, 128]):
    patches = mixer['totalPatches']
    cols = mixer['patchesPerRow']
    rows = patches//cols

    # Perform inference.
    predictions = model.predict(imageDataset, steps=patches, verbose=1)
    
    # some models will outputs probs and classes as a list
    if type(predictions) == list:
        # in this case, concatenate list elments into a single 4d array along last dimension
        predictions = np.concatenate(predictions, axis = -1)
        
    x_buffer = int(kernel_buffer[0] / 2)
    y_buffer = int(kernel_buffer[1] / 2)
    x_size = kernel_shape[0]+y_buffer
    y_size = kernel_shape[1]+x_buffer

    x = 1
    for prediction in predictions:
      print('Writing patch ' + str(x) + '...')
      # lets just write probabilities, classes can be calculated post processing if not present already
      patch = prediction[y_buffer:y_size, x_buffer:x_size, :]
#      predPatch = np.add(np.argmax(prediction, axis = 2), 1)
#      probPatch = np.max(prediction, axis = 2)
#      predPatch = predPatch[x_buffer:x_buffer+KERNEL_SIZE, y_buffer:y_buffer+KERNEL_SIZE]
#      probPatch = probPatch[x_buffer:x_buffer+KERNEL_SIZE, y_buffer:y_buffer+KERNEL_SIZE]
#      # stack probabilities and classes along channel dimension
#      patch = np.stack([predPatch, probPatch], axis = 2)

      ## NOTE: Predictions come out with y as 0 dimension (ie. rows), x as 1 dimension (ie. columns)
      # if we're at the beginning of a row
      if x%cols == 1:
        row = patch
      else:
        row = np.append(row, patch, axis = 1)
      # if we reached the end of a row start a new one
      if x%cols == 0:
        # for the first row, create single row rows object
        if x <= cols:
          rows = row
        else:
        # add current row to previous rows along y axis
          rows = np.append(rows, row, axis = 0)
      x += 1

    return rows  

def make_array_predictions(imageDataset, model, jsonFile, kernel_shape = [256, 256], kernel_buffer = [128,128]):
    """Create a 3D array of prediction outputs from TFRecord dataset
    
    Given a set of TFRecords representing image patches on which to run model predictions,
    and a json file specifying the spatial reference system and arrangement of patches,
    this function writes predictions to a single, reconstructed numpy array of shape
    (?,?,2). Dimension 2 holds class probabilities and most likely class.
    
    Parameters:
        imageDataset (tf.Dataset): image patch tensors on which to run predictions
        model (keras Model): model used to make predictions
        jsonFile (str): complete GCS filepath to json file
        kernel_size(tpl): size of image patch in pixels
        kernel_buffer (tpl): pixels to trim from H, W, dimensions of each output patch
    Return:
        ndarray: 3D array of prediction outputs.
    """
    # we need metadata from the json file to reconstruct prediction patches
    # Load the contents of the mixer file to a JSON object.
#    jsonFile = '/'.join(jsonFile.split(sep = '/')[3:])
#    blob = bucket.get_blob(jsonFile) #23Mar21 update to use google-cloud-storage library
#    jsonText = blob.download_as_string().decode('utf-8')
#    mixer = json.loads(jsonText)
    
    with open(jsonFile,) as file:
        mixer = json.load(file)
    
#    # Load the contents of the mixer file to a JSON object.
#    jsonText = !gsutil cat {jsonFile}
#    
#    # Get a single string w/ newlines from the IPython.utils.text.SList
#    mixer = json.loads(jsonText.nlstr)
    
    print(mixer)
    patches = mixer['totalPatches']
    cols = mixer['patchesPerRow']
    rows = patches//cols

    # Perform inference.
    print('Running predictions...')
    predictions = model.predict(imageDataset, steps=patches, verbose=1)
    
    # some models will outputs probs and classes as a list
    if type(predictions) == list:
        # in this case, concatenate list elments into a single 4d array along last dimension
        predictions = np.concatenate(predictions, axis = 3)
        
    x_buffer = int(kernel_buffer[0] / 2)
    y_buffer = int(kernel_buffer[1] / 2)
    x_size = kernel_shape[0]+y_buffer
    y_size = kernel_shape[1]+x_buffer

    x = 1
    for prediction in predictions:
      print('Writing patch ' + str(x) + '...')
      # lets just write probabilities, classes can be calculated post processing if not present already
      patch = prediction[y_buffer:y_size, x_buffer:x_size, :]
#      predPatch = np.add(np.argmax(prediction, axis = 2), 1)
#      probPatch = np.max(prediction, axis = 2)
#      predPatch = predPatch[x_buffer:x_buffer+KERNEL_SIZE, y_buffer:y_buffer+KERNEL_SIZE]
#      probPatch = probPatch[x_buffer:x_buffer+KERNEL_SIZE, y_buffer:y_buffer+KERNEL_SIZE]
#      # stack probabilities and classes along channel dimension
#      patch = np.stack([predPatch, probPatch], axis = 2)

      ## NOTE: Predictions come out with y as 0 dimension (ie. rows), x as 1 dimension (ie. columns)
      # if we're at the beginning of a row
      if x%cols == 1:
        row = patch
      else:
        row = np.append(row, patch, axis = 1)
      # if we reached the end of a row start a new one
      if x%cols == 0:
        # for the first row, create single row rows object
        if x <= cols:
          rows = row
        else:
        # add current row to previous rows along y axis
          rows = np.append(rows, row, axis = 0)
      x += 1

    return rows

def write_tfrecord_predictions(predictions, pred_path, out_image_base, kernel_shape = [256, 256], kernel_buffer = [128,128]):
    """Generate predictions and save as TFRecords to Cloud Storage
    Parameters:
      imageDataset (tf.Dataset): data on which to run predictions
      pred_path (str): full path to output directory 
      out_image_base (str): file basename for input and output files
      kernel_shape (tpl): [y, x] size of image patch in pixels
      kernel_buffer (tpl): [y, x] size of buffer to be trimmed from predictions

    Return:
      empty: Writes TFRecord files to specified destination
    """
    # Perform inference.
    # print('Running predictions...')
    # predictions = model.predict(imageDataset, steps=None, verbose=1)
    # print(predictions[0])
    
    # some models will outputs probs and classes as a list
    if type(predictions) == list:
        # in this case, concatenate list elments into a single 4d array along last dimension
        predictions = np.concatenate(predictions, axis = 3)
    
    # get the number of bands (should usually be one or two)
    C = predictions.shape[-1]
    
    out_image_file = join(pred_path, f'{out_image_base}.tfrecords')
    
    print('Writing predictions to ' + out_image_file + '...')
    writer = tf.io.TFRecordWriter(out_image_file)

    patches = 1

    x_buffer = int(kernel_buffer[0] / 2)
    y_buffer = int(kernel_buffer[1] / 2)
    x_size = x_buffer + kernel_shape[1]
    y_size = y_buffer + kernel_shape[0]

    for prediction in predictions:
      print('Writing patch ' + str(patches) + '...')
      # lets just write probabilities, classes can be calculated post processing if not present already
      patch = prediction[y_buffer:y_size, x_buffer:x_size, :]
#      predPatch = np.add(np.argmax(prediction, axis = 2), 1)
#      probPatch = np.max(prediction, axis = 2)
#      predPatch = predPatch[x_buffer:x_buffer+KERNEL_SIZE, y_buffer:y_buffer+KERNEL_SIZE]
#      probPatch = probPatch[x_buffer:x_buffer+KERNEL_SIZE, y_buffer:y_buffer+KERNEL_SIZE]
      
      # for each band in prediction, create a tf train feature
      feature = {}
      for i in range(C):
          feat = tf.train.Feature(float_list = tf.train.FloatList(value = np.ndarray.flatten(patch[:,:,i])))
          feature['b{}'.format(i+1)] = feat
          
      # Create an example.
      example = tf.train.Example(
        features=tf.train.Features(
                feature = feature
#          feature={
#            'class': tf.train.Feature(
#                int64_list=tf.train.Int64List(
#                    value = np.ndarray.flatten(predPatch))),
#            'prob': tf.train.Feature(
#                float_list = tf.train.FloatList(
#                    value = np.ndarray.flatten(probPatch)))
#          }
        )
      )
      # Write the example.
      writer.write(example.SerializeToString())
      patches += 1

    writer.close()

def write_geotiff_prediction(image, jsonFile, aoi):
  with open(jsonFile,) as file:
    mixer = json.load(file)
          
    transform = mixer['projection']['affine']['doubleMatrix']
    crs = mixer['projection']['crs']
    ppr = mixer['patchesPerRow']
    tp = mixer['totalPatches']
    rows = int(tp/ppr)

  if image.ndim < 3:
      image = np.expand_dims(image, axis = -1)
      
  affine = rio.Affine(transform[0], transform[1], transform[2], transform[3], transform[4], transform[5])

  with rio.open(
      f'{aoi}.tif',
      'w',
      driver = 'GTiff',
      width = image.shape[1],
      height = image.shape[0],
      count = image.shape[2],
      dtype = image.dtype,
      crs = crs,
      transform = affine) as dst:
      dst.write(np.transpose(image, (2,0,1)))
      
# TODO: re-calculate n and write files not strictly based on rows
def write_geotiff_predictions(predictions, mixer, outImgBase, outImgPath, kernel_buffer = [128,128]):
  """Run predictions on a TFRecord dataset and save as a GeoTIFF
  Parameters:
    imageDataset (tf.Dataset): data on which to run predictions
    jsonFile (str): filename of json mixer file
    outImgPath (str): directory in which to write predictions
    outImgBase (str): file basename
    kernel_buffer (tpl): x and y padding around patches
  Return:
    empty: writes geotiff records temporarily to working directory
  """
  # with open(jsonFile, ) as file:
  #   mixer = json.load(file)
  transform = mixer['projection']['affine']['doubleMatrix']
  crs = mixer['projection']['crs']
  ppr = mixer['patchesPerRow']
  tp = mixer['totalPatches']
  rows = int(tp/ppr)
  kernel_shape = mixer['patchDimensions']

  H = rows*kernel_shape[0]
  W = ppr*kernel_shape[1]
  y_indices = list(range(0, H, kernel_shape[0]))
  x_indices = list(range(0, W, kernel_shape[1]))
  indices = [(y,x) for y in y_indices for x in x_indices]
  out_array = np.zeros((H, W, 1), dtype = np.float32)
  print('out array', out_array.shape)
  x_buffer = int(kernel_buffer[0]/2)
  y_buffer = int(kernel_buffer[1]/2)
  x_size = x_buffer + kernel_shape[1]
  y_size = y_buffer + kernel_shape[0]

  # prediction = model.predict(imageDataset, steps = tp, verbose = 1)
  if type(predictions) == list:
    predictions = np.concatenate(predictions, axis = 3)
    
  for i, (y,x) in enumerate (indices):
    prediction = predictions[i]
    print('prediction', prediction.shape)
    out_array[y:y+kernel_shape[0], x:x+kernel_shape[1],0] += prediction[y_buffer:y_size, x_buffer:x_size, 0]
      
  affine = rio.Affine(transform[0], transform[1], transform[2], transform[3], transform[4], transform[5])

  out_image_file = join(outImgPath, f'{outImgBase}.tif')
  print(f'writing image to {out_image_file}')
  with rio.open(
    out_image_file,
    'w',
    driver = 'GTiff',
    width = W,
    height = H,
    count = 1,
    dtype = out_array.dtype,
    crs = crs,
    transform = affine) as dst:
    dst.write(np.transpose(out_array, (2,0,1)))

#def ingest_predictions(pred_path, out_image_base, user_folder):
#  """
#  Upload prediction image(s) to Earth Engine.
#  Parameters:
#    pred_path (str): Google cloud (or Drive) path storing prediction image files
#    pred_image_base (str):
#    user_folder (str): GEE directory to store asset
#    out_image_base (str): base filename for GEE asset
#  """
#  blob = bucket.get_blob(join(pred_path, out_image_base + '_mixer.json'))
#  jsonFile = blob.name
#  
##  jsonFile = !gsutil ls {join('gs://', pred_path, out_image_base + '*.json')}
#  print(jsonFile)
#  blobs = bucket.list_blobs(join(pred_path, 'outputs', out_image_base + ))
#  predFiles = !gsutil ls {join('gs://', pred_path, 'outputs', out_image_base + '*TFRecord')}
#  print(predFiles)
#  out_image_files = ' '.join(predFiles)
#  # Start the upload.
#  out_image_asset = join(user_folder, out_image_base)
#  !earthengine upload image --asset_id={out_image_asset} {out_image_files} {jsonFile[0]}
  
def get_img_bounds(img, jsonFile, dst_crs = None):
      """Get the projected top left and bottom right coordinates of an image
      Parameters:
      img (ndarray): image to generate bounding coordinates for
      jsonFile (str): path to json file defining crs and image size
      dst_crs (str): epsg code for output crs
      Return:
      tpl: [[lat min, lon min],[lat max, lon max]]
      """
      # Get a single string w/ newlines from the IPython.utils.text.SList
      with open(jsonFile,) as f:
        mixer = json.load(f)
      # mixer = json.loads(jsonText.nlstr)
      transform = mixer['projection']['affine']['doubleMatrix']
      print(transform)
      src_crs = CRS.from_string(mixer['projection']['crs'])
      print(src_crs)
      affine = rio.Affine(transform[0], transform[1], transform[2], transform[3], transform[4], transform[5])
      H,W = [0,0]
    
      if type(img) == np.ndarray:
          print('input image is numpy')
          H,W = img.shape
          print('image shape is ', H, W)
          bounds = array_bounds(H, W, affine)
    
      elif type(img) == str:
          print('input image is geotiff')
          with rio.open(img) as src:
              bounds = src.bounds
      # H, W = src.shape
    
      print(bounds)
      lon_min, lat_min, lon_max, lat_max = bounds
      # if we need to transform the bounds, such as for folium ('EPSG:3857')
      if dst_crs:
          dst_crs = CRS.from_string(dst_crs)
          out_bounds = transform_bounds(src_crs, dst_crs, left = lon_min, bottom = lat_min, right = lon_max, top = lat_max, densify_pts=21)
          lon_min, lat_min, lon_max, lat_max = out_bounds
          print(out_bounds)
      return [[lat_min, lon_min], [lat_max, lon_max]]

def doPrediction(bucket, pred_path, pred_image_base, features, one_hot, out_image_base, kernel_shape, kernel_buffer):
  """
  Given a bucket and path to prediction images, create a prediction dataset, make predictions
  and write tfrecords to GCS
  Parameters:
    bucket: (Bucket): google-cloud-storage bucket object
    pred_path (str): relative GCS path storing prediction image files
    pred_image_base (str): base filename of prediction files
    user_folder (str): GEE directory to store asset
    out_image_base (str): base filename for GEE asset
    kernel_buffer (Array<int>): length 2 array 
  Return:
    list: list of written image filenames to be used in earthengine upload
  """

  print('Looking for TFRecord files...')
  
  # Get a list of all the files in the output bucket.
  blobs = bucket.list_blobs(prefix = join(pred_path, pred_image_base))
  filesList = [file.name for file in blobs if pred_image_base in file.name]
#  filesList = !gsutil ls {pred_path}
  # Get only the files generated by the image export.
#  exportFilesList = [s for s in filesList if pred_image_base in s]

  # Get the list of image files and the JSON mixer file.
  imageFilesList = []
  jsonFile = None
  for f in filesList:
    if f.endswith('.tfrecord.gz'):
      imageFilesList.append(f)
    elif f.endswith('.json'):
      jsonFile = f

  # Make sure the files are in the right order.
  imageFilesList.sort()

  from pprint import pprint
  pprint('image files:', imageFilesList)
  print('json file:', jsonFile)
  
  # make a prediction dataset from the given files
  
  # Load the contents of the mixer file to a JSON object.
  blob = bucket.get_blob(jsonFile)
  jsonText = blob.download_as_string().decode('utf-8')
  mixer = json.loads(jsonText)
#  jsonText = !gsutil cat {jsonFile}
  # Get a single string w/ newlines from the IPython.utils.text.SList
#  mixer = json.loads(jsonText.nlstr)
  pprint(mixer)
  patches = mixer['totalPatches']
  
#  # Get set up for prediction.
#  x_buffer = int(kernel_buffer[0] / 2)
#  y_buffer = int(kernel_buffer[1] / 2)
#
#  buffered_shape = [
#      KERNEL_SHAPE[0] + kernel_buffer[0],
#      KERNEL_SHAPE[1] + kernel_buffer[1]]
#
#  imageColumns = [
#    tf.io.FixedLenFeature(shape=buffered_shape, dtype=tf.float32) 
#      for k in BANDS
#  ]
#
#  imageFeaturesDict = dict(zip(BANDS, imageColumns))
#
#  def parse_image(example_proto):
#    return tf.io.parse_single_example(example_proto, imageFeaturesDict)
#
#  def toTupleImage(dic):
#    inputsList = [dic.get(key) for key in BANDS]
#    stacked = tf.stack(inputsList, axis=0)
#    stacked = tf.transpose(stacked, [1, 2, 0])
#    stacked = normalize(stacked, [0, 1])
#    return stacked
  
  # Create a dataset(s) from the TFRecord file(s) in Cloud Storage.
  i = 0
  patches = 0
  written_files = []
  while i < len(imageFilesList):
    imageDataset = makePredDataset(file_list = imageFilesList[i:i+100], kernel_shape = kernel_shape, kernel_buffer = kernel_buffer, features = features, one_hot = one_hot)
#    imageDataset = tf.data.TFRecordDataset(imageFilesList[i:i+100], compression_type='GZIP')
#    imageDataset = imageDataset.map(parse_image, num_parallel_calls=5)
#    imageDataset = imageDataset.map(toTupleImage).batch(1)
    
    out_image_base = out_image_base + '{:04d}'.format(i)
    out_image_file = join('gs://', bucket.name, pred_path, 'outputs/tfrecord', out_image_base + '.TFRecord')
    write_tfrecord_predictions(imageDataset, pred_path = pred_path, out_image_base = out_image_base, kernel_buffer = kernel_buffer)
#    # Perform inference.
#    print('Running predictions...')
#    predictions = m.predict(imageDataset, steps=None, verbose=1)
#    # print(predictions[0])
#
#
#    
#    print('Writing predictions to ' + out_image_file + '...')
#    writer = tf.io.TFRecordWriter(out_image_file)
#    for predictionPatch in predictions:
#      print('Writing patch ' + str(patches) + '...')
#      predictionPatch = predictionPatch[
#          x_buffer:x_buffer+KERNEL_SIZE, y_buffer:y_buffer+KERNEL_SIZE]
#
#      # Create an example.
#      example = tf.train.Example(
#        features=tf.train.Features(
#          feature={
#            'probability': tf.train.Feature(
#                float_list=tf.train.FloatList(
#                    value=predictionPatch.flatten()))
#          }
#        )
#      )
#      # Write the example.
#      writer.write(example.SerializeToString())
#      patches += 1
#
#    writer.close()
    i += 100
    written_files.append(out_image_file)
 
  out_image_files = ' '.join(written_files)
  # Start the upload.
#  out_image_asset = join(user_folder, out_image_base)
#  !earthengine upload image --asset_id={out_image_asset} {out_image_files} {jsonFile}
  # return list of written image files for use in gee upload
  return out_image_files