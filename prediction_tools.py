# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:24:42 2020

@author: MEvans
"""
from os.path import join
import ee
import json
#import gsutil
import rasterio as rio

# TODO: automate spliting of full GEE path
def doExport(image, bucket, path, out_image_base, kernel_shape, kernel_buffer, region):
  """
  Export an image from GEE as TFRecords for prediction.  Block until complete.
  Parameters:
    image (ee.Image): image to be exported for prediction
    path (str): google cloud directory path for export
    out_image_base (str): base filename of exported image
    kernel_buffer (array<int>): pixels to buffer the prediction patch. half added to each side
    region (ee.Geometry): geometry containing the image
  """
  task = ee.batch.Export.image.toCloudStorage(
    image = image, 
    description = out_image_base, 
    bucket = bucket, 
    fileNamePrefix = join(path, out_image_base),
    region = region,#.getInfo()['coordinates'], 
    scale = 10, 
    fileFormat = 'TFRecord', 
    maxPixels = 1e13,
    formatOptions = { 
      'patchDimensions': kernel_shape,
      'kernelSize': kernel_buffer,
      'compressed': True,
      'maxFileSize': 104857600
    }
  )
  task.start()

  # Block until the task completes.
  print('Running image export to Cloud Storage...')
  import time
  while task.active():
    time.sleep(30)

  # Error condition
  if task.status()['state'] != 'COMPLETED':
    print('Error with image export.')
  else:
    print('Image export completed.')
    
def doPrediction(pred_path, pred_image_base, user_folder, out_image_base, kernel_buffer, region):
  """
  Perform inference on exported imagery, upload to Earth Engine.
  Parameters:
    pred_path (str): Full Google cloud (or Drive) path storing prediction image files
    pred_image_base (str):
    user_folder (str): GEE directory to store asset
    out_image_base (str): base filename for GEE asset
    kernel_buffer (Array<int>): length 2 array 
    region (ee.Geometry)):
  """

  print('Looking for TFRecord files...')
  
  # Get a list of all the files in the output bucket.
  filesList = !gsutil ls {pred_path}
  # Get only the files generated by the image export.
  exportFilesList = [s for s in filesList if pred_image_base in s]

  # Get the list of image files and the JSON mixer file.
  imageFilesList = []
  jsonFile = None
  for f in exportFilesList:
    if f.endswith('.tfrecord.gz'):
      imageFilesList.append(f)
    elif f.endswith('.json'):
      jsonFile = f

  # Make sure the files are in the right order.
  imageFilesList.sort()

  from pprint import pprint
  pprint(imageFilesList)
  print(jsonFile)
  
  import json
  # Load the contents of the mixer file to a JSON object.
  jsonText = !gsutil cat {jsonFile}
  # Get a single string w/ newlines from the IPython.utils.text.SList
  mixer = json.loads(jsonText.nlstr)
  pprint(mixer)
  patches = mixer['totalPatches']
  
  # Get set up for prediction.
  x_buffer = int(kernel_buffer[0] / 2)
  y_buffer = int(kernel_buffer[1] / 2)

  buffered_shape = [
      KERNEL_SHAPE[0] + kernel_buffer[0],
      KERNEL_SHAPE[1] + kernel_buffer[1]]

  imageColumns = [
    tf.io.FixedLenFeature(shape=buffered_shape, dtype=tf.float32) 
      for k in BANDS
  ]

  imageFeaturesDict = dict(zip(BANDS, imageColumns))

  def parse_image(example_proto):
    return tf.io.parse_single_example(example_proto, imageFeaturesDict)

  def toTupleImage(dic):
    inputsList = [dic.get(key) for key in BANDS]
    stacked = tf.stack(inputsList, axis=0)
    stacked = tf.transpose(stacked, [1, 2, 0])
    stacked = normalize(stacked, [0, 1])
    return stacked
  
  # Create a dataset(s) from the TFRecord file(s) in Cloud Storage.
  i = 0
  patches = 0
  written_files = []
  while i < len(imageFilesList):

    imageDataset = tf.data.TFRecordDataset(imageFilesList[i:i+100], compression_type='GZIP')
    imageDataset = imageDataset.map(parse_image, num_parallel_calls=5)
    imageDataset = imageDataset.map(toTupleImage).batch(1)
    
    # Perform inference.
    print('Running predictions...')
    predictions = m.predict(imageDataset, steps=None, verbose=1)
    # print(predictions[0])

    out_image_file = join(pred_path,
                          'outputs',
                          '{}{:2d}.TFRecord'.format(out_image_base, i))
    
    print('Writing predictions to ' + out_image_file + '...')
    writer = tf.io.TFRecordWriter(out_image_file)
    for predictionPatch in predictions:
      print('Writing patch ' + str(patches) + '...')
      predictionPatch = predictionPatch[
          x_buffer:x_buffer+KERNEL_SIZE, y_buffer:y_buffer+KERNEL_SIZE]

      # Create an example.
      example = tf.train.Example(
        features=tf.train.Features(
          feature={
            'probability': tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=predictionPatch.flatten()))
          }
        )
      )
      # Write the example.
      writer.write(example.SerializeToString())
      patches += 1

    writer.close()
    i += 100
    written_files.append(out_image_file)
 
  out_image_files = ' '.join(written_files)
  # Start the upload.
  out_image_asset = join(user_folder, out_image_base)
  !earthengine upload image --asset_id={out_image_asset} {out_image_files} {jsonFile}
  
def makePredDataset(pred_path, pred_image_base, kernel_buffer):
    """ Make a TFRecord Dataset that can be used for predictions
    Parameters:
        pred_path (str): path to .tfrecord files
        pred_image_base (str): pattern matching basename of file(s)
        kernel_buffer (tpl): pixels to trim from H, W dimensions of prediction
    Return:
        TFRecord Dataset
    """
  print('Looking for TFRecord files...')
  
  # Get a list of all the files in the output bucket.
  filesList = !gsutil ls {pred_path}
  # Get a list of all the prediction files in GDrive
  # fileList = tf.io.gfile.glob(join('content/drive/My Drive', pred_path))
  # Get only the files generated by the image export.
  exportFilesList = [s for s in filesList if pred_image_base in s]

  # Get the list of image files and the JSON mixer file.
  imageFilesList = []
  jsonFile = None
  for f in exportFilesList:
    if f.endswith('.tfrecord.gz'):
      imageFilesList.append(f)
    elif f.endswith('.json'):
      jsonFile = f

  # Make sure the files are in the right order.
  imageFilesList.sort()

  x_buffer = int(kernel_buffer[0] / 2)
  y_buffer = int(kernel_buffer[1] / 2)

  buffered_shape = [
      KERNEL_SHAPE[0] + kernel_buffer[0],
      KERNEL_SHAPE[1] + kernel_buffer[1]]

  imageColumns = [
    tf.io.FixedLenFeature(shape=buffered_shape, dtype=tf.float32) 
      for k in BANDS
  ]

  imageFeaturesDict = dict(zip(BANDS, imageColumns))

  def parse_image(example_proto):
    return tf.io.parse_single_example(example_proto, imageFeaturesDict)

  def toTupleImage(dic):
    inputsList = [dic.get(key) for key in BANDS]
    stacked = tf.stack(inputsList, axis=0)
    stacked = tf.transpose(stacked, [1, 2, 0])
    stacked = normalize(stacked, [0, 1])
    return stacked
  
  # Create a dataset(s) from the TFRecord file(s) in Cloud Storage.

  imageDataset = tf.data.TFRecordDataset(imageFilesList, compression_type='GZIP')
  imageDataset = imageDataset.map(parse_image, num_parallel_calls=5)
  imageDataset = imageDataset.map(toTupleImage).batch(1)
  return imageDataset

def make_array_predictions(imageDataset, jsonFile, kernel_buffer):
    """Create a 3D array of prediction outputs from TFRecord dataset
    
    Given a set of TFRecords representing image patches on which to run model predictions,
    and a json file specifying the spatial reference system and arrangement of patches,
    this function writes predictions to a single, reconstructed numpy array of shape
    (?,?,2). Dimension 2 holds class probabilities and most likely class.
    
    Parameters:
        imageDataset (tf.Dataset): image patch tensors on which to run predictions
        jsonFile (str): complete filepath to json file
        kernel_buffer (tpl): pixels to trim from H, W, dimensions of each output patch
    Return:
        ndarray: 3D array of prediction outputs.
    """
    # we need metadata from the json file to reconstruct prediction patches

    # Load the contents of the mixer file to a JSON object.
    jsonText = !gsutil cat {jsonFile}
    # Get a single string w/ newlines from the IPython.utils.text.SList
    mixer = json.loads(jsonText.nlstr)
    print(mixer)
    patches = mixer['totalPatches']
    cols = mixer['patchesPerRow']
    rows = patches//cols

    # Perform inference.
    print('Running predictions...')
    predictions = m.predict(imageDataset, steps=patches, verbose=1)

    x_buffer = int(kernel_buffer[0] / 2)
    y_buffer = int(kernel_buffer[1] / 2)

    x = 1
    for prediction in predictions:
      print('Writing patch ' + str(x) + '...')
      predPatch = np.add(np.argmax(prediction, axis = 2), 1)
      probPatch = np.max(prediction, axis = 2)
      predPatch = predPatch[x_buffer:x_buffer+KERNEL_SIZE, y_buffer:y_buffer+KERNEL_SIZE]
      probPatch = probPatch[x_buffer:x_buffer+KERNEL_SIZE, y_buffer:y_buffer+KERNEL_SIZE]
      # stack probabilities and classes along channel dimension
      patch = np.stack([predPatch, probPatch], axis = 2)

      ## NOTE: Predictions come out with y as 0 dimension, x as 1 dimension
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

def write_tfrecord_predictions(imageDataset, pred_path, out_image_base, kernel_buffer):
    """Generate predictions and save as TFRecords to Cloud Storage
    Parameters:
      imageDataset (tf.Dataset): data on which to run predictions
      pred_path (str): full path to output directory 
      out_image_base (str): file basename for input and output files
      kernel_buffer (tpl): [x, y] size of buffer to be trimmed from predictions

    Return:
      empty: Writes TFRecord files to specified destination
    """
    # Perform inference.
    print('Running predictions...')
    predictions = m.predict(imageDataset, steps=None, verbose=1)
    # print(predictions[0])

    out_image_file = join(pred_path,
                          'outputs',
                          '{}.TFRecord'.format(out_image_base))
    
    print('Writing predictions to ' + out_image_file + '...')
    writer = tf.io.TFRecordWriter(out_image_file)

    patches = 1

    x_buffer = int(kernel_buffer[0] / 2)
    y_buffer = int(kernel_buffer[1] / 2)

    for prediction in predictions:
      print('Writing patch ' + str(patches) + '...')
      predPatch = np.add(np.argmax(prediction, axis = 2), 1)
      probPatch = np.max(prediction, axis = 2)
      predPatch = predPatch[x_buffer:x_buffer+KERNEL_SIZE, y_buffer:y_buffer+KERNEL_SIZE]
      probPatch = probPatch[x_buffer:x_buffer+KERNEL_SIZE, y_buffer:y_buffer+KERNEL_SIZE]

      # Create an example.
      example = tf.train.Example(
        features=tf.train.Features(
          feature={
            'class': tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value = np.ndarray.flatten(predPatch))),
            'prob': tf.train.Feature(
                float_list = tf.train.FloatList(
                    value = np.ndarray.flatten(probPatch)))
          }
        )
      )
      # Write the example.
      writer.write(example.SerializeToString())
      patches += 1

    writer.close()
    
def write_geotiff_predictions(pred_path, pred_image_base, kernel_buffer, cloud = True):
  """Write a numpy array as a GeoTIFF and optionally export to Google Cloud
  Parameters:
    pred_path (str): path to .tfrecord files
    pred_image_base (str): pattern matching basename of file(s)
    kernel_buffer (tpl): pixels to trim from H, W dimensions of prediction
    cloud (bool): copy output .tif file to google cloud using gsutil?
  Return:
    empty: writes a geotiff file to current working directory
  """
  # Set our output fiilenames
  out_geotiff = pred_image_base + '.tif'
  out_image_file = join(pred_path, 'outputs', out_geotiff)
  print(out_image_file)

  # Load the contents of the mixer file to a JSON object.
  jsonFile = join(pred_path, pred_image_base + '*.json')
  jsonText = !gsutil cat {jsonFile}
  # Get a single string w/ newlines from the IPython.utils.text.SList
  mixer = json.loads(jsonText.nlstr)
  transform = mixer['projection']['affine']['doubleMatrix']
  crs = mixer['projection']['crs']
  affine = rio.Affine(transform[0], transform[1], transform[2], transform[3], transform[4], transform[5])

  # get our prediction data and make predictions
  data = makePredDataset(pred_path, pred_image_base, kernel_buffer)
  image = make_array_predictions(data, jsonFile, kernel_buffer)

  with rio.open(
    out_geotiff,
    'w',
    driver = 'GTiff',
    width = image.shape[0],
    height = image.shape[1],
    count = 2,
    dtype = image.dtype,
    crs = crs,
    transform = affine) as dst:
    dst.write(np.transpose(image, (2,0,1)))
  print('Successfully wrote geotiff to local storage')
  
  if cloud:
      !gsutil cp {out_geotiff} {out_image_file}

def ingest_predictions(pred_path, out_image_base, user_folder):
  """
  Upload prediction image(s) to Earth Engine.
  Parameters:
    pred_path (str): Google cloud (or Drive) path storing prediction image files
    pred_image_base (str):
    user_folder (str): GEE directory to store asset
    out_image_base (str): base filename for GEE asset
  """
  jsonFile = !gsutil ls {join('gs://', pred_path, out_image_base + '*.json')}
  print(jsonFile)
  predFiles = !gsutil ls {join('gs://', pred_path, 'outputs', out_image_base + '*TFRecord')}
  print(predFiles)
  out_image_files = ' '.join(predFiles)
  # Start the upload.
  out_image_asset = join(user_folder, out_image_base)
  !earthengine upload image --asset_id={out_image_asset} {out_image_files} {jsonFile[0]}
  
def get_img_bounds(img, jsonFile):
    """Get the projected top left and bottom right coordinates of an image
    Parameters:
        img (ndarray): image to generate bounding coordinates for
        jsonFile (str): path to json file defining crs and image size
    Return:
        tpl: [[lat min, lon min],[lat max, lon max]]
    """
  jsonText = !gsutil cat {jsonFile}
  # Get a single string w/ newlines from the IPython.utils.text.SList
  mixer = json.loads(jsonText.nlstr)
  transform = mixer['projection']['affine']['doubleMatrix']
  print(transform)
  affine = rio.Affine(transform[0], transform[1], transform[2], transform[3], transform[4], transform[5])
  H,W = img.shape
  bounds = rio.transform.array_bounds(H, W, affine)
  print(bounds)
  lon_min, lat_min, lon_max, lat_max = bounds
  return [[lat_min, lon_min], [lat_max, lon_max]]