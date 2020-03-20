# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:44:19 2020

@author: MEvans
"""

import math
import ee
from stats import normalize

def clamp_and_scale(img, bands, p, AOI):
  """ 
  clip the upper range of an image based on percentile

  This function is similar to ee.Image().clip() and ee.Image().unitScale(),
  but operates on multiple bands with potentially different upper limits.

  Parameters:
    img (ee.Image): the image to modify
    bands (ee.List<str>): 
    p (int): upper percentile above which to truncate values
    AOI (ee.Geometry): area within which to calculate percentile

  Returns:
    ee.Image: rescaled image with band values [0, 1]
  """
  #create a list of the 99th percentile value for all bands
  percentiles = img.select(bands).reduceRegion(
      reducer = ee.Reducer.percentile([99]).repeat(ee.List(bands).size()),
      geometry = AOI,
      scale = 100,
      maxPixels = 1e13,
      tileScale = 12
  ).get('p99')

  #turn list of 99th percentiles into constant image
  upperImg = ee.Image.constant(percentiles).rename(bands)

  #clip the upper range of extreme values where sensors get washed out
  normImage = img.where(img.gte(upperImg), upperImg)

  # rescale the truncated image to [0, 1]
  rescaled = normalize(normImage, upperImg, ee.Image.constant(0))
  return ee.Image(rescaled)

def scene_median(imgCol, bands, sceneID):
  """ 
  Create median images for each unique scene in an image collection
  Parameters:
    imgCol (ee.ImageCollection):
    bands (list<str>): image bands on which to calculate medians
    sceneID (str): metadata field storing unique scene ID values
  Returns:
    ee.ImageCollection: composed of median images per scene
  """
  # first get list of all scene IDs
  scenes = ee.List(imgCol.aggregate_array(sceneID)).distinct()
  # define function to filter by scene id and take median

  medians = scenes.map(lambda str: imgCol.filter(ee.Filter.eq(sceneID, str)).median().set(sceneID, str))
  return ee.ImageCollection(medians).select(bands)

def get_overlap(imgCol1, imgCol2):
  """
  Calculate the area of overlap between two image collections
  Parameters:
    imgCol1 (ee.ImageCollection): first image collection
    imgCol2 (ee.ImageCollection): second image collection
  Returns:
    ee.Geometry: area of overlap
  """
  intersect = imgCol1.geometry(5).intersection(imgCol2.geometry(5), 5)
  return intersect

def hist_to_FC(hist, band):
  """
  convert a histogram of band values to a feature collection

  Args:
    hist (ee.Dictionary): output of histogram reducer on an image
    band (str): band name

  Return:
    ee.FeatureCollection: one feature for each histogram bin with 
  """
  # properties 'bucketMeans' and 'probability' (normalized cummulative probability).
  valsList = ee.List(ee.Dictionary(ee.Dictionary(hist).get(band)).get('bucketMeans'))
  freqsList = ee.List(ee.Dictionary(ee.Dictionary(hist).get(band)).get('histogram'))
  cdfArray = ee.Array(freqsList).accum(0)
  total = cdfArray.get([-1])
  normalizedCdf = cdfArray.divide(total)
  
  # create 2D array with histogram bucket means and normalized cdf values
  array = ee.Array.cat([valsList, normalizedCdf], 1)

  # define function to create a feature colleciton with properties determined by list
  def fxn(ls):
    return ee.Feature(None, {'dn': ee.List(ls).get(0), 'probability': ee.List(ls).get(1)})

  output = ee.FeatureCollection(array.toList().map(fxn))
  return output 

def make_FC(image, AOI):
  """
  create a feature colleciton from the histograms of an images bands

  Parameters:
    image (ee.Image): input image
    AOI (ee.Feaure): area within which to...
  Returns:
    ee.List: list of feature collections returned by hist_to_FC
  """
  # Histogram equalization start:
  bands = image.bandNames()
  histo = image.reduceRegion(
    reducer = ee.Reducer.histogram(
      maxBuckets = math.pow(2, 12)
    ), 
    geometry = AOI, 
    scale = 100, 
    maxPixels = 1e13, 
    tileScale = 12
  )
  
  def fxn(band):
    return hist_to_FC(histo, band)

  # map hist -> FC conversion fxn across bands
  output = bands.map(fxn)
  
  return output

def equalize(image1, image2, AOI):
  """
  use histogram matching to calibrate two images
  
  Parameters:
    image1 (ee.Image): reference image
    image2 (ee.Image): image to be calibrated
    AOI (ee.Geometry): area of overlap between the two images

  Returns:
    ee.Image: image2 with bands calibrated to the histogram(s) of image1 bands
  """
  bands = image1.bandNames()
  nBands = bands.size().subtract(1)
  
  # These are lists of feature collections
  fc1 = make_FC(image1, AOI)
  fc2 = make_FC(image2, AOI)

  def fxn(i):
    band = bands.get(i)
    classifier1 = ee.Classifier.randomForest(100)\
      .setOutputMode('REGRESSION')\
      .train(
        features = ee.FeatureCollection(ee.List(fc1).get(i)), 
        classProperty = 'dn', 
        inputProperties = ['probability']
    )

    classifier2 = ee.Classifier.randomForest(100)\
    .setOutputMode('REGRESSION')\
    .train(
      features = ee.FeatureCollection(ee.List(fc2).get(i)), 
      classProperty = 'probability', 
      inputProperties = ['dn']
    )
  
    # Do the shuffle: DN -> probability -> DN. Return the result.
    b = image2.select([band]).rename('dn');
    # DN -> probability -> DN
    output = b.classify(classifier2, 'probability')\
    .classify(classifier1, band)   
    
    return output

  imgList = ee.List.sequence(0, nBands).map(fxn)
  return ee.ImageCollection(imgList).toBands().rename(bands)

def equalize_collection(imgCol, bands, sceneID):
  """ 
  histogram equalize images in a collection by unique orbit path

  Parameters:
    imgCol (ee.ImageCollection): collection storing images to equalize
    bands (list<str>): list of band names to be calibrated
    sceneID (str): property by which images will be grouped

  Returns:
    ee.ImageCollection: median images per scene equalized to the westernmost path
  """
  # first get list of all scene IDs
  scenes = ee.List(imgCol.aggregate_array(sceneID)).distinct()
  # create an image collection of scene medians
  medians = scene_median(imgCol, bands, sceneID)
  # define a function to return the centroid longitude of each scene
  def get_coord_min(str):
    centroids = imgCol.filter(ee.Filter.eq(sceneID, str)).geometry(1).centroid(1)
    longs = centroids.coordinates().get(0)
    return longs
  # create a list of centroid longitudes
  coords = scenes.map(get_coord_min)
  # sort the scenes by increasing longitude
  scenes = scenes.sort(coords)
  # define a function that will equalize the list of scenes in succession
  def iterate_equalize(scene, prev):
    # take the previous median image
    prev = ee.List(prev)
    img1 = ee.Image(prev.get(-1))
    # take the next median image
    img2 = ee.Image(medians.filter(ee.Filter.eq(sceneID, scene)).first())
    # filter image collection to the previous scene
    index = scenes.indexOf(scene).subtract(1)
    imgCol1 = imgCol.filter(ee.Filter.eq(sceneID, scenes.get(index)))
    #imgCol1 = imgCol.filter(ee.Filter.eq(sceneID, prev))
    # filter image collection to the next scene
    imgCol2 = imgCol.filter(ee.Filter.eq(sceneID, scene))
    overlap = get_overlap(imgCol1, imgCol2)
    # if there is overlap between collections, equalize (returns image)
    # otherwise return the current image
    equalized = ee.Algorithms.If(overlap.area(5).gt(0), equalize(img1, img2, overlap), img2)
    update = ee.List(prev).add(equalized)
    return update
  # create a list of successively equalized scenes
  # initial value for iterate is the first median scene
  first = ee.Image(medians.filter(ee.Filter.eq(sceneID, scenes.get(0))).first())
  # take all but the first scene median and iteratively equalize
  output = scenes.slice(1).iterate(iterate_equalize, ee.List([first]))
  return ee.ImageCollection.fromImages(output)