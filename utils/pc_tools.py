
import json
from pathlib import Path
import numpy as np
import os
import sys
from os.path import join
from glob import glob

import xarray as xr
import rio

import planetary_computer as pc
from dask_gateway import GatewayCluster
from dask.distributed import wait, Client
import pystac_client
import stackstac

FILE = Path(__file__).resolve() # full path to the current file, including extension
print('filepath', FILE)
ROOT = FILE.parents[0]  # list of upstream directories containing file
print('root', ROOT)
REL = Path(os.path.relpath(ROOT, Path.cwd()))
if str(REL) not in sys.path:
    sys.path.append(str(REL))  # add REL to PATH
 
from prediction_tools import extract_chips, predict_chips

from tensorflow.keras import models
from azure.storage.blob import BlobClient

def normalize_dataArray(da: xr.DataArray, dim: str) -> xr.DataArray:
  """Normalize (mean = 0, sd = 1) values in a xarray DataArray along given axis
  
  Parameters
  ---
  da: xarray.DataArray
    array to be normalized
  dim: str
    name of dimension along which to calculate mean and standard deviation (e.g. 'band')
    
  Return
  ---
  xarray.DataArray: input array with values scaled to mean = 0 and sd = 1
  """
  mean = da.mean(dim = dim, skipna = True)
  sd = da.std(dim = dim, skipna = True)
  normalized = (da - mean)/(sd+0.000001)
  return normalized

def trim_dataArray(da: xr.DataArray, size: int) -> xr.DataArray: 
  """Trim the remainder from x and y dimensions of a DataArray
  
  Parameters
  ---
  da: xarray:DataArray
    input array to be trimmed
  size: int
    size of chunks in x and y dimension. remaining array x&y size will be evenly divisible by this value 
  
  Return:
  xarray:DataArray: resized input array with x & y dimensions evenly divisible by 'size'
  """
  slices = {}
  for coord in ["y", "x"]:
      remainder = len(da.coords[coord]) % size
      slice_ = slice(-remainder) if remainder else slice(None)
      slices[coord] = slice_

  trimmed = da.isel(**slices)
  return trimmed

def get_blob_model(model_blob_url: str, weights_blob_url: str, custom_objects: dic = None) -> tensorflow.keras.models.Model:
  """Load a keras model from blob storage to local machine
  
  Provided urls to a model structure (.h5) and weights (.hdf5) files stored as azure blobs, download local copies of
  files and use them to instantiate a trained keras model
  
  Parameters
  ---
  model_blob_url: str
    authenticated url to the azure storage blob holding the model structure file
  weights_blob_url: str
    authenticated blob url to the azure storage blob holding the weights file
  custom_objects: dic
    optional, dictionary with named custom functions (usually loss fxns) needed to instatiate model
   
  Return
  ---
  tf.keras.models.Model: model with loaded weights
  """
  
  wp = Path('weights.hdf5')
  mp = Path('model.h5')

  # if we haven't already downloaded the trained weights
  if not wp.exists():
    weights_client = BlobClient.from_blob_url(
      blob_url = weights_blob_url
      )
    # write weights blob file to local 
    with wp.open("wb") as f:
      f.write(weights_client.download_blob().readall())
  # if we haven't already downlaoded the model structure       
  if not mp.exists():
    model_client = BlobClient.from_blob_url(
      blob_url = model_blob_url
    )
    # write the model structure to local file
    with mp.open("wb") as f:
       f.write(model_client.download_blob().readall())


  # m = models.load_model(mp, custom_objects = {'get_weighted_bce': get_weighted_bce})
  # m = get_binary_model(6, optim = OPTIMIZER, loss = get_weighted_bce, mets = METRICS)
#     m = get_unet_model(nclasses = 2, nchannels = 6, optim = OPTIMIZER, loss = get_weighted_bce, mets = METRICS)
  m = models.load_model(mp, custom_objects = custom_objects)
  m.load_weights(wp)
  return m
  
def predict_chunk(data: np.ndarray, model_blob_url: str, weights_blob_url: str, custom_objects: dic = None) -> np.ndarray:
    # print('input shape', data.shape)
    # print(np.max(data))
    m = get_model(model_blob_url, weights_blob_url, custom_objects)
    hwc = np.moveaxis(data, 0, -1)
    # our model expects 4D data
    nhwc = np.expand_dims(hwc, axis = 0)
    # tensor = tf.constant(nhwc, shape = (1,384,384,4))
    pred = m.predict(nhwc)
    logits = np.squeeze(pred[0])
    # predictions come out as 4d (0, W, H, 8)
    # classes = np.squeeze(np.argmax(pred, axis = -1))
    print('logits shape', logits.shape)
    return logits

def recursive_api_try(search: pystac_client.ItemSearch) -> pystac.Item:
  """Recursively try to sign and return items from planetary computer STAC catalog
  
  Parameters
  ---
  search: pystac_client.ItemSearch
    query to STAC endpoint from which to retrieve and sign items
  Return
  ---
  pystac.Item: list of signed items from search
  """
  try:
      collection = search.item_collection()
      signed = [pc.sign(item).to_dict() for item in collection]
  except pystac_client.exceptions.APIError as error:
      print('APIError, trying again')
      signed = recursive_api_try(search)
  return signed


# def get_pc_imagery(aoi, dates, crs):
#     """Get S2 imagery from Planetary Computer. REQUIRES a valid API token be added to the os environment
#     Args:
#         aoi: POLYGON geometry json
#         dates (tpl): four YYYY-MM-DD date strings defining before and after
#         crs (int): 4-digit epsg code representing coordinate reference system
#     """
#     # Creates the Dask Scheduler. Might take a minute.
#     cluster = GatewayCluster(
#         address = "https://pccompute.westeurope.cloudapp.azure.com/compute/services/dask-gateway",
#         proxy_address = "gateway://pccompute-dask.westeurope.cloudapp.azure.com:80",
#         auth = 'jupyterhub',
#         worker_cores = 4
#     )

#     client = cluster.get_client()
    
#     # allow our dask cluster to adaptively scale from 2 to 24 nodes
#     cluster.adapt(minimum=2, maximum=24)
    
#     # extract before and after dates from input in format required by PC
#     before_dates = dates[0]+'/'+dates[1]
#     after_dates = dates[2]+'/'+dates[3]

#     # connect to the planetary computer catalog
#     catalog = pystac_client.Client().open("https://planetarycomputer.microsoft.com/api/stac/v1")
#     sentinel = catalog.get_child('sentinel-2-l2a')
    
#     search_before = catalog.search(
#         collections = ['sentinel-2-l2a'],
#         datetime=before_dates,
#         intersects=aoi
#     )

#     search_after = catalog.search(
#         collections = ['sentinel-2-l2a'],
#         datetime=after_dates,
#         intersects=aoi
#     )

#     before_list = list(search_before.get_items())
#     after_list = list(search_after.get_items())

#     before_least_cloudy = [item for item in before_list if item.properties['eo:cloud_cover'] <= 10]
#     after_least_cloudy = [item for item in after_list if item.properties['eo:cloud_cover'] <= 10]

#     before_items = [pc.sign_item(i).to_dict() for i in before_least_cloudy]
#     after_items = [pc.sign_item(i).to_dict() for i in after_least_cloudy]
    
#     # sanity check to make sure we have retrieved and authenticated items fro planetary computer
#     blen = len(before_items)
#     alen = len(after_items)
#     print(f'{blen} images in before collection')
#     print(f'{alen} images in after collection')

#     # convert provided coordinates into appropriate format for clipping xarray imagery
#     xs = [x for x,y in aoi['coordinates'][0]]
#     ys = [y for x,y in aoi['coordinates'][0]]
#     bounds = [min(xs), min(ys), max(xs), max(ys)]
    
#     # create an 
#     before_data = (
#         stackstac.stack(
#             before_items[0],
#             epsg = 32617,
#             bounds_latlon = bounds,
#             # resolution=10,
#             assets=['B02', 'B03', 'B04', 'B08'],  # blue, green, red, nir
#             # chunks is for parallel computing on Dask cluster, only refers to spatial dimension
#             chunksize=(10000, 10000) # don't make smaller than native S2 tiles (100x100km)
#         )
#         .where(lambda x: x > 0, other=np.nan)  # sentinel-2 uses 0 as nodata
#         .assign_coords(band = lambda x: x.common_name.rename("band"))  # use common names
#     )

#     after_data = (
#         stackstac.stack(
#             after_items,
#             epsg=32617,
#             bounds_latlon = bounds,
#             # resolution=10,
#             assets=['B02', 'B03', 'B04', 'B08'],  # blue, green, red, nir
#             chunksize=(10000, 10000) # set chunk size to 256 to get one chunk per time step
#         )
#         .where(lambda x: x > 0, other=np.nan)  # sentinel-2 uses 0 as nodata
#         .assign_coords(band = lambda x: x.common_name.rename("band"))  # use common names
#      )
    
#     # reduce the before and after image collections to a single image using median value per pixel
#     before = before_data.median(dim="time")
#     after = after_data.median(dim="time")

#     # assign the native sentinel-2 crs the resulting xarrays
#     bef = before.rio.set_crs(32617)
#     aft = after.rio.set_crs(32617)
    
#     # compute the result and load to local machine
#     bef_clip = bef.rio.clip([aoi], crs).compute()
#     aft_clip = aft.rio.clip([aoi], crs).compute()

#     # This non-distributed method seems to be working but timing out
#     # TODO: try changing chunk dimensions, try increasing timeout time of Webservice
#     # bd, ad = dask.compute(bef_clip, aft_clip)

#     # result_dict = wait([bef_clip, aft_clip], return_when = 'ALL_COMPLETED')
    
#     # close our cluster
#     client.close()
#     cluster.shutdown()
#     # return the before and after images as numpy arrays
#     return bef_clip.data, aft_clip.data

# def test_PC_connection():
#     """Test our ability to retrieve satellite imagery from Planetary Computer
    
#     Without any processing, return the first Sentinel-2 image from a date range at
#     a known location
#     """
#         # Creates the Dask Scheduler. Might take a minute.
#     cluster = GatewayCluster(
#         address = "https://pccompute.westeurope.cloudapp.azure.com/compute/services/dask-gateway",
#         proxy_address = "gateway://pccompute-dask.westeurope.cloudapp.azure.com:80",
#         auth = 'jupyterhub',
#         worker_cores = 4
#     )

#     client = cluster.get_client()
    
#     # allow our dask cluster to adaptively scale from 2 to 24 nodes
#     cluster.adapt(minimum=2, maximum=24)
    
#     # define fixed start and end date for summer 2021
#     before_dates = '2021-05-01/2021-08-01'

#     # connect to the planetary computer catalog
#     catalog = pcClient.open("https://planetarycomputer.microsoft.com/api/stac/v1")
#     sentinel = catalog.get_child('sentinel-2-l2a')
    
#     search = catalog.search(
#         collections = ['sentinel-2-l2a'],
#         datetime=before_dates,
#         intersects=aoi
#     )

#     search_list = list(search_before.get_items())

#     least_cloudy = [item for item in search_list if item.properties['eo:cloud_cover'] <= 10]

#     items = [pc.sign_item(i).to_dict() for i in least_cloudy]
    
#     # sanity check to make sure we have retrieved and authenticated items fro planetary computer
#     ilen = len(items)
#     print(f'{ilen} images in collection')

#     # convert provided coordinates into appropriate format for clipping xarray imagery
#     bounds = [-76.503778, 38.988321, -76.530776, 38.988322]
    
#     # create an 
#     data = (
#         stackstac.stack(
#             items[0],
#             epsg = 32617,
#             bounds_latlon = bounds,
#             sortby_date = 'desc',
#             # resolution=10,
#             assets=['B02', 'B03', 'B04', 'B08'],  # blue, green, red, nir
#             # chunks is for parallel computing on Dask cluster, only refers to spatial dimension
#             chunksize= 'auto' # don't make smaller than native S2 tiles (100x100km)
#         )
#         .where(lambda x: x > 0, other=np.nan)  # sentinel-2 uses 0 as nodata
#         .assign_coords(band = lambda x: x.common_name.rename("band"))  # use common names
#     )
    
#     # reduce the before and after image collections to a single image using first valid pixel
#     before = data.mosaic(dim="time")

#     # assign the native sentinel-2 crs the resulting xarrays
#     bef = before.rio.set_crs(32617)
    
#     # compute the result and load to local machine
#     bef_local = bef.compute()

#     # This non-distributed method seems to be working but timing out
#     # TODO: try changing chunk dimensions, try increasing timeout time of Webservice
#     # bd, ad = dask.compute(bef_clip, aft_clip)

#     # result_dict = wait([bef_clip, aft_clip], return_when = 'ALL_COMPLETED')
    
#     # close our cluster
#     client.close()
#     cluster.shutdown()
#     # return the image as numpy arrays
#     return bef_local.data


# def run(aoi, dates, crs, m, buff = 128, kernel = 256):
#     """Retrieve Sentinel-2 imagery from Microsoft Planetary Computer and run change detection
#     Arguments:
#         aoi (dict): GeoJson like dictionary defining area of interest
#         crs (int): 4-digit epsg code representing coordinate reference system of the aoi
#         dates (tpl<str>): Four YYYY-MM-DD strings defining the before and after periods
#         m (keras.Model): model to be used to make predictions
#         buff (int): buffer to strip from prediction patches
#         kernel (int): size of side of prediction patches
#     Return:
#         numpy.ndarray: 3D array with per-pixel change probabilities
#     """
#     # returns before and after image tuple
#     bef_img, aft_img = get_pc_imagery(aoi, dates, crs)
#     arr = np.rollaxis(np.concatenate([bef_img, aft_img], axis = 0), 0, 3)

#     H,W,C = arr.shape
#     output = np.zeros((H, W, 1), dtype=np.float32)
    
#     chips, chip_indices = extract_chips(arr)

#     output = predict_chips(chips, chip_indices, m, output = output, kernel = kernel, buff = buff)

#     print(f'returning array of {output.shape}')
#     return output
