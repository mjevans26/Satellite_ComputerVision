
import json
from pathlib import Path
from importlib import reload
import numpy as np
import os
import sys
from os.path import join
from glob import glob
import io

import xarray as xr
import rasterio as rio
import rioxarray
from pyproj import CRS

import planetary_computer
from dask_gateway import GatewayCluster
from dask.distributed import wait, Client
import pystac_client
import pystac
import stackstac
import stac_vrt

FILE = Path(__file__).resolve() # full path to the current file, including extension
print('filepath', FILE)
ROOT = FILE.parents[0]  # list of upstream directories containing file
print('root', ROOT)
REL = Path(os.path.relpath(ROOT, Path.cwd()))
print('rel', REL)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(REL) not in sys.path:
    sys.path.append(str(REL))  # add REL to PATH

from azure.storage.blob import ContainerClient, BlobClient

def recursive_api_try(search):
    try:
        signed = planetary_computer.sign(search.get_all_items())
        # collection = search.item_collection()
        # print(len(collection), 'assets')
        # signed = [planetary_computer.sign(item).to_dict() for item in collection]
    except pystac_client.exceptions.APIError as error:
        print('APIError, trying again')
        signed = recursive_api_try(search)
    return signed
    
def export_blob(data: np.ndarray, container_client: ContainerClient, blobUrl: str) -> None:
    with io.BytesIO() as buffer:
        np.save(buffer, data)
        buffer.seek(0)
        blob_client = container_client.get_blob_client(blobUrl)
        blob_client.upload_blob(buffer, overwrite=True)  

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

def get_naip_stac(aoi, dates):
    
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    collections = ['naip']

    search = catalog.search(
        intersects = aoi,
        datetime = dates,
        collections = collections,
        limit = 500
    )

    items = planetary_computer.sign(search.get_all_items())
    # items is a pystac ItemCollection
    items2 = items.to_dict()
    features = items2['features'] 
    dates = [x['properties']['datetime'] for x in features]
    years = [date[0:4] for date in dates]
    years.sort()
    filtered = [x for x in features if x['properties']['datetime'][0:4] == years[-1]]

    # organize all naip images overlapping box into a vrt stac

    crss = np.unique(np.array([item['properties']['proj:epsg'] for item in filtered]))
    naip = stac_vrt.build_vrt(filtered, block_width=512, block_height=512, data_type="Byte")
    return naip

def get_lidar_stac(aoi, dates, crs = None, resolution = None):
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        intersects = aoi,
        datetime = dates, 
        collections = ['3dep-lidar-hag']
    )

    items = recursive_api_try(search)
    # items is a pystac ItemCollection
    items2 = items.to_dict()
    lidar = items2['features']

    # lidarUrl = lidar[0]['assets']['data']['href']
    lidarProperties = lidar[0]['properties']
    lidarCrs = lidarProperties['proj:projjson']['components'][0]['id']['code']
    lidarTransform = lidarProperties['proj:transform']
    if resolution:
        lidarRes = resolution
    else:
        lidarRes = lidarTransform[0]

    # lidarSide = 360//lidarRes
    # lidarZoom = round(600/lidarSide, 4)

    # lidarCrs = [asset['properties']['proj:projjson']['components'][0]['id']['code'] for asset in lidar]
    # print('LiDAR CRS', lidarCrs[0])
    lidarStac = stackstac.stack(
        lidar,
        epsg = lidarCrs,
        resolution = lidarRes,
        sortby_date = False,
        assets = ['data'])

    lidarMedian = lidarStac.median(dim = 'time') 
    projected = lidarMedian.rio.set_crs(lidarCrs)
    # reprojected = projected.rio.reproject(lidarCrs)

    return projected

def naip_mosaic(naips: list, crs: int):
    """ mosaic a list of naip stac items into a single xarray DataArray
    Parameters
    --------
    naips: list:
        list of naip image items in stac format
    crs: int
        epsg code specifying the common crs to project naip images
    Return
    ---
    xr.DataArray: single array of mosaicd naip images
    """
    data = [item for item in naips if item['properties']['proj:epsg'] == crs]
    crs = CRS.from_user_input(26918)
    naipStac = stac_vrt.build_vrt(
        data, block_width=512, block_height=512, data_type="Byte", crs = crs)
    naipImage = rioxarray.open_rasterio(naipStac, chunks = (4, 8192, 8192), lock = False)
    # reprojected = naipImage.rio.reproject('EPSG:4326')
    return(naipImage)

def get_s2_stac(dates, aoi):
    """from a pystac client return a stac of s2 imagery

    Parameters 
    ----
    client: pystac_client.Client()
        pystac catalog from which to retrieve assets
    dates: str
        start/end dates
    bbox: tpl
        [xmin, ymin, xmax, ymax]
    
    Return
    ---
    stackstac.stac()
    """
    # connect to the planetary computer catalog
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier = planetary_computer.sign_inplace)

    search = catalog.search(
        collections = ['sentinel-2-l2a'],
        datetime = dates,
        intersects = aoi,
        query={"eo:cloud_cover": {"lt": 10}} 
    )

    s2items = [item.to_dict() for item in list(search.get_items())]
    s2 = s2items[0]
    s2epsg = s2['properties']['proj:epsg']
    s2Stac = (
        stackstac.stack(
            s2items,
            epsg = s2epsg,
            assets=["B02", "B03", "B04", "B08"],  # red, green, blue
            chunksize=4096,
            resolution=10,
        )
        .where(lambda x: x > 0, other=np.nan)  # sentinel-2 uses 0 as nodata
        .assign_coords({'band':['B2', 'B3', 'B4', 'B8']})  # use GEE names that model expects
    )

    s2crs = s2Stac.attrs['crs']
    s2projected = s2Stac.rio.set_crs(s2crs)
    # trimmed = s2projected.rio.clip(geometries = [aoi], crs = 4326)
    return s2projected
    
def get_pc_imagery(aoi, dates, crs):
    """Get S2 imagery from Planetary Computer. REQUIRES a valid API token be added to the os environment
    Args:
        aoi: POLYGON geometry json
        dates (tpl): four YYYY-MM-DD date strings defining before and after
        crs (int): 4-digit epsg code representing coordinate reference system
    """
    # Creates the Dask Scheduler. Might take a minute.
    cluster = GatewayCluster(
        address = "https://pccompute.westeurope.cloudapp.azure.com/compute/services/dask-gateway",
        proxy_address = "gateway://pccompute-dask.westeurope.cloudapp.azure.com:80",
        auth = 'jupyterhub',
        worker_cores = 4
    )

    client = cluster.get_client()
    
    # allow our dask cluster to adaptively scale from 2 to 24 nodes
    cluster.adapt(minimum=2, maximum=24)
    
    # extract before and after dates from input in format required by PC
    before_dates = f'{dates[0]}/{dates[1]}'
    after_dates = f'{dates[2]}/{dates[3]}'

    # connect to the planetary computer catalog
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    # sentinel = catalog.get_child('sentinel-2-l2a')
    
    before_data = get_s2_stack(catalog, before_dates, aoi)
    after_data = get_s2_stack(catalog, after_dates, aoi)

    # convert provided coordinates into appropriate format for clipping xarray imagery
    xs = [x for x,y in aoi['coordinates'][0]]
    ys = [y for x,y in aoi['coordinates'][0]]
    bounds = [min(xs), min(ys), max(xs), max(ys)]

    # reduce the before and after image collections to a single image using median value per pixel
    before = before_data.median(dim="time")
    after = after_data.median(dim="time")
    
    # compute the result and load to local machine
    bef_clip = bef.rio.clip([aoi], crs).compute()
    aft_clip = aft.rio.clip([aoi], crs).compute()

    # This non-distributed method seems to be working but timing out
    # TODO: try changing chunk dimensions, try increasing timeout time of Webservice
    # bd, ad = dask.compute(bef_clip, aft_clip)

    # result_dict = wait([bef_clip, aft_clip], return_when = 'ALL_COMPLETED')
    
    # close our cluster
    client.close()
    cluster.shutdown()
    # return the before and after images as numpy arrays
    return bef_clip.data, aft_clip.data

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

