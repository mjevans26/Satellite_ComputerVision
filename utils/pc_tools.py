
import json
from pathlib import Path
from importlib import reload
import numpy as np
import os
import sys
from os.path import join
from glob import glob
import io

from osgeo import gdal
import xarray as xr
import rasterio as rio
from rasterio.vrt import WarpedVRT
from rioxarray.merge import merge_arrays
import rioxarray
from rioxarray.merge import merge_arrays
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

    items = planetary_computer.sign(search.item_collection_as_dict())
    # items is a pystac ItemCollection
    # items2 = items.to_dict()
    features = items['features'] 
    dates = [x['properties']['datetime'] for x in features]
    years = [date[0:4] for date in dates]
    years.sort()
    filtered = [x for x in features if x['properties']['datetime'][0:4] == years[-1]]
    urls = [item['assets']['image']['href'] for item in filtered]
    # organize all naip images overlapping box into a vrt stac
    crs_list = np.array([item['properties']['proj:epsg'] for item in filtered])
    crss = np.unique(crs_list)
    crs_counts = [len(crs_list[crs_list == crs]) for crs in crss]
    print('naip crss', crss)
    if len(crss) > 1:
        # rioxrs = []
        minority_idx = np.argmin(crs_counts)
        majority_idx = np.argmax(crs_counts)
        majority_urls = [url for i, url in enumerate(urls) if crs_list[i] == crss[majority_idx]]
        minority_urls = [url for i, url in enumerate(urls) if crs_list[i] == crss[minority_idx]]
        print('minority urls', minority_urls)
        minority_vrt = gdal.BuildVRT("./minority.vrt", minority_urls)
        majority_vrt = gdal.BuildVRT("./majority.vrt", majority_urls)
        warped_vrt = gdal.Warp("./warped.vrt", minority_vrt, format = 'vrt', dstSRS = f'EPSG:{crss[majority_idx]}')
        naipVRT = gdal.BuildVRT('./naiptmp.vrt',  [warped_vrt, majority_vrt])
        # naipVRT = None
        # for i, url in enumerate(urls):
        #     rioxr = rioxarray.open_rasterio(url)
        #     if crs_list[i] == crss[minority_idx]:
        #         reprojected = rioxr.rio.reproject(f'EPSG:{crss[majority_idx]}')
        #         rioxrs.append(reprojected)
        #     else:
        #         rioxrs.append(rioxr)
        # merged = merge_arrays(rioxrs)
        # return merged
    else:
        # rioxrs = [rioxarray.open_rasterio(url, lock = False) for url in urls]
        # merged = merge_arrays(rioxrs)
        # vrt = stac_vrt.build_vrt(filtered, block_width=512, block_height=512, data_type="Byte")
        # naipImg = rioxarray.open_rasterio(vrt, lock = False)
        naipVRT = gdal.BuildVRT('./naiptmp.vrt', urls)
    naipVRT = None
    naipImg = rioxarray.open_rasterio('./naiptmp.vrt', lock = False)
    return naipImg

def get_hag_stac(aoi, dates, crs = None, resolution = None):
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        intersects = aoi,
        datetime = dates, 
        collections = ['3dep-lidar-hag']
    )

    items = recursive_api_try(search)
    # items is a pystac ItemCollection
    items2 = items.to_dict()
    hag = items2['features']

    # hagUrl = hag[0]['assets']['data']['href']
    hagProperties = hag[0]['properties']
    hagCrs = hagProperties['proj:projjson']['components'][0]['id']['code']
    hagTransform = hagProperties['proj:transform']
    if resolution:
        hagRes = resolution
    else:
        hagRes = hagTransform[0]

    # hagSide = 360//hagRes
    # hagZoom = round(600/hagSide, 4)

    # hagCrs = [asset['properties']['proj:projjson']['components'][0]['id']['code'] for asset in hag]
    # print('hag CRS', hagCrs[0])
    hagStac = stackstac.stack(
        hag,
        epsg = hagCrs,
        resolution = hagRes,
        sortby_date = False,
        assets = ['data'])

    hagMedian = hagStac.median(dim = 'time') 
    projected = hagMedian.rio.set_crs(hagCrs)
    # reprojected = projected.rio.reproject(hagCrs)

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

def get_s2_stac(dates, aoi, cloud_thresh = 10, bands = ["B02", "B03", "B04", "B08"], epsg = None):
    """from a pystac client return a stac of s2 imagery

    Parameters 
    ----
    dates: str
        start/end dates
    aoi: shapely.geometry.Polygon
        polygon defining area of search
    cloud_thresh: int
        maximum cloudy pixel percentage of s2 images to return
    bands: list
        asset (band) names to return and stack
    epsg: int
        epsg coordinate system to reproject s2 data to
    
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
        query={"eo:cloud_cover": {"lt": cloud_thresh}} 
    )

    s2items = [item.to_dict() for item in list(search.get_items())]
    if len(s2items) > 0:
        s2 = s2items[0]
        if epsg:
            s2epsg = epsg
        else:
            s2epsg = s2['properties']['proj:epsg']

        s2Stac = (
            stackstac.stack(
                s2items,
                epsg = s2epsg,
                assets=bands,  # red, green, blue, nir
                chunksize=4096,
                resolution=10,
            )
            .where(lambda x: x > 0, other=np.nan)  # sentinel-2 uses 0 as nodata
        )

        s2crs = s2Stac.attrs['crs']
        s2projected = s2Stac.rio.set_crs(s2crs)
    else:
        # clipped = s2projected.rio.clip(geometries = [aoi], crs = epsg)
        s2Stac = None   
    return s2Stac

def get_s1_stac(dates, aoi, epsg  = None, bands = ["vv", "vh"]):
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
        datetime = dates,
        intersects = aoi,
        collections=["sentinel-1-rtc"],
        query={"sar:polarizations": {"eq": ['VV', 'VH']},
                'sar:instrument_mode': {"eq": 'IW'},
                'sat:orbit_state': {"eq": 'ascending'}
                }
    )

    s1items = search.item_collection()
    if not epsg:
        s1 = s1items[0]
        epsg = s1.properties['proj:epsg']
    s1Stac = stackstac.stack(
        s1items,
        epsg = epsg,
        assets=bands,
        resolution=10,
        gdal_env=stackstac.DEFAULT_GDAL_ENV.updated(
            always=dict(GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=1)
            )
    )

    # # get spatial reference info
    # s1crs = s1Stac.attrs['crs']
    # s1transform = s1Stac.attrs['transform']
    # s1res = s1transform[0]

    # s1projected = s1Stac.rio.set_crs(s1crs)
    # clipped = s1projected.rio.clip(geometries = [aoi], crs = 4326)
    return s1Stac

def get_s1_stac(dates, aoi, epsg  = None, bands = ["vv", "vh"]):
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
        datetime = dates,
        intersects = aoi,
        collections=["sentinel-1-rtc"],
        query={"sar:polarizations": {"eq": ['VV', 'VH']},
                'sar:instrument_mode': {"eq": 'IW'},
                'sat:orbit_state': {"eq": 'ascending'}
                }
    )

    s1items = search.item_collection()
    if not epsg:
        s1 = s1items[0]
        epsg = s1.properties['proj:epsg']
    s1Stac = stackstac.stack(
        s1items,
        epsg = epsg,
        assets=bands,
        resolution=10,
        gdal_env=stackstac.DEFAULT_GDAL_ENV.updated(
            always=dict(GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=1)
            )
    )

    # # get spatial reference info
    # s1crs = s1Stac.attrs['crs']
    # s1transform = s1Stac.attrs['transform']
    # s1res = s1transform[0]

    # s1projected = s1Stac.rio.set_crs(s1crs)
    # clipped = s1projected.rio.clip(geometries = [aoi], crs = 4326)
    return s1Stac

def get_ssurgo_stac(aoi, epsg)-> np.ndarray:
    """Sample ssurgo data in raster format
    
    Parameters
    ---
    aoi: shapely.geometry.Polygon
        polygon coordinates defining search aoi
    epsg: int
        cooridnate reference system epsg code to reproject ssurgo data to
    
    Returns
    ---
    np.ndarray: 3-dimensional raster (window_size, window_size, 4) containing ssurgo data
    """
    # connect to the PC STAC catalog
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    # get the gnatsco raster, which has 'mukey' values per pixel
    search = catalog.search(
        collections=["gnatsgo-rasters"],
        intersects=aoi
    )
    surgoitems = planetary_computer.sign(search.get_all_items())
    return surgoitems
    # surgoitems = [planetary_computer.sign(item).to_dict() for item in list(search.items())]
    # surgo = surgoitems[0]

    # surgowkt = surgo['properties']['proj:wkt2']
    # if epsg:
    #     surgoEPSG = epsg #surgoCrs.to_epsg()
    # else:
    #     surgoEPSG = CRS.from_wkt(surgowkt).to_epsg()

    # print(surgoEPSG)
    # # surgoepsg = surgo['properties']['proj:epsg']
    # surgoStac = stackstac.stack(
    #         surgoitems,
    #         # epsg = surgoEPSG,
    #         epsg = surgoEPSG,
    #         assets=['mukey'])

    # surgoTransform = surgoStac.attrs['transform']
    # # surgores = 10 #surgoTransform[0] TODO: COnfirm ssurgo is always 10 m resolution
    # # print('resolution', surgores)
    
    # temporal = surgoStac.median(dim = 'time')
    # return temporal, surgoTransform, surgoEPSG

def join_ssurgo(ssurgo_table, ssurgo_raster:np.ndarray):
    C,H,W = ssurgo_raster.shape
    # get the unique values and their indices from the raster so we can join to table data
    unique_mukeys, inverse = np.unique(ssurgo_raster, return_inverse=True) 
    # print('\t\tJoining SSURGO Arrays. Unique mukeys', unique_mukeys)
    rearranged = ssurgo_table[['mukey', 'hydclprs', 'drclassdcd', 'flodfreqdcd', 'wtdepannmin']].groupby('mukey').first().reindex(unique_mukeys, fill_value=np.nan).astype(np.float64)
    rearranged.loc[rearranged['wtdepannmin'] > 200.0, 'wtdepannmin'] = 200.0 # anything above 200 should be clipped to 200
    rearranged['wtdepannmin'] = rearranged['wtdepannmin'].fillna(200.0) # missing values are above 200 cm deep
    rearranged['wtdepannmin'] = rearranged['wtdepannmin']/200.0 # 200 cm is the max measured value

    rearranged['flodfreqdcd'] = rearranged['flodfreqdcd'].fillna(0.0) # missing values mean no flooding
    
    rearranged['drclassdcd'] = rearranged['drclassdcd'].fillna(0.0) # missing values mean no soil e.g. excessively drained
    
    rearranged['hydclprs'] = rearranged['hydclprs'].fillna(0.0) # missing values mean no soil e.g. not hydric
    rearranged['hydclprs'] = rearranged['hydclprs']/100.0 # 100 percent hydric is max
    # join tabluar data to ssurgo raster based on mukey
    ssurgo_hwc = rearranged.to_numpy()[inverse].reshape((H, W, 4)) # HWC
    return ssurgo_hwc

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

def run_local(aoi, dates, m, buff = 128, kernel = 256):
    """Retrieve Sentinel-2 imagery from Microsoft Planetary Computer and run change detection
    Arguments:
        aoi (dict): GeoJson like dictionary defining area of interest
        crs (int): 4-digit epsg code representing coordinate reference system of the aoi
        dates (tpl<str>): Four YYYY-MM-DD strings defining the before and after periods
        m (keras.Model): model to be used to make predictions
        buff (int): buffer to strip from prediction patches
        kernel (int): size of side of prediction patches
    Return:
        numpy.ndarray: 3D array with per-pixel change probabilities
    """
    # extract before and after dates from input in format required by PC
    before_dates = f'{dates[0]}/{dates[1]}'
    after_dates = f'{dates[2]}/{dates[3]}'

    # get our before and after stacs
    print('retrieving s2 data')
    bef_stac, bef_transform = get_s2_stac(before_dates, aoi)
    aft_stac, aft_transform = get_s2_stac(after_dates, aoi) # these are projected rioxarrays

    # create median composites
    bef_median = bef_stac.median(dim="time")
    aft_median = aft_stac.median(dim="time")

    #normalize
    bef_norm = normalize_dataArray(bef_median, 'band')
    aft_norm = normalize_dataArray(aft_median, 'band')
    
    # concatenate
    ds = xr.concat([bef_norm, aft_norm], dim="band").assign_coords({'band':['B2', 'B3', 'B4', 'B8','B2_1', 'B3_1', 'B4_1', 'B8_1']})

    C,H,W = ds.shape
    print('data shape:', ds.shape) # from planetary computer this is C, H, W
    rearranged = ds.transpose('y','x','band')
    print('rearranged shape', rearranged.shape)
    indices = prediction_tools.generate_chip_indices(rearranged, buff, kernel)
    print(len(indices), 'indices generated')
    template = np.zeros((H, W))
    print('template shape:', template.shape)
    # print('generating chips')
    # chips, chip_indices = extract_chips(ds)
    # print(len(chip_indices), 'chips generated')
    dat = rearranged.values
    print('running predictions')
    output = predict_chips(dat, indices, template, m, kernel = kernel, buff = buff)

    # print(f'returning array of {output.shape}')
    return output, bef_median, aft_median, aft_transform

def run_dask(model_blob_url, weights_blob_url, custom_objects, dates, aoi):
    # # create a dask cluster
    # print('spinning up Dask Cluster')
    # cluster = GatewayCluster(
    #     address = "https://pccompute.westeurope.cloudapp.azure.com/compute/services/dask-gateway",
    #     proxy_address = "gateway://pccompute-dask.westeurope.cloudapp.azure.com:80",
    #     auth = 'jupyterhub',
    #     worker_cores = 4
    # )

    # client = cluster.get_client()
    # client.upload_file(f'{str(ROOT)}/model_tools.py', load = True)

    # # allow our dask cluster to adaptively scale from 2 to 24 nodes
    # cluster.adapt(minimum=4, maximum=24)
    # print('cluster created', cluster.dashboard_link)

    # extract before and after dates from input in format required by PC
    before_dates = f'{dates[0]}/{dates[1]}'
    after_dates = f'{dates[2]}/{dates[3]}'

    # get our before and after stacs
    print('retrieving s2 data')
    bef_stac = get_s2_stac(before_dates, aoi)
    aft_stac = get_s2_stac(after_dates, aoi) # these are projected rioxarrays

    # create median composites
    bef_median = bef_stac.median(dim="time")
    aft_median = aft_stac.median(dim="time")

    #normalize
    bef_norm = normalize_dataArray(bef_median, 'band')
    aft_norm = normalize_dataArray(aft_median, 'band')
    
    # concatenate
    ds = xr.concat([bef_norm, aft_norm], dim="band").assign_coords({'band':['B2', 'B3', 'B4', 'B8','B2_1', 'B3_1', 'B4_1', 'B8_1']})

    trimmed = trim_dataArray(ds, 256)
    chunked = trimmed.chunk({'x':256, 'y':256})

    print('running chunked predictions')
    meta = np.array([[]], dtype="float32")
    predictions_array = chunked.data.map_overlap(
        lambda x: predict_chunk(x, model_blob_url, weights_blob_url, custom_objects),
        depth = (0, 64, 64),
        boundary = 0,
        meta=meta,
        drop_axis=0    
    )

    # predictions = predictions_array

    # # to restore spatial reference, cast back to Xarray
    # out = xr.DataArray(
    #     predictions,
    #     coords=trimmed.drop_vars("band").coords,
    #     dims=("y", "x"),
    # )
    
    return(predictions_array)


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

