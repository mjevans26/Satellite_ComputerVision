
import json
import numpy as np
import os
from os.path import join
from glob import glob

import xarray as xr
import rioxarray

import planetary_computer as pc
from dask_gateway import GatewayCluster
from dask.distributed import wait, Client
from pystac_client import Client as pcClient
import stackstac

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
    before_dates = dates[0]+'/'+dates[1]
    after_dates = dates[2]+'/'+dates[3]

    # connect to the planetary computer catalog
    catalog = pcClient.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    sentinel = catalog.get_child('sentinel-2-l2a')
    
    search_before = catalog.search(
        collections = ['sentinel-2-l2a'],
        datetime=before_dates,
        intersects=aoi
    )

    search_after = catalog.search(
        collections = ['sentinel-2-l2a'],
        datetime=after_dates,
        intersects=aoi
    )

    before_list = list(search_before.get_items())
    after_list = list(search_after.get_items())

    before_least_cloudy = [item for item in before_list if item.properties['eo:cloud_cover'] <= 10]
    after_least_cloudy = [item for item in after_list if item.properties['eo:cloud_cover'] <= 10]

    before_items = [pc.sign_item(i).to_dict() for i in before_least_cloudy]
    after_items = [pc.sign_item(i).to_dict() for i in after_least_cloudy]
    
    # sanity check to make sure we have retrieved and authenticated items fro planetary computer
    blen = len(before_items)
    alen = len(after_items)
    print(f'{blen} images in before collection')
    print(f'{alen} images in after collection')

    # convert provided coordinates into appropriate format for clipping xarray imagery
    xs = [x for x,y in aoi['coordinates'][0]]
    ys = [y for x,y in aoi['coordinates'][0]]
    bounds = [min(xs), min(ys), max(xs), max(ys)]
    
    # create an 
    before_data = (
        stackstac.stack(
            before_items[0],
            epsg = 32617,
            bounds_latlon = bounds,
            # resolution=10,
            assets=['B02', 'B03', 'B04', 'B08'],  # blue, green, red, nir
            # chunks is for parallel computing on Dask cluster, only refers to spatial dimension
            chunksize=(10000, 10000) # don't make smaller than native S2 tiles (100x100km)
        )
        .where(lambda x: x > 0, other=np.nan)  # sentinel-2 uses 0 as nodata
        .assign_coords(band = lambda x: x.common_name.rename("band"))  # use common names
    )

    after_data = (
        stackstac.stack(
            after_items,
            epsg=32617,
            bounds_latlon = bounds,
            # resolution=10,
            assets=['B02', 'B03', 'B04', 'B08'],  # blue, green, red, nir
            chunksize=(10000, 10000) # set chunk size to 256 to get one chunk per time step
        )
        .where(lambda x: x > 0, other=np.nan)  # sentinel-2 uses 0 as nodata
        .assign_coords(band = lambda x: x.common_name.rename("band"))  # use common names
     )
    
    # reduce the before and after image collections to a single image using median value per pixel
    before = before_data.median(dim="time")
    after = after_data.median(dim="time")

    # assign the native sentinel-2 crs the resulting xarrays
    bef = before.rio.set_crs(32617)
    aft = after.rio.set_crs(32617)
    
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

def run(aoi, dates, crs, m, buff = 128, kernel = 256):
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
    # returns before and after image tuple
    bef_img, aft_img = get_pc_imagery(aoi, dates, crs)
    arr = np.rollaxis(np.concatenate([bef_img, aft_img], axis = 0), 0, 3)

    H,W,C = arr.shape
    output = np.zeros((H, W, 1), dtype=np.float32)
    
    chips, chip_indices = extract_chips(arr)

    output = predict_chips(chips, chip_indices, m, output = output, kernel = kernel, buff = buff)

    print(f'returning array of {output.shape}')
    return output
