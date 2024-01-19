# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:07:52 2022

@author: mevans
"""

import os
from os.path import join
import rasterio as rio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.merge import merge
import shapely
from shapely.geometry import box
import geopandas as gpd
import numpy as np
from matplotlib.pyplot import imsave
import warnings
import random

def generate_chip_indices(arr, buff = 128, kernel = 256):
  """
  Parameters
  ---
    arr: np.ndarray
      3D array (H, W, C) for which indices should be generated
    buff: int
      size of pixels to be trimmed from chips
    kernel: int
      size of contiguous image chips
  Return
  ---
    list::np.ndarray: list containing (y,x) index of chips upper left corner
  """
  H, W, C = arr.shape
  side = buff + kernel
  x_buff = y_buff = buff//2
  
  y_indices = list(range(y_buff, H - side, kernel))
  x_indices = list(range(x_buff, W - side, kernel))

  indices = [(y_index, x_index) for y_index in y_indices for x_index in x_indices]
  return indices

def extract_chips(arr, buff = 128, kernel = 256):
    """Break an array into (potentially) overlapping chips for analysis
    Arguments:
        arr (ndarray): 3D array to run predictions on
        buff (int): size of pixels to be trimmed from chips
        kernel (int): size of contiguous image chips
    Return:
        list::np.ndarray: list containing image chips of size (kernel+buff, kernel+buff)
    """
    H, W, C = arr.shape
    side = buff + kernel
    x_buff = y_buff = buff//2
    chips = []

    chip_indices = generate_chip_indices(arr, buff, kernel)

    for x, y in chip_indices:
      chip = arr[y-y_buff:y+kernel+y_buff, x-x_buff:x+kernel+x_buff, :]
      chips.append(chip)
    
    return chips
    
def convert(size, box):
    """
    Convert coordinates of a bounding box given in image pixels to
    normalized [0,1] yolo coordinates
    
    Parameters
    ---
    size: tpl<int,int>
        height, width of image in pixels
    box: list[x0, y0, x1, y1]
        corners of box in pixels
        
    Return
    ---
      tpl(int, int, int, int): normalized x,y centroid and width, height of box
    """
    dw = 1./size[1]
    dh = 1./size[0]
    xmid = (box[0] + box[2])/2.0
    ymid = (box[1] + box[3])/2.0
    w0 = box[2] - box[0]
    h0 = box[3] - box[1]
    x = xmid*dw
    y = ymid*dh
    w = w0*dw
    h = h0*dh
    return (x,y,w,h)

def make_window(cx: int, cy:int, window_size: int) -> tuple:
    """Create an array window around a centroid
    
    Parameters
    ---
    cx: int
        centroid x-coord
    cy: int
        centroid y-coord
    window_size: int
        size of window in pixels
    
    Return
    ---
    tpl: coordinates of top left (x0, y0) and bottom right (x1, y1) window points
    """
    x0 = round(cx - window_size//2)
    y0 = round(cy - window_size//2)
    x1 = round(cx + window_size//2)
    y1 = round(cy + window_size//2)
    return (x0, y0, x1, y1)

def get_geo_transform(raster_src):
    """Get the geotransform for a raster image source.
    Arguments
    ---------
    raster_src : str, :class:`rasterio.DatasetReader`, or `osgeo.gdal.Dataset`
        Path to a raster image with georeferencing data to apply to `geom`.
        Alternatively, an opened :class:`rasterio.Band` object or
        :class:`osgeo.gdal.Dataset` object can be provided. Required if not
        using `affine_obj`.
    Returns
    -------
    transform : :class:`affine.Affine`
        An affine transformation object to the image's location in its CRS.
    """

    if isinstance(raster_src, str):
      with rio.open(raster_src) as src:
        affine_obj = src.transform
    elif isinstance(raster_src, rio.DatasetReader):
        affine_obj = raster_src.transform

    return affine_obj

def convert_poly_coords(geom, raster_src=None, affine_obj=None, inverse=False,
                        precision=None):
    """Georegister geometry objects currently in pixel coords or vice versa.
    Params
    ---------
    geom : :class:`shapely.geometry.shape` or str
        A :class:`shapely.geometry.shape`, or WKT string-formatted geometry
        object currently in pixel coordinates.
    raster_src : str, optional
        Path to a raster image with georeferencing data to apply to `geom`.
        Alternatively, an opened :class:`rasterio.Band` object or
        :class:`osgeo.gdal.Dataset` object can be provided. Required if not
        using `affine_obj`.
    affine_obj: list or :class:`affine.Affine`
        An affine transformation to apply to `geom` in the form of an
        ``[a, b, d, e, xoff, yoff]`` list or an :class:`affine.Affine` object.
        Required if not using `raster_src`.
    inverse : bool, optional
        If true, will perform the inverse affine transformation, going from
        geospatial coordinates to pixel coordinates.
    precision : int, optional
        Decimal precision for the polygon output. If not provided, rounding
        is skipped.
    Returns
    -------
    out_geom
        A geometry in the same format as the input with its coordinate system
        transformed to match the destination object.
    """

    if not raster_src and not affine_obj:
        raise ValueError("Either raster_src or affine_obj must be provided.")

    if raster_src is not None:
        affine_xform = get_geo_transform(raster_src)
    else:
        if isinstance(affine_obj, Affine):
            affine_xform = affine_obj
        else:
            # assume it's a list in either gdal or "standard" order
            # (list_to_affine checks which it is)
            if len(affine_obj) == 9:  # if it's straight from rasterio
                affine_obj = affine_obj[0:6]
            affine_xform = Affine(*affine_obj)

    if inverse:  # geo->px transform
        affine_xform = ~affine_xform

    if isinstance(geom, str):
        # get the polygon out of the wkt string
        g = shapely.wkt.loads(geom)
    elif isinstance(geom, shapely.geometry.base.BaseGeometry):
        g = geom
    else:
        raise TypeError('The provided geometry is not an accepted format. '
                        'This function can only accept WKT strings and '
                        'shapely geometries.')

    xformed_g = shapely.affinity.affine_transform(g, [affine_xform.a,
                                                      affine_xform.b,
                                                      affine_xform.d,
                                                      affine_xform.e,
                                                      affine_xform.xoff,
                                                      affine_xform.yoff])
    if isinstance(geom, str):
        # restore to wkt string format
        xformed_g = shapely.wkt.dumps(xformed_g)
    if precision is not None:
        xformed_g = _reduce_geom_precision(xformed_g, precision=precision)

    return xformed_g

def convert_pt(geometry: gpd.GeoSeries, out_crs: int, src_transform: list) -> tuple:
    """ Change a point to another crs
    
    Parameters
    ---
    geomegry: gpd.GeoSeries
        geoseries of points
    out_crs: int
        epsg for the desired crs
    
    Return
    ---
    tpl: (x,y) coordinates of point in new crs
    """
    pt = geometry.to_crs(out_crs)
    coords = convert_poly_coords(pt.iloc[0], affine_obj = src_transform, inverse = True, precision = None)
    x, y = np.rint(coords.x), np.rint(coords.y)
    return (x,y)

def win_jitter(window_size, jitter_frac=0.1):
    '''get x and y jitter
    Parameters
    ---------
        window_size (tpl<int, int.): dimensions of window to be jittered
        jitter_frac (float): proportion of window size to move window
    Returns
    --------
        tpl <int, int>: dx, dy in pixels
    '''
    val = np.rint(jitter_frac * window_size)
    dx = np.random.randint(-val, val)
    dy = np.random.randint(-val, val)
    
    return dx, dy

def get_centroid(geom_pix, verbose = True):
    """
    Get the centroid of a polygon

    Parameters
    ----------
    geom_pix : shapely POLYGON
    verbose : bool, optional
        Return print statements? The default is True.

    Returns
    -------
    cx : float
        centroid x coordinate in input crs.
    cy : float
        centroid y coordinate in input crs.

    """
    bounds = geom_pix.bounds
    area = geom_pix.area
    (minx, miny, maxx, maxy) = bounds
    dx, dy = maxx-minx, maxy-miny
    
    # get centroid
    centroid = geom_pix.centroid
    
    cx_tmp, cy_tmp = list(centroid.coords)[0]
    cx, cy = np.rint(cx_tmp), np.rint(cy_tmp)
    if verbose:
      print ("  bounds:", bounds )
      print ("  dx, dy:", dx, dy )
      print ("  area:", area )
      print("centroid:", centroid)
    
    return cx, cy

def make_jittered_window(cx, cy, image_h, image_w, window_size = 1280, jitter_frac = 0.1):
    """
    Create a jittered image window from and input image and geometry centroid

    Parameters
    ----------
    cx : float
        x-coordinate of centroid around which to jitter window.
    cy : float
        y-coordinate of centroid around which to jitter window.
    image_h : int
        height in pixels of input image.
    image_w : int
        width in pixels of input image.
    window_size : int, optional
        desired dimension of output window. The default is 1280.
    jitter_frac : float, optional
        proportion of window size to move window. The default is 0.2.

    Returns
    -------
    x0 : int
        minx coordinate of jittered window
    y0 : int
        miny coordinate of jittered window.
    x1 : int
        maxx coordinate of jittered window.
    y1 : int
        maxy coordinate of jittered window.

    """
    # number of pixels in x and y directions to shift window
    jx, jy = win_jitter(window_size, jitter_frac=jitter_frac)
    x0 = cx - window_size/2 + jx
    y0 = cy - window_size/2 + jy
    # ensure window does not extend outside larger image
    x0 = max(x0, 0)
    x0 = int(min(x0, image_w - window_size))
    y0 = max(y0, 0)
    y0 = int(min(y0, image_h - window_size))
    # set other side of square
    x1 = x0 + window_size
    y1 = y0 + window_size
    print('x0', x0, 'y0', y0, 'x1', x1, 'y1', y1)
    return x0, y0, x1, y1

def rasterio_to_img(array, out_path, nbands = 3, ext = None):
    """
    Write an array read by rasterio to an 8-bit integer image file

    Parameters
    ----------
    array : numpy.ndarray
        image array read by rasterio.
    out_path : str
        out image file path.
    nbands : int, optional
        number of image bands to write. The default is 3.
    ext : str, optional
        image file format extension. The default is 'png'.

    Returns
    -------
    None.

    """
    # convert from CHW to HWC and cast as unsigned 8-band int for saving
    t = array.transpose((1,2,0)).astype('uint8')
    print('array shape', t.shape)
    print('array min', t.min())
    print('array max', t.max())
    print('array type', t.dtype)
    # to use pre-trained YOLO weights, only grab RGB bands
    if ext:
        out_file = f"{out_path}.{ext}"
    else:
        out_file = out_path
    print('writing image to', out_file)
    imsave(out_file, t[:,:,:nbands], vmin = 0, vmax = 255)

def numpy_to_raster(arr: np.ndarray, mixer: dict, out_file: str, dtype:str):
    """
    Params
    ---
    arr: np.ndarray
        input (H,W,C) array to be converted to raster
    mixer_file: dict
        dictionary containing image dimension and spatial reference metadata required by rasterio.write
    out_file: str
        file path to destination raster file
    dtype: str
        output dtype accepted by rasterio.write (e.g., 'uint16', 'int32', 'float32', 'float64')
    
    Return
    ---
    None: writes raster data to destination file
    """
    C = arr.shape[-1]
    meta = {
        'driver':'GTiff',
        'width':mixer['cols'],
        'height':mixer['rows'],
        'count':C,
        'dtype':dtype,
        'transform':rio.Affine(*mixer['transform'][0:6]),
        'crs':mixer['crs'],
        'nodata':255
    }
    band_list = list(range(1,C+1))
    with rio.open(out_file, mode = 'w', **meta) as src:
        src.write(arr, band_list)
