# Import the Earth Engine Python Package
import ee

# Initialize Earth Engine
ee.Initialize()

def sentinel2toa(img):
    """
    Convert processed sentinel toa reflectance to raw values, and extract azimuth / zenith metadata
    
    Parameters:
        img (ee.Image): Sentinel-2 image to convert
        
    Returns:
        ee.Image: 
    """
    toa = img.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']) \
    .divide(10000)\
    .set('solar_azimuth', img.get('MEAN_SOLAR_AZIMUTH_ANGLE')) \
    .set('solar_zenith', img.get('MEAN_SOLAR_ZENITH_ANGLE')) \
    .set('viewing_azimuth', img.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_B8')) \
    .set('viewing_zenith', img.get('MEAN_INCIDENCE_ZENITH_ANGLE_B8')) \
    .set('CLOUDY_PIXEL_PERCENTAGE', img.get('CLOUDY_PIXEL_PERCENTAGE')) \
    #.set('system:time_start', img.get('system:time_start'));
    return img.select(['QA60']).addBands(toa);

def rescale(img, exp, thresholds):
    #print('rescale:', img, exp, thresholds)
    #return img.subtract(thresholds[0]).divide(thresholds[1]-thresholds[0])
    return img.expression(exp, {'img': img}).subtract(thresholds[0]).divide(thresholds[1] - thresholds[0])

def maskS2SR(img):
    """
    Apply built in masks to Sentinel-2 surface reflectance imagery
    Parameters:
        img (ee.Image): Sentinel-2 level 2A surface reflectange image
    Returns:
        ee.Image: masked image
    """
    scored = basicQA(img)
    maskBand = img.select('SCL')
    cloudMask = maskBand.neq(8).And(maskBand.neq(9))
    waterMask = maskBand.neq(6)
    cirrusMask = maskBand.neq(10)
    snowMask = maskBand.neq(11)
    darkMask = maskBand.neq(2).And(maskBand.neq(3))
    return scored.updateMask(cloudMask.And(waterMask).And(cirrusMask).And(snowMask).And(darkMask))

def waterScore(img):
    """ 
    Calculate a water likelihood score [0, 1]
    
    Parameters:
        img (ee.Image): Sentinel-2 image
        
    Returns:
        ee.Image: image with single ['waterscore'] band
    """
    print('waterScore:', img)
    img = sentinel2toa(img)
    # Compute several indicators of water and take the minimum of them.
    score = ee.Image(1.0)

    # Set up some params
    darkBands = ['B3', 'B4', 'B8', 'B11', 'B12']
    brightBand = 'B2'
    shadowSumBands = ['B8', 'B11', 'B12']
    # Water tends to be dark
    sum = img.select(shadowSumBands).reduce(ee.Reducer.sum())
    #sum = rescale(sum, [0.35, 0.2]).clamp(0, 1)
    sum = rescale(sum, 'img', [0.35, 0.2]).clamp(0, 1)
    score = score.min(sum)

    # It also tends to be relatively bright in the blue band
    mean = img.select(darkBands).reduce(ee.Reducer.mean())
    std = img.select(darkBands).reduce(ee.Reducer.stdDev())
    z = (img.select([brightBand]).subtract(std)).divide(mean)
    z = rescale(z, 'img', [0, 1]).clamp(0, 1)
    #z = rescale(z, [0,1]).clamp(0,1)
    score = score.min(z)

    # Water is at or above freezing
    # score = score.min(rescale(img, 'img.temp', [273, 275]));

    # Water is nigh in ndsi(aka mndwi)
    ndsi = img.normalizedDifference(['B3', 'B11'])
    ndsi = rescale(ndsi, 'img', [0.3, 0.8])
    #ndsi = rescale(ndsi, [0.3, 0.8])

    score = score.min(ndsi)

    return score.clamp(0, 1).rename(['waterScore'])

def basicQA(img):
    """
    Mask clouds in a Sentinel-2 image using builg in quality assurance band
    Parameters:
        img (ee.Image): Sentinel-2 image with QA band
    Returns:
        ee.Image: original image masked for clouds and cirrus
    """
    qa = img.select('QA60').int16()
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1024 # math.pow(2, 10)
    cirrusBitMask = 2048 #math.pow(2, 11)
    # Both flags should be set to zero, indicating clear conditions.
    #mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    dated = img.updateMask(mask).divide(10000).copyProperties(img)
    #dated = img.addBands(img.metadata('system:time_start', 'date')).updateMask(mask)
    return dated

# Function to cloud mask from the Fmask band of Landsat 8 SR data.
def maskL8sr(image):
  # Bits 3 and 5 are cloud shadow and cloud, respectively.
  cloudShadowBitMask = ee.Number(2).pow(3).int()
  cloudsBitMask = ee.Number(2).pow(5).int()

  # Get the pixel QA band.
  qa = image.select('pixel_qa')

  # Both flags should be set to zero, indicating clear conditions.
  mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))

  # Return the masked image, scaled to [0, 1].
  return image.updateMask(mask)


def cloudBands(img):
    ndmi = img.normalizedDifference(['B8', 'B11']).rename(['ndmi'])
    ndsi = img.normalizedDifference(['B3', 'B11']).rename(['ndsi'])
    cirrus = img.select(['B1', 'B10']).reduce(ee.Reducer.sum()).rename(['cirrus'])
    vis = img.select(['B4', 'B3', 'B2']).reduce(ee.Reducer.sum()).rename(['vis'])
    return img.addBands(ndmi).addBands(ndsi).addBands(cirrus).addBands(vis)


def darkC (img, R, G, B):
  R = img.select(R)
  G = img.select(G)
  B = img.select(B)
  maxRB = R.max(B)
  maxGB = G.max(B)
  maxRG = R.max(G)
  C1 = G.divide(maxRB).atan().rename(['C1'])
  C2 = R.divide(maxGB).atan().rename(['C2'])
  C3 = B.divide(maxRG).atan().rename(['C3'])
  return img.addBands(C1).addBands(C2).addBands(C3)

def sentinelCloudScore(img):
    """
    Compute a custom cloud likelihood score for Sentinel-2 imagery
    Parameters:
        img (ee.Image): Sentinel-2 image
    Returns:
        ee.Image: original image with added ['cloudScore'] band
    """
  #print('sentinelCloudScore:', img)
    im = sentinel2toa(img)
  # Compute several indicators of cloudyness and take the minimum of them.
    score = ee.Image(1)
      
      # Clouds are reasonably bright in the blue and cirrus bands.
      #score = score.min(rescale(im.select(['B2']), [0.1, 0.5]))
    score = score.min(rescale(im, 'img.B2', [0.1, 0.5]))
      #score = score.min(rescale(im.select(['B1']), [0.1, 0.3]))
    score = score.min(rescale(im, 'img.B1', [0.1, 0.3]))
      #score = score.min(rescale(im.select(['B1']).add(im.select(['B10'])), [0.15, 0.2]))
    score = score.min(rescale(im, 'img.B1 + img.B10', [0.15, 0.2]))
      
      # Clouds are reasonably bright in all visible bands.
      #score = score.min(rescale(im.select('B4').add(im.select('B3')).add(im.select('B2')), [0.2, 0.8]))
    score = score.min(rescale(im, 'img.B4 + img.B3 + img.B2', [0.2, 0.8]))
    
      # Clouds are moist
    ndmi = im.normalizedDifference(['B8','B11'])
      #score=score.min(rescale(ndmi, [-0.1, 0.1]))
    score=score.min(rescale(ndmi, 'img', [-0.1, 0.1]))
      
      # However, clouds are not snow.
    ndsi = im.normalizedDifference(['B3', 'B11'])
      #score=score.min(rescale(ndsi, [0.8, 0.6]))
    score=score.min(rescale(ndsi, 'img', [0.8, 0.6]))
      
    score = score.multiply(100).byte()
    print('score:', type(score))
     
    return img.addBands(score.rename(['cloudScore']))

