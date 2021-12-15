import ee

# Initialize Earth Engine
ee.Initialize()

def norm_p(z):
    """ 
    Caclulate (approx) the p-value for a standard normal distribution
    
    Parameters:
        z (ee.Image): image containing z-scores
        
    Returns:
        ee.Image: image containing p-values
    """
    return ee.Image.constant(1).subtract(z.multiply(-1.65451).exp().add(1).pow(-1))

def chi_p(chi, df):
    """ Caclulate the CDF probability of a chi-square statistic
    Parameters:
        chi (ee.Image): single band image with observations from a chi-squared dist
        df (int): degrees of freedom
    Returns:
        ee.Image: single band image of probabilities
    """
    cdf = ee.Image(chi.divide(2)).gammainc(ee.Number(df).divide(2))
    return cdf.rename(['p'])

def gamma_p(stat, df):
    shape = ee.Image(1)
    scale = ee.Image(df)
    denom = shape.gamma()
    num = shape.gammainc(stat.divide(scale))
    return num.divide(denom).rename(['p'])

def normalize(img, maxImg, minImg):
  """
  Scale an image from 0 to 1

  Parameters:
    img (ee.Image): image to be rescaled
    maxImg (ee.Image): image storing the maximum value of the image
    minImg (ee.Image): image storing the minimum value of the image
  Returns:
    ee.Image:
  """
  return img.subtract(minImg).divide(maxImg.subtract(minImg))

def standardize(img):
  """
  Standardize an image to z-scores using mean and sd

  Parameters:
    img (ee.Image): image to be rescaled standardized
    
  Returns:
    ee.Image: image containing z-scores per band
  """
  bands = img.bandNames()
  mean = img.reduceRegion(
      reducer= ee.Reducer.mean(),
      scale= 300).toImage()
  sd = img.reduceRegion(
      reducer= ee.Reducer.stdDev(),
      scale= 300
  ).toImage(bands)
  return img.subtract(mean).divide(sd)


def ldaScore(img, inter, xbands, coefficients):
  """
  Function converting multiband image into single band image of LDA scores
  
  Parameters:
      img (ee.Image): multiband image
      int (float): intercept parameter from LDA analysis
      xbands (ee.List<string>): string list of n band names
      coefficients (ee.List<float>): numeric list of length n containing LDA coefficients
  Returns:
    ee.Image: image with one band containing LDA scores based on provided coefficients
  """
  bands = img.select(xbands)
  coeffs = ee.Dictionary.fromLists(xbands, coefficients).toImage(xbands)
  score = bands.multiply(coeffs).addBands(ee.Image(inter)).reduce(ee.Reducer.sum())
  return score

