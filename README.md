# Computer Vision with Free Satellite Data
This repository contains code used to produce computer vision models that can identify infrastructure in publicly available satellite imagery.

## Organization
The bulk of useful code in this repository is contained in the 'utils' directory. These python files are modules that can be used. They are organized, generally, by the imports they rely on and the kinds of functions they contain. For instance, utils/pc_tools.py imports the planetary computer ecosystem of packages and contains functions and classes for working with data from the MPC. Similarly, model_tools imports tensorflow and keras libraries and contains functions and classes for constructing and training deep learning models using these libraries.

## Parking lots
As part of the [Long Island Solar Roadmap](https://solarroadmap.org), we are testing the ability for computer vision models to automate the detection and delineation of parking lots in NAIP satellite imagery.  This analysis uses the Deeplab v3 model with a pre-trained ResNet backbone.

## Solar arrays
Ground mounted solar arrays are prominent features on the landscape, and their proliferation can be hard to keep up with.  The Chesapeake Conservancy trained a computer vision model to detect and delineate solar arrays from Sentinel-2 data.  This UNET model can be used to rapidly update the map of solar energy in DE, MD, PA, NY, VA, WV and other eastern states. These outputs were recently published in a [Biological Conservation](https://www.sciencedirect.com/science/article/pii/S0006320723001751) paper. 

### App
The outputs are available for inspection interactively through a [Google Earth Engine App]([https://defendersofwildlifegis.users.earthengine.app/view/compviz](https://mevans-cic.users.earthengine.app/view/cpksolar)https://mevans-cic.users.earthengine.app/view/cpksolar)
![App image](/images/compVizApp.png)
