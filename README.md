# Free Computer Vision on Satellite Data
This repository contains code used to produce computer vision models that can identify infrastructure in publicly available satellite imagery using Google Earth Engine and Tensorflow.

## Parking lots
As part of the [Long Island Solar Roadmap](https://solarroadmap.org), we are testing the ability for computer vision models to automate the detection and delineation of parking lots in NAIP satellite imagery.  This analysis uses the Deeplab v3 model with a pre-trained ResNet backbone.

## Solar arrays
Ground mounted solar arrays are prominent features on the landscape, and their proliferation can be hard to keep up with.  In collaboration with The Nature Conservancy's North Carolina chapter, we trained a computer vision model to detect and delineate solar arrays from Sentinel-2 data.  This UNET model can be used to rapidly update the map of solar energy in NC and other states. 

The outputs are available for inspection interactively through a [Google Earth Engine App](https://defendersofwildlifegis.users.earthengine.app/view/compviz)
![App image](/images/compVizApp.png)
