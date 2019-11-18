# Deep Learning for Mental Health

This repo is based on another paper: https://github.com/adymaharana/ObesityDL

## How to run the code:
1. python3 download_img.py (just to get la city images, to obtain other images, load shapefiles and csv)
2. run Models.ipynb and go to the VGG section

## Steps
1. Obtain census tract data per city selected
2. Obtain shapefile per city
3. Obtain images from google static maps api

## TODO
- Finish up EDA of mental health dataset
- Write scraper that gets images from google static maps api
- Use VGG-CNN, freeze 2nd fully connected layer, attach regression model
- Writeup findings
