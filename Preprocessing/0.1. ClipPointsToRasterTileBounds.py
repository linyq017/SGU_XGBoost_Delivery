import numpy as np
import os
import pandas as pd
import geopandas as gpd
import rasterio as rio
import glob
from rasterio.coords import BoundingBox
from shapely.geometry import box

"""
This script clips a point vector file to the bounding box of multiple raster tiles. 
It iterates through all raster files in a specified directory, extracts their spatial 
bounds, and clips a given point shapefile to each raster's extent. The clipped point 
data is then saved as individual shapefiles corresponding to each raster tile.
"""

# All multiband raster stacks used to clip the point file
allfiles = glob.glob('/workspace/data/SGU/orginal_fran_wl/CompositeBands/*.tif')

# Point vector file
pointfile = gpd.read_file('/workspace/data/SGU/SFSI/project_shapefile/sfsi3.shp')

for file in allfiles:
    print('Current file:', file)

    # Extract the raster tile name (without extension)
    name = os.path.splitext(os.path.basename(file))[0]

    # Get raster bounding box
    bbox = rio.open(file).bounds
    bounds = BoundingBox(left=bbox.left, bottom=bbox.bottom, right=bbox.right, top=bbox.top)

    # Create a polygon from the bounding box
    poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    
    print('Bounding box:', bounds)

    # Clip the point file to the raster's bounding box
    clipped = gpd.clip(pointfile, poly)

    # Save the clipped point file
    try:
        clipped.to_file(f'/workspace/data/SGU/SFSI/SFSI/new_clips/{name}.shp')
        print(f'Successfully saved: {name}.shp')
    except Exception as e:
        print(f'Error saving {name}.shp: {e}')
        continue
