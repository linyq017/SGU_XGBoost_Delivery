import numpy as np
import os
import pandas as pd
import rasterio as rio
import glob
import geopandas as gpd


"""
This script extracts raster values at specific point locations from multiple large TIFF raster files and stores them in a GeoDataFrame. 
It first loads a point shapefile containing sampling locations, extracts their coordinates, and then iterates through a folder of raster files. 
The result is a dataset where each row represents a point with its extracted values from different raster layers, allowing for spatial analysis and comparison
"""

# Load the point vector file (shapefile) containing sampling locations
pointfile = gpd.read_file('/workspace/data/SGU/SFSI/project_shapefile/sfsi3.shp')

# Extract coordinates from the shapefile as a list of (x, y) tuples
coords = [(x, y) for x, y in zip(pointfile.newx_point, pointfile.newy_point)]

# Define the folder containing the large raster (TIFF) files
tif_folder = '/workspace/data/wbt/newtifs'

# Iterate through all TIFF files in the folder
for tif_file in os.listdir(tif_folder):
    if tif_file.endswith('.tif'):  # Ensure the file is a TIFF
        tif_path = os.path.join(tif_folder, tif_file)
        print("Reading file:", tif_file)
        
        # Open the raster file
        with rio.open(tif_path) as src:
            # Sample raster values at the given point coordinates
            values = [x[0] for x in src.sample(coords)]
            
            # Add the extracted raster values as a new column in the GeoDataFrame
            pointfile[tif_file] = values

print("Extraction completed.")
# Define output file paths
output_csv = "/workspace/data/SGU/SFSI/extracted_raster_values.csv"
output_shp = "/workspace/data/SGU/SFSI/extracted_raster_values.shp"

# Export the updated GeoDataFrame to CSV and Shapefile formats
pointfile.to_csv(output_csv, index=False)
pointfile.to_file(output_shp)

print(f"Results saved to:\nCSV: {output_csv}\nShapefile: {output_shp}")