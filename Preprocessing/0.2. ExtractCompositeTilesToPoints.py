import numpy as np
import os
import pandas as pd
import rasterio as rio
import glob
import geopandas as gpd

"""
This script extracts raster values from raster composites at point locations and stores the data in CSV files. 
It processes all point shapefiles that have been clipped to raster tile bounds, retrieves values 
from corresponding raster tiles, and appends them as new columns. The extracted data is then saved as individual CSV files. 
Finally, all CSV files are concatenated into a single dataset.
"""

# Navigate to folder containing point shapefiles clipped to raster extent
allpoints = glob.glob('/workspace/data/SGU/SFSI/SFSI/new_clips/*.shp')

for point in allpoints:
    # Load the point shapefile as a GeoDataFrame
    pts = gpd.read_file(point)
    pts_name = os.path.splitext(os.path.basename(point))[0]
    print('Reading point tile:', pts_name)

    # Create a list of coordinate pairs
    coords = [(x, y) for x, y in zip(pts.newx_point, pts.newy_point)]

    # Open the corresponding raster tile
    raster_path = f'/workspace/data/SGU/orginal_fran_wl/CompositeBands/{pts_name}.tif'
    with rio.open(raster_path) as src:
        proper_col_name = {
            1: "DEM", 2: "EAS1ha", 3: "EAS10ha", 4: "DI2m",
            5: "CVA", 6: "SDFS", 7: "DFME", 8: "Rugged", 9: "NMD", 10: "SoilMap",
            11: "HKDepth", 12: "SoilDepth", 13: "LandAge", 14: "MSRM", 15: 'Easting', 16: 'Northing', 17: "MED"
        }

        # Extract raster values for each band and attach to the dataframe
        for i in range(1, src.count + 1):
            pts[proper_col_name[i]] = [x[i - 1] for x in src.sample(coords)]
            print(f'{proper_col_name[i]} attached!')

    # Save extracted data to a CSV file
    output_csv = f'/workspace/data/SGU/SFSI/SFSI/extracted_csv/output_{pts_name}.csv'
    pts.to_csv(output_csv, index=False)
    print(f'Output {pts_name}.csv saved!')

# Step 2: Concatenate all extracted CSV files into a single dataset
csv_folder = '/workspace/data/SGU/SFSI/SFSI/extracted_csv'
csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))

# Initialize an empty DataFrame
concatenated_df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

# Save the concatenated data to a new CSV file
output_file = '/workspace/data/SGU/SFSI/SFSI/concatenated_data_original.csv'
concatenated_df.to_csv(output_file, index=False)
print('All CSV files successfully concatenated and saved!')
