import os
import glob
import fiona
import rasterio as rio
from rasterio.mask import mask


"""
This script processes raster files by clipping them to the extent of shapefiles. It supports both individual raster files and directories containing multiple TIFF files. 
Each raster is clipped using all shapefiles within a specified folder, and the results are saved in a structured output directory. 
The processed files are organized into subfolders corresponding to the original raster file names.
"""

# Path to input raster(s) and shapefiles
input_raster = "/workspace/data/wbt/New_Indices"  # Can be a folder or a single .tif file
shapefile_folder = "/workspace/data/SGU/orginal_fran_wl/rastertindex"
output_base_dir = "/workspace/data/wbt/New_Indices_Tiles"

# Ensure output directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Check if input_raster is a folder or a single file
if os.path.isdir(input_raster):
    # If it's a folder, get all TIFF files
    raster_files = glob.glob(os.path.join(input_raster, "*.tif"))
else:
    # If it's a single file, wrap it in a list
    if input_raster.lower().endswith(".tif"):
        raster_files = [input_raster]
    else:
        raise ValueError("Input must be a directory containing TIFFs or a single .tif file.")

# Get all shapefiles in the folder
shapefiles = glob.glob(os.path.join(shapefile_folder, "*.shp"))

for raster_path in raster_files:
    # Extract raster file name (without extension)
    raster_name = os.path.splitext(os.path.basename(raster_path))[0]

    # Create a subfolder for this raster's clipped outputs
    raster_output_dir = os.path.join(output_base_dir, raster_name)
    os.makedirs(raster_output_dir, exist_ok=True)

    for shp_file in shapefiles:
        # Extract shapefile name (without extension)
        shp_name = os.path.splitext(os.path.basename(shp_file))[0]
        
        try:
            print(f"Processing {raster_name} with {shp_name}")

            # Read the shapefile
            with fiona.open(shp_file, "r") as shapefile:
                shapes = [feature["geometry"] for feature in shapefile]

            # Open the raster file
            with rio.open(raster_path) as src:
                # Mask the raster using the shape
                out_image, out_transform = mask(src, shapes, crop=True)
                out_meta = src.meta.copy()

            # Update metadata
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            # Define output file path inside the raster's subfolder
            output_path = os.path.join(raster_output_dir, f"{shp_name}.tif")

            # Write the clipped raster
            with rio.open(output_path, "w", **out_meta) as dst:
                dst.write(out_image)
            
            print(f"Saved clipped raster: {output_path}")

        except Exception as e:
            print(f"Error processing {raster_name} with {shp_name}: {e}")
