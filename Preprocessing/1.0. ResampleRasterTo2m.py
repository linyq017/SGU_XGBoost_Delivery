import os
import argparse
import rasterio
from rasterio import Affine
import glob

try:
    import whitebox
    wbt = whitebox.WhiteboxTools()
except:
    from WBT.whitebox_tools import WhiteboxTools
    wbt = WhiteboxTools()

# Define input and output paths
input_path = "/workspace/data/wbt/New_Indices"  # Can be a folder or a single file
output_folder = "/workspace/data/wbt/New_Indices/resampled2m"

# Check if output folder exists; if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Check if input is a directory or a single file
if os.path.isdir(input_path):
    tif_files = glob.glob(os.path.join(input_path, "*.tif"))  # Get all .tif files in the folder
elif os.path.isfile(input_path) and input_path.endswith(".tif"):
    tif_files = [input_path]  # Process single file
else:
    raise ValueError("Invalid input path. Provide a folder or a .tif file.")

# Loop through each file and resample
for tif_path in tif_files:
    # Extract base filename
    base_name = os.path.basename(tif_path)

    # Define output file path
    output_path = os.path.join(output_folder, base_name)

    # Perform resampling
    wbt.resample(
        inputs=tif_path,
        output=output_path,
        cell_size=2,
        method="bilinear",
    )

    print(f"Resampled: {tif_path} -> {output_path}")