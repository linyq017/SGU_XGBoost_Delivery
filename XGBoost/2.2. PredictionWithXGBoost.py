"""
This script processes geospatial raster data using an XGBoost model for terrain analysis and prediction.
It handles multiple input sources:
1. Composite raster files containing multiple bands (DEM, SoilMap, etc.)
2. Additional indices stored in separate folders
The script reads these inputs, combines them, and uses a pre-trained XGBoost model to make predictions.
The predictions are then saved as new raster files, preserving the original geospatial metadata.

The workflow is:
- Load the trained XGBoost model
- For each composite raster file:
    - Read all bands from the composite file
    - Read corresponding indices from separate folders
    - !!!IMPORTANT!!! Ensure columns are in the exact order expected by the model
    - Make predictions using the model
    - Save results as new raster files with proper geospatial information

Required data structure:
- Composite folder: Contains multi-band raster files (.tif)
- Indices folder: Contains subfolders with additional index rasters
- Model file: Pre-trained XGBoost model in JSON format
"""

import xgboost as xgb
from osgeo import gdal, gdal_array
import os
import numpy as np
import pandas as pd
import rasterio as rio
from typing import List

def load_model(model_path: str) -> xgb.Booster:
    """
    Load a pre-trained XGBoost model from a JSON file.
    """
    return xgb.Booster(model_file=model_path)

def read_composite_bands(file_path: str) -> List[np.ndarray]:
    """
    Read all bands from a multi-band raster file.
    """
    bands_list = []
    ds = gdal.Open(file_path)
    
    # Iterate through each band in the raster
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        print(f'Reading band {i} from {os.path.basename(file_path)}')
        bands_list.append(band.ReadAsArray())
        
    return bands_list

def read_indices(indices_folder: str, composite_filename: str) -> List[np.ndarray]:
    """
    Read additional index rasters from subfolders.
    """
    indices_list = []
    
    # Search through all subfolders in the indices directory
    for subfolder in os.listdir(indices_folder):
        subfolder_path = os.path.join(indices_folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            file_path = os.path.join(subfolder_path, composite_filename)
            
            if os.path.exists(file_path):
                band = gdal_array.LoadFile(file_path)
                print(f'Reading index from {subfolder}/{composite_filename}')
                indices_list.append(band)
                
    return indices_list

def prepare_data(bands_list: List[np.ndarray]) -> pd.DataFrame:
    """
    Prepare combined band data for model prediction.    
    Note:
        Reshapes the data and ensures columns are in the correct order
        expected by the model
    """
    # Convert list of 2D arrays to single 3D array
    all_data = np.array(bands_list)
    print(f"Array shape before reshape: {all_data.shape}")
    print(f"Total number of elements: {all_data.size}")
    
    # Reshape to 2D array where each row is a pixel and each column is a band
    all_data = all_data.reshape(35, 1250 * 1250).T
    
    # Define and order columns to match model training data
    columns = ['DEM', 'EAS1ha', 'EAS10ha', 'DI2m', 'CVA', 'SDFS', 'DFME', 'Rugged', 'NMD',
              'SoilMap', 'HKDepth', 'SoilDepth', 'LandAge', 'MSRM', 'x', 'y', 'MED',
              'ANVAD20_15', 'CVA20', 'CVA50', 'Directiona', 'DownslopeI',
              'Geomorphon', 'MAXCURV20', 'MAXCURV50', 'MED20', 'MED50',
              'MINICURV20', 'MaxDownslo', 'NDVI', 'ProfileCur',
              'RELTOPOPOS', 'SLOPE20', 'SLOPE50', 'TWI20']
    
    df = pd.DataFrame(all_data, columns=columns)
    
    # Ensure columns are in the exact order expected by the model
    ordered_cols = ['x', 'y', 'DEM', 'EAS1ha', 'EAS10ha', 'DI2m', 'CVA', 'SDFS', 'DFME', 
                   'Rugged', 'NMD', 'SoilMap', 'HKDepth', 'SoilDepth', 'LandAge', 'MSRM', 
                   'MED', 'CVA20', 'CVA50', 'MAXCURV20', 'MAXCURV50', 'MINICURV20', 
                   'SLOPE20', 'SLOPE50', 'MED20', 'MED50', 'ANVAD20_15', 'Directiona', 
                   'DownslopeI', 'Geomorphon', 'MaxDownslo', 'NDVI', 'ProfileCur', 
                   'RELTOPOPOS', 'TWI20']
    
    return df[ordered_cols]

def predict_and_save(model: xgb.Booster, df: pd.DataFrame, composite_path: str, output_path: str) -> None:
    """
    Make predictions and save results as a new raster.
    
    Args:
        model: Loaded XGBoost model
        df: Prepared DataFrame with features
        composite_path: Path to original composite file (for metadata)
        output_path: Where to save the prediction raster
    
    Note:
        Preserves original geospatial metadata in the output raster
    """
    # Prepare data and make predictions
    dmatrix = xgb.DMatrix(df)
    pred = model.predict(dmatrix)
    pred = pred.reshape(1250, 1250)  # Reshape back to raster dimensions
    
    # Copy geospatial metadata from original file
    with rio.open(composite_path) as src:
        ras_meta = src.profile
        ras_meta.update(count=1)  # Output will be single band
    
    # Save prediction with proper geospatial information
    with rio.open(output_path, 'w', **ras_meta) as dst:
        dst.write(pred, 1)
        print(f'Saved prediction to {output_path}')

def process_file(model: xgb.Booster, composite_path: str, indices_folder: str, output_folder: str) -> None:
    """
    Process a single composite file through the entire workflow.
    
    Args:
        model: Loaded XGBoost model
        composite_path: Path to composite raster file
        indices_folder: Folder containing additional indices
        output_folder: Where to save prediction results
    """
    filename = os.path.basename(composite_path)
    if not filename.endswith('.tif'):
        print(f"Skipping {filename}: Not a TIFF file")
        return

    try:
        # Read and validate raster bands
        bands = read_composite_bands(composite_path)
        if len(bands) != 17:  
            print(f"Skipping {filename}: Expected 17 bands, found {len(bands)}")
            return

        # Read indices
        indices = read_indices(indices_folder, filename)
        all_bands = bands + indices

        # Validate number of features
        expected_features = 35  # Update if needed
        if len(all_bands) != expected_features:
            print(f"Skipping {filename}: Expected {expected_features} features, found {len(all_bands)}")
            return

        # Prepare data and generate prediction
        df = prepare_data(all_bands)
        output_path = os.path.join(output_folder, filename)
        predict_and_save(model, df, composite_path, output_path)

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    """
    Main execution function that sets up paths and processes all files.
    """
    # Configure paths - adjust these to match your system
    model_path = '/workspace/data/SGU/SFSI/SFSI/XBG10x_akermark_7class/20240417095539all_xy/best_model.json'  # path to xgboost model
    composite_folder = '/workspace/data/krycklan/composite' # path to raster composite folder
    indices_folder = '/workspace/data/krycklan/NewIndices' # path to directory containing subfolders of new indices
    output_folder = '/workspace/data/krycklan/XGB_output' # output directory
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model once to use for all predictions
    model = load_model(model_path)
    
    # Process each file in the composite folder
    for filename in os.listdir(composite_folder):
        composite_path = os.path.join(composite_folder, filename)
        process_file(model, composite_path, indices_folder, output_folder)