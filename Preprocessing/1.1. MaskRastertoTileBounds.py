from osgeo import gdal
import os
from pathlib import Path

def mask_with_bounds(mask_tif_path, source_tif_path, output_path):
    """
    Mask a source TIF using the bounds of another TIF file.
    
    Args:
        mask_tif_path (str): Path to the TIF file whose bounds will be used as mask
        source_tif_path (str): Path to the large TIF file to be masked
        output_path (str): Path where the masked TIF will be saved
    """
    # Open the mask TIF to get its bounds
    mask_ds = gdal.Open(mask_tif_path)
    mask_geotransform = mask_ds.GetGeoTransform()
    mask_minx = mask_geotransform[0]
    mask_maxy = mask_geotransform[3]
    mask_maxx = mask_minx + mask_geotransform[1] * mask_ds.RasterXSize
    mask_miny = mask_maxy + mask_geotransform[5] * mask_ds.RasterYSize
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Construct the gdal_translate command to crop using bounds
    gdal.Translate(
        output_path,
        source_tif_path,
        projWin=[mask_minx, mask_maxy, mask_maxx, mask_miny],
        format='GTiff'
    )

def process_multiple_sources(mask_folder, source_folder, base_output_folder):
    """
    Process multiple source TIFs, creating separate output folders for each source file.
    
    Args:
        mask_folder (str): Path to folder containing masking TIF files
        source_folder (str): Path to folder containing source TIF files
        base_output_folder (str): Base path where output folders will be created
    """
    # Get list of mask and source TIFs
    mask_files = list(Path(mask_folder).glob('*.tif'))
    source_files = list(Path(source_folder).glob('*.tif'))
    
    if not mask_files:
        print(f"No mask TIF files found in {mask_folder}")
        return
    
    if not source_files:
        print(f"No source TIF files found in {source_folder}")
        return
    
    # Process each source TIF
    for source_file in source_files:
        # Create output folder for this source file
        source_name = source_file.stem  # Get filename without extension
        output_folder = os.path.join(base_output_folder, source_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nProcessing source file: {source_file.name}")
        print(f"Creating output folder: {output_folder}")
        
        # Process each mask for this source file
        for mask_file in mask_files:
            output_path = os.path.join(output_folder, mask_file.name)
            
            print(f"  Masking with {mask_file.name}...")
            try:
                mask_with_bounds(str(mask_file), str(source_file), output_path)
                print(f"  Successfully created {output_path}")
            except Exception as e:
                print(f"  Error processing {mask_file.name}: {str(e)}")

# Example usage
if __name__ == "__main__":
    mask_folder = "/workspace/data/soildepth/Indices/Indices_Tiles/Aspect20_resample"      # Folder containing TIF files to use as masks
    source_folder = "/workspace/data/soildepth/Indices/Mosaic_Resampled/OneHot"  # Folder containing large TIF files to be masked
    base_output_folder = "/workspace/data/soildepth/Indices/Indices_Tiles"    # Base folder where output folders will be created
    
    process_multiple_sources(mask_folder, source_folder, base_output_folder)
