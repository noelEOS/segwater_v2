import os
import logging
from typing import List

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import LineString, Polygon
from skimage import measure

logger = logging.getLogger(__name__)

class ShorelineVectorizer:
    """
    Extracts high-precision sub-pixel contours from a probability map.
    Applies mathematical smoothing and projects pixel coordinates into 
    real-world geospatial formats (GeoJSON/Shapefile).
    """
    def __init__(
        self,
        prob_map_path: str,
        reference_tif_path: str,
        shape: tuple[int, int],
        precision: str,
        threshold: float,
        min_length_meters: float,
        simplify_tolerance_meters: float
    ):
        self.prob_map_path = prob_map_path
        self.reference_tif_path = reference_tif_path
        self.shape = shape
        self.dtype = np.float32 if precision == "float32" else np.float16
        
        # MLOps Configured Parameters
        self.threshold = threshold
        self.min_length_meters = min_length_meters
        self.simplify_tolerance_meters = simplify_tolerance_meters

        # Load spatial metadata from the original SAR swath
        with rasterio.open(self.reference_tif_path) as src:
            self.transform = src.transform
            self.crs = src.crs
            # Get pixel size to convert length/area filters from meters to pixels
            self.pixel_size_x = abs(self.transform.a)
            self.pixel_size_y = abs(self.transform.e)

        logger.info(f"Initialized Vectorizer. CRS: {self.crs} | Threshold: {self.threshold}")
        logger.info(f"Filters -> Min Length: {self.min_length_meters}m | Tolerance: {self.simplify_tolerance_meters}m")

    def extract_and_save(self, output_geojson_path: str):
        """
        Reads the memmap, executes Marching Squares, georeferences the lines, 
        and exports a GeoDataFrame.
        """
        logger.info("Loading global probability memmap into system RAM for contouring...")
        # Since contouring requires full topological context, we load the memmap into RAM.
        # A 25000x25000 float16 array is ~1.2GB, which fits comfortably in system memory.
        prob_map = np.memmap(self.prob_map_path, dtype=self.dtype, mode='r', shape=self.shape)
        
        logger.info(f"Executing Marching Squares at probability threshold {self.threshold}...")
        # Returns a list of (N, 2) arrays containing (row, col) coordinates
        contours = measure.find_contours(prob_map, level=self.threshold)
        logger.info(f"Extracted {len(contours)} raw contour fragments.")

        geometries = []
        
        logger.info("Applying geospatial projection and Douglas-Peucker simplification...")
        for contour in contours:
            # 1. Coordinate Transformation
            # contour[:, 0] is row (Y), contour[:, 1] is col (X)
            # We use rasterio's affine transform to map sub-pixel floats to Lat/Lon or UTM
            xs, ys = rasterio.transform.xy(
                self.transform, 
                contour[:, 0], 
                contour[:, 1]
            )
            
            # 2. Vectorization
            # A contour must have at least 2 points to be a line
            if len(xs) < 2:
                continue
                
            line = LineString(zip(xs, ys))
            
            # 3. Geometric Filtering
            # Drop tiny artifacts (e.g., small puddles or isolated noisy SAR pixels)
            if line.length < self.min_length_meters:
                continue
                
            # 4. Douglas-Peucker Smoothing
            # Smooths out the jagged edges caused by the 10m SAR resolution grid
            if self.simplify_tolerance_meters > 0:
                line = line.simplify(self.simplify_tolerance_meters, preserve_topology=True)
            
            geometries.append(line)

        # 5. Export to GeoJSON
        logger.info(f"Filtering complete. {len(geometries)} valid geometries remain.")
        
        gdf = gpd.GeoDataFrame(geometry=geometries, crs=self.crs)
        
        os.makedirs(os.path.dirname(output_geojson_path), exist_ok=True)
        gdf.to_file(output_geojson_path, driver="GeoJSON")
        
        logger.info(f"Shoreline vector saved successfully to {output_geojson_path}")
        return output_geojson_path