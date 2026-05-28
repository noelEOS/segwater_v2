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
        simplify_tolerance_meters: float,
        keep_top_k: int | None = None,
    ):
        self.prob_map_path = prob_map_path
        self.reference_tif_path = reference_tif_path
        self.shape = shape
        self.dtype = np.float32 if precision == "float32" else np.float16
        
        # MLOps Configured Parameters
        self.threshold = threshold
        self.min_length_meters = min_length_meters
        self.simplify_tolerance_meters = simplify_tolerance_meters
        self.keep_top_k = keep_top_k

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

        records = []

        logger.info("Applying geospatial projection...")

        for contour_id, contour in enumerate(contours):
            xs, ys = rasterio.transform.xy(
                self.transform,
                contour[:, 0],
                contour[:, 1],
                offset="center",
            )

            if len(xs) < 2:
                continue

            line = LineString(zip(xs, ys))

            if line.is_empty or line.length == 0:
                continue

            if line.length < self.min_length_meters:
                continue

            records.append(
                {
                    "contour_id": contour_id,
                    "threshold": self.threshold,
                    "length": line.length,
                    "n_vertices": len(line.coords),
                    "geometry": line,
                }
            )

        logger.info(
            f"Filtering complete. {len(records)} valid geometries remain before top-k filtering."
        )

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=self.crs)

        if self.keep_top_k is not None and self.keep_top_k > 0:
            gdf = (
                gdf.sort_values("length", ascending=False)
                .head(self.keep_top_k)
                .reset_index(drop=True)
            )
            gdf["rank"] = range(1, len(gdf) + 1)

        logger.info(f"Final shoreline output contains {len(gdf)} geometries before simplification.")

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=self.crs)

        if self.keep_top_k is not None and self.keep_top_k > 0:
            gdf = (
                gdf.sort_values("length", ascending=False)
                .head(self.keep_top_k)
                .reset_index(drop=True)
            )
            gdf["rank"] = range(1, len(gdf) + 1)

        logger.info(f"Final shoreline output contains {len(gdf)} geometries.")

        if self.simplify_tolerance_meters > 0 and len(gdf) > 0:
            logger.info(
                f"Simplifying final shoreline geometries with "
                f"{self.simplify_tolerance_meters} meters tolerance..."
            )

            if gdf.crs is None:
                logger.warning(
                    "Source CRS is undefined. Cannot safely simplify in meters. "
                    "Skipping simplification."
                )
            else:
                try:
                    metric_crs = gdf.estimate_utm_crs()

                    if metric_crs is None:
                        logger.warning(
                            "Could not estimate a suitable metric CRS. "
                            "Skipping simplification."
                        )
                    else:
                        logger.info(f"Using metric CRS for simplification: {metric_crs}")

                        original_crs = gdf.crs
                        gdf_metric = gdf.to_crs(metric_crs)

                        gdf_metric["geometry"] = gdf_metric.geometry.simplify(
                            tolerance=self.simplify_tolerance_meters,
                            preserve_topology=True,
                        )

                        gdf_metric["length_simplified_m"] = gdf_metric.geometry.length
                        gdf_metric["n_vertices_simplified"] = gdf_metric.geometry.apply(
                            lambda geom: len(geom.coords) if geom is not None and not geom.is_empty else 0
                        )

                        gdf = gdf_metric.to_crs(original_crs)

                except Exception as exc:
                    logger.warning(
                        f"Metric simplification failed: {exc}. "
                        "Continuing with unsimplified geometries."
                    )        

        os.makedirs(os.path.dirname(output_geojson_path), exist_ok=True)
        gdf.to_file(output_geojson_path, driver="GeoJSON")
        
        logger.info(f"Shoreline vector saved successfully to {output_geojson_path}")
        return output_geojson_path