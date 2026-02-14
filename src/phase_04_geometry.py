"""
Phase 4: High-Performacalculate_nearest_cloud_distances_bandedce Computational Geometry
=================================================

This module performs ECEF coordinate transformation and KD-Tree nearest-neighbor
search to calculate the distance from each OCO-2 footprint to the nearest MODIS
cloud pixel.

Key Functions:
- convert_to_ecef: Transform geodetic coordinates to Earth-Centered Earth-Fixed (ECEF)
- integrate_myd03_geolocation: Map MODIS pixel indices to real lat/lon from MYD03
- build_kdtree: Construct spatial KD-Tree from cloud pixels
- calculate_nearest_cloud_distances: Query distances by OCO-2 granule

ECEF Transformation:
- Uses WGS84 ellipsoid parameters
- Converts (lat, lon, alt) → (x, y, z) in meters
- Avoids polar distortions in distance calculations

KD-Tree Search:
- O(N log M) complexity for N queries over M cloud pixels
- Processes each OCO-2 granule separately for memory efficiency
- Caps distances at 50 km (beyond typical cloud influence)
"""

import logging
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import urllib.request
import urllib.parse

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - install with: pip install scipy")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - install with: pip install matplotlib")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - install with: pip install pillow")

try:
    import xml.etree.ElementTree as ET
    XML_AVAILABLE = True
except ImportError:
    XML_AVAILABLE = False

try:
    from shapely.geometry import Polygon, Point
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logger.warning("shapely not available - install with: pip install shapely")

try:
    from pyhdf.SD import SD, SDC
    HDF4_AVAILABLE = True
except ImportError:
    HDF4_AVAILABLE = False
    logger.debug("pyhdf not available - MYD03 geolocation integration will be limited")

# Handle both package and direct imports
try:
    from .phase_03_processing import OCO2Footprint, MODISCloudMask
    from .phase_02_ingestion import DownloadedFile
except ImportError:
    from phase_03_processing import OCO2Footprint, MODISCloudMask
    from phase_02_ingestion import DownloadedFile


@dataclass
class CollocationResult:
    """Result of OCO-2/MODIS cloud collocation."""
    
    # OCO-2 sounding information
    sounding_id: int
    granule_id: str
    footprint_lat: float
    footprint_lon: float
    viewing_mode: str
    
    # Nearest cloud information
    nearest_cloud_dist_km: float
    nearest_cloud_lat: float
    nearest_cloud_lon: float
    cloud_classification: str  # 'Cloudy' or 'Uncertain'
    
    def __repr__(self):
        return (f"CollocationResult(sounding_id={self.sounding_id}, "
                f"distance={self.nearest_cloud_dist_km:.2f} km, "
                f"cloud={self.cloud_classification})")


class GeometryProcessor:
    """
    Phase 4: High-Performance Computational Geometry
    
    Performs ECEF transformation and KD-Tree nearest-neighbor search
    to calculate cloud distances for OCO-2 soundings.
    """
    
    # WGS84 ellipsoid parameters
    WGS84_A = 6378137.0  # Semi-major axis (equatorial radius) in meters
    WGS84_E2 = 0.00669437999014  # First eccentricity squared
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the geometry processor.
        
        Args:
            data_dir: Base directory containing downloaded files
        """
        self.data_dir = Path(data_dir).resolve()
        self.kdtree = None
        self.cloud_pixels_ecef = None
        self.cloud_pixels_list = None
        
        logger.info(f"📐 Geometry processor initialized")
        logger.info(f"   Data directory: {self.data_dir}")
    
    def convert_to_ecef(self, lat: np.ndarray, lon: np.ndarray, 
                       alt: np.ndarray = None) -> np.ndarray:
        """
        Convert geodetic coordinates to ECEF (Earth-Centered Earth-Fixed).
        
        Uses WGS84 ellipsoid parameters for accurate Earth shape modeling.
        
        Args:
            lat: Latitude in degrees (scalar or array)
            lon: Longitude in degrees (scalar or array)
            alt: Altitude in meters (default: 0 for sea level)
        
        Returns:
            Array of shape (N, 3) with ECEF coordinates [x, y, z] in meters
        """
        # Convert to numpy arrays
        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)
        if alt is None:
            alt = np.zeros_like(lat)
        else:
            alt = np.atleast_1d(alt)
        
        # Convert to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Calculate prime vertical radius of curvature
        N = self.WGS84_A / np.sqrt(1 - self.WGS84_E2 * np.sin(lat_rad)**2)
        
        # ECEF coordinates
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - self.WGS84_E2) + alt) * np.sin(lat_rad)
        
        # Stack into (N, 3) array
        ecef = np.column_stack([x, y, z])
        
        return ecef
    
    def ecef_to_geodetic(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert ECEF coordinates to geodetic (lat, lon, alt).
        
        Used for validation and round-trip testing.
        
        Args:
            x, y, z: ECEF coordinates in meters
        
        Returns:
            Tuple of (latitude, longitude, altitude)
        """
        # Longitude is straightforward
        lon = np.arctan2(y, x)
        
        # Iterative calculation for latitude and altitude
        p = np.sqrt(x**2 + y**2)
        lat = np.arctan2(z, p * (1 - self.WGS84_E2))
        
        # Iterate to refine latitude
        for _ in range(5):
            N = self.WGS84_A / np.sqrt(1 - self.WGS84_E2 * np.sin(lat)**2)
            alt = p / np.cos(lat) - N
            lat = np.arctan2(z, p * (1 - self.WGS84_E2 * N / (N + alt)))
        
        # Final altitude calculation
        N = self.WGS84_A / np.sqrt(1 - self.WGS84_E2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N
        
        # Convert to degrees
        lat = np.degrees(lat)
        lon = np.degrees(lon)
        
        return lat, lon, alt
    
    # def integrate_myd03_geolocation(self,
    #                                 cloud_pixels_by_granule: Dict[str, List[MODISCloudPixel]],
    #                                 myd03_files: List[DownloadedFile]) -> List[MODISCloudPixel]:
    #     """
    #     Add true lat/lon from MYD03 geolocation files to cloud pixels.
        
    #     NOTE: This method is DEPRECATED as of Phase 3 updates. Phase 3 now integrates
    #     MYD03 geolocation directly during cloud mask extraction using caching.
    #     Calling this method will cause redundant HDF4 file I/O.
        
    #     Use Phase 3's extract_modis_cloud_mask() with myd03_files parameter instead,
    #     which loads each MYD03 file only once and caches the result.
        
    #     Replaces placeholder (0.0, 0.0) coordinates with actual 1 km resolution
    #     geolocation data from MYD03.
        
    #     Args:
    #         cloud_pixels_by_granule: Cloud pixels organized by granule ID
    #         myd03_files: List of MYD03 geolocation files
        
    #     Returns:
    #         Flat list of cloud pixels with updated lat/lon coordinates
    #     """
    #     logger.warning("⚠️  integrate_myd03_geolocation() is DEPRECATED - Phase 3 now handles MYD03 integration with caching")
    #     if not HDF4_AVAILABLE:
    #         logger.error("pyhdf not available - cannot read MYD03 files")
    #         return []
        
    #     logger.info(f"\n[MYD03 Integration] Adding geolocation to cloud pixels")
    #     logger.info(f"   MYD03 files: {len(myd03_files)}")
    #     logger.info(f"   Cloud granules: {len(cloud_pixels_by_granule)}")
        
    #     # Create lookup: MYD35 granule ID -> MYD03 file
    #     myd03_lookup = {}
    #     for myd03_file in myd03_files:
    #         # Extract time stamp from MYD03 granule ID
    #         # Format: MYD03.A2018291.0025.061.*.hdf
    #         import re
    #         match = re.search(r'A(\d{7})\.(\d{4})', myd03_file.granule_id)
    #         if match:
    #             time_stamp = f"{match.group(1)}.{match.group(2)}"  # e.g., "2018291.0025"
    #             myd03_lookup[time_stamp] = myd03_file
        
    #     logger.debug(f"   Built MYD03 lookup: {len(myd03_lookup)} time stamps")
        
    #     all_pixels = []
    #     updated_count = 0
    #     skipped_count = 0
        
    #     for myd35_granule_id, cloud_pixels in cloud_pixels_by_granule.items():
    #         # Extract time stamp from MYD35 granule ID
    #         match = re.search(r'A(\d{7})\.(\d{4})', myd35_granule_id)
    #         if not match:
    #             logger.warning(f"   Could not parse granule ID: {myd35_granule_id}")
    #             all_pixels.extend(cloud_pixels)
    #             skipped_count += len(cloud_pixels)
    #             continue
            
    #         time_stamp = f"{match.group(1)}.{match.group(2)}"
            
    #         # Find matching MYD03 file
    #         myd03_file = myd03_lookup.get(time_stamp)
    #         if not myd03_file:
    #             logger.warning(f"   No MYD03 file for {myd35_granule_id}")
    #             all_pixels.extend(cloud_pixels)
    #             skipped_count += len(cloud_pixels)
    #             continue
            
    #         # Read MYD03 geolocation
    #         try:
    #             hdf = SD(str(myd03_file.filepath), SDC.READ)
                
    #             lat_sds = hdf.select('Latitude')
    #             lon_sds = hdf.select('Longitude')
                
    #             latitudes = lat_sds.get()
    #             longitudes = lon_sds.get()
                
    #             logger.debug(f"   {myd35_granule_id}: MYD03 shape {latitudes.shape}")
                
    #             # Update cloud pixels
    #             for pixel in cloud_pixels:
    #                 try:
    #                     # Check bounds
    #                     if (0 <= pixel.pixel_y < latitudes.shape[0] and 
    #                         0 <= pixel.pixel_x < latitudes.shape[1]):
                            
    #                         lat = latitudes[pixel.pixel_y, pixel.pixel_x]
    #                         lon = longitudes[pixel.pixel_y, pixel.pixel_x]
                            
    #                         # Check for fill values
    #                         if -90 <= lat <= 90 and -180 <= lon <= 180:
    #                             pixel.latitude = float(lat)
    #                             pixel.longitude = float(lon)
    #                             updated_count += 1
    #                         else:
    #                             skipped_count += 1
    #                     else:
    #                         logger.debug(f"   Pixel ({pixel.pixel_y}, {pixel.pixel_x}) out of bounds")
    #                         skipped_count += 1
    #                 except Exception as e:
    #                     logger.debug(f"   Error updating pixel: {e}")
    #                     skipped_count += 1
                
    #             all_pixels.extend(cloud_pixels)
                
    #             lat_sds.endaccess()
    #             lon_sds.endaccess()
    #             hdf.end()
                
    #         except Exception as e:
    #             logger.error(f"   Failed to read MYD03 {myd03_file.filepath.name}: {e}")
    #             all_pixels.extend(cloud_pixels)
    #             skipped_count += len(cloud_pixels)
        
    #     logger.info(f"✓ Updated {updated_count} cloud pixels with MYD03 geolocation")
    #     if skipped_count > 0:
    #         logger.warning(f"  Skipped {skipped_count} pixels (out of bounds or invalid)")
        
    #     return all_pixels
    
    def build_kdtree(self, cloud_pixels=None,
                     footprints=None,
                     max_distance_km: float = 50.0,
                     cloud_lons=None,
                     cloud_lats=None,
                     cloud_flags=None) -> Tuple[Optional[cKDTree], np.ndarray]:
        """
        Build spatial KD-Tree from cloud pixel ECEF coordinates.
        
        Accepts either:
        - cloud_pixels: List of cloud pixel objects with .latitude, .longitude, .cloud_flag attributes
        - cloud_lons, cloud_lats, cloud_flags: Numpy arrays of cloud data (numpy array mode)
        
        Only includes pixels flagged as 'Cloudy' or 'Uncertain' for distance calculations (1 in numeric flags).
        Optionally filters cloud pixels to those within max_distance_km of OCO-2 footprints
        to reduce memory and improve performance (eliminates far-away edge pixels).
        
        Args:
            cloud_pixels: List of cloud pixels (object mode) OR None for array mode
            footprints: Optional list of OCO-2 footprints for spatial filtering (object mode)
            max_distance_km: Maximum distance for spatial filtering (default: 50 km)
            cloud_lons: Numpy array of cloud longitudes (array mode)
            cloud_lats: Numpy array of cloud latitudes (array mode)
            cloud_flags: Numpy array of cloud flags - 0=Uncertain, 1=Cloudy (array mode)
        
        Returns:
            Tuple of (KD-Tree, ECEF coordinate array)
        """
        if not SCIPY_AVAILABLE:
            logger.error("scipy not available - cannot build KD-Tree")
            return None, None
        
        logger.info(f"\n[KD-Tree Construction] Building spatial index")
        
        # Determine input mode: object-based or array-based
        use_arrays = cloud_lons is not None and cloud_lats is not None and cloud_flags is not None
        
        if use_arrays:
            # Array-based input mode
            logger.info(f"   Using array-based cloud data")
            logger.info(f"   Total cloud pixels: {len(cloud_lons)}")
            
            # Filter to only Cloudy and Uncertain pixels (flags: 0=Uncertain, 1=Cloudy)
            valid_flags = (cloud_flags == 0) | (cloud_flags == 1)
            cloudy_uncertain_indices = np.where(valid_flags)[0]
            
            cloudy_count = np.sum(cloud_flags == 1)
            uncertain_count = np.sum(cloud_flags == 0)
            logger.info(f"   Cloud flag distribution:")
            logger.info(f"      ✓ Cloudy: {cloudy_count}")
            logger.info(f"      ✓ Uncertain: {uncertain_count}")
            logger.info(f"   Using Cloudy + Uncertain pixels: {len(cloudy_uncertain_indices)}")
            
            if len(cloudy_uncertain_indices) == 0:
                logger.error("   No Cloudy/Uncertain pixels to build KD-Tree")
                return None, None
            
            # Filter coordinates to valid pixels
            lons = cloud_lons[cloudy_uncertain_indices]
            lats = cloud_lats[cloudy_uncertain_indices]
            
            # Spatial filtering for arrays
            if footprints is not None and len(footprints) > 0:
                logger.info(f"   Applying spatial filter ({max_distance_km} km from {len(footprints)} footprints)")
                
                fp_lats_arr = np.array([f.latitude for f in footprints])
                fp_lons_arr = np.array([f.longitude for f in footprints])
                
                buffer_deg = max_distance_km / 111.0
                lat_min = np.min(fp_lats_arr) - buffer_deg
                lat_max = np.max(fp_lats_arr) + buffer_deg
                lon_min = np.min(fp_lons_arr) - buffer_deg
                lon_max = np.max(fp_lons_arr) + buffer_deg
                
                logger.info(f"      OCO-2 bounding box: lat [{np.min(fp_lats_arr):.2f}, {np.max(fp_lats_arr):.2f}], "
                           f"lon [{np.min(fp_lons_arr):.2f}, {np.max(fp_lons_arr):.2f}]")
                logger.info(f"      Cloud filter box (±{buffer_deg:.2f}°): lat [{lat_min:.2f}, {lat_max:.2f}], "
                           f"lon [{lon_min:.2f}, {lon_max:.2f}]")
                
                in_bounds = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
                before_count = len(lons)
                lons = lons[in_bounds]
                lats = lats[in_bounds]
                after_count = len(lons)
                
                if after_count < before_count:
                    reduction_pct = 100 * (before_count - after_count) / before_count
                    logger.info(f"   ✓ Spatial filter reduced pixels: {before_count:,} → {after_count:,} "
                               f"({reduction_pct:.1f}% reduction)")
                else:
                    logger.info(f"   ✓ All pixels within bounding box: {after_count:,}")
                
                if after_count == 0:
                    logger.warning("   No cloud pixels within spatial filter bounds")
                    return None, None
            
            # Filter invalid coordinates
            valid_mask = (lats >= -90) & (lats <= 90) & (lons >= -180) & (lons <= 180)
            lons = lons[valid_mask]
            lats = lats[valid_mask]
            
            if len(lons) == 0:
                logger.error("   No valid cloud pixels to build KD-Tree")
                return None, None
            
        else:
            # Object-based input mode (legacy)
            logger.info(f"   Total cloud pixels: {len(cloud_pixels)}")
            
            from collections import Counter
            flag_counts = Counter(p.cloud_flag for p in cloud_pixels)
            logger.info(f"   Cloud flag distribution:")
            for flag, count in sorted(flag_counts.items()):
                marker = "✓" if flag in ['Cloudy', 'Uncertain'] else "○"
                logger.info(f"      {marker} {flag}: {count}")
            
            cloudy_uncertain_pixels = [p for p in cloud_pixels 
                                       if p.cloud_flag in ['Cloudy', 'Uncertain']]
            
            logger.info(f"   Using Cloudy + Uncertain pixels: {len(cloudy_uncertain_pixels)}")
            
            if len(cloudy_uncertain_pixels) == 0:
                logger.error("   No Cloudy/Uncertain pixels to build KD-Tree")
                return None, None
            
            if footprints is not None and len(footprints) > 0:
                logger.info(f"   Applying spatial filter ({max_distance_km} km from {len(footprints)} footprints)")
                
                fp_lats = [f.latitude for f in footprints]
                fp_lons = [f.longitude for f in footprints]
                
                buffer_deg = max_distance_km / 111.0
                lat_min = min(fp_lats) - buffer_deg
                lat_max = max(fp_lats) + buffer_deg
                lon_min = min(fp_lons) - buffer_deg
                lon_max = max(fp_lons) + buffer_deg
                
                logger.info(f"      OCO-2 bounding box: lat [{min(fp_lats):.2f}, {max(fp_lats):.2f}], "
                           f"lon [{min(fp_lons):.2f}, {max(fp_lons):.2f}]")
                logger.info(f"      Cloud filter box (±{buffer_deg:.2f}°): lat [{lat_min:.2f}, {lat_max:.2f}], "
                           f"lon [{lon_min:.2f}, {lon_max:.2f}]")
                
                before_count = len(cloudy_uncertain_pixels)
                cloudy_uncertain_pixels = [
                    p for p in cloudy_uncertain_pixels
                    if lat_min <= p.latitude <= lat_max and lon_min <= p.longitude <= lon_max
                ]
                after_count = len(cloudy_uncertain_pixels)
                
                if after_count < before_count:
                    reduction_pct = 100 * (before_count - after_count) / before_count
                    logger.info(f"   ✓ Spatial filter reduced pixels: {before_count:,} → {after_count:,} "
                               f"({reduction_pct:.1f}% reduction)")
                else:
                    logger.info(f"   ✓ All pixels within bounding box: {after_count:,}")
                
                if after_count == 0:
                    logger.warning("   No cloud pixels within spatial filter bounds")
                    return None, None
            
            valid_pixels = [p for p in cloudy_uncertain_pixels 
                           if -90 <= p.latitude <= 90 and -180 <= p.longitude <= 180]
            
            if len(valid_pixels) < len(cloud_pixels):
                logger.warning(f"   Filtered out {len(cloud_pixels) - len(valid_pixels)} invalid pixels")
            
            if len(valid_pixels) == 0:
                logger.error("   No valid cloud pixels to build KD-Tree")
                return None, None
            
            lats = np.array([p.latitude for p in valid_pixels])
            lons = np.array([p.longitude for p in valid_pixels])
        
        # Convert to ECEF (common for both modes)
        alts = np.zeros(len(lons))  # Sea level assumption
        cloud_ecef = self.convert_to_ecef(lats, lons, alts)
        
        logger.info(f"   ECEF array shape: {cloud_ecef.shape}")
        logger.info(f"   Building KD-Tree...")
        
        # Build KD-Tree
        tree = cKDTree(cloud_ecef, leafsize=40)
        
        logger.info(f"✓ KD-Tree built: {tree.n} points, {tree.m} dimensions")
        
        self.kdtree = tree
        self.cloud_pixels_ecef = cloud_ecef
        if not use_arrays:
            self.cloud_pixels_list = valid_pixels
        
        return tree, cloud_ecef
    
    def calculate_nearest_cloud_distances_banded(
        self,
        footprints_by_granule: tuple[np.ndarray, np.ndarray],
        band_width_deg: float = 5.0,
        band_overlap_deg: float = 1.0,
        max_distance_km: float = 30.0,
        oco2_granule_id: str = "",
        cloud_lons = None,
        cloud_lats = None,
        cloud_flags = None,
    ) -> List[CollocationResult]:
        """
        Calculate nearest cloud distances using latitude banding for performance.
        
        Accepts either:
        - cloud_pixels: List of cloud pixel objects (legacy mode)
        - cloud_lons, cloud_lats, cloud_flags: Numpy arrays (array mode)
        
        Divides footprints into fixed latitude bands and builds separate KD-Trees
        for each band. This is significantly faster for large datasets and enables
        parallel processing. Recommended for batch/year-long processing.
        
        Args:
            footprints_by_granule: OCO-2 footprints organized by granule
            cloud_pixels: List of cloud pixels (legacy mode)
            band_width_deg: Latitude band width in degrees (default: 10°)
            band_overlap_deg: Overlap buffer in degrees (default: 1°)
            max_distance_km: Maximum distance to report
            cloud_lons: Numpy array of cloud longitudes (array mode)
            cloud_lats: Numpy array of cloud latitudes (array mode)
            cloud_flags: Numpy array of cloud flags (array mode)
        
        Returns:
            List of CollocationResult objects with distances
        """
        if not SCIPY_AVAILABLE:
            logger.error("scipy not available - cannot build KD-Trees")
            return []
        
        logger.info(f"\n[Banded Distance Calculation] Using {band_width_deg}° latitude bands")
        logger.info(f"   Band overlap: ±{band_overlap_deg}°")
        logger.info(f"   Max distance cap: {max_distance_km} km")
        
        if len(footprints_by_granule[0]) == 0:
            logger.error("No footprints to process")
            return []
        
        logger.info(f"   Total footprints: {len(footprints_by_granule[0]):,}")
        
        if cloud_lons is None or cloud_lats is None or cloud_flags is None:
            raise ValueError("Cloud longitude, latitude, and flag arrays must be provided for array mode")
        
        # Array-based input mode
        logger.info(f"   Using array-based cloud data")
        # Filter to Cloudy + Uncertain (flags 0 and 1)
        valid_flags = (cloud_flags == 0) | (cloud_flags == 1)
        cloudy_uncertain_indices = np.where(valid_flags)[0]
        cloudy_uncertain_lons = cloud_lons[cloudy_uncertain_indices]
        cloudy_uncertain_lats = cloud_lats[cloudy_uncertain_indices]
        cloudy_uncertain_flags = cloud_flags[cloudy_uncertain_indices]
        
        cloudy_count = np.sum(cloud_flags == 1)
        uncertain_count = np.sum(cloud_flags == 0)
        logger.info(f"   Total cloud pixels: {len(cloud_flags):,}")
        logger.info(f"   Using Cloudy + Uncertain: {len(cloudy_uncertain_indices):,}")

        
        # Determine latitude range
        fp_lons_all = footprints_by_granule[0]
        fp_lats_all = footprints_by_granule[1]
        fp_ids_all = footprints_by_granule[2]
        fp_viewing_modes_all = footprints_by_granule[3]
        lat_min_global = min(fp_lats_all)
        lat_max_global = max(fp_lats_all)
        
        logger.info(f"   OCO-2 latitude range: [{lat_min_global:.2f}°, {lat_max_global:.2f}°]")
        
        # Create latitude bands
        bands = []
        lat = -90.0 
        while lat < 90.0:
            band_min = lat
            band_max = min(lat + band_width_deg, 90.0)
            
            # Only process if there are footprints in this band
            if np.any((fp_lats_all >= band_min) & (fp_lats_all < band_max)):
                bands.append((band_min, band_max))
            
            lat += band_width_deg
        
        logger.info(f"   Processing {len(bands)} latitude band(s)")
        
        # Process each band
        all_results = []
        total_band_time = 0
        
        for band_idx, (band_min, band_max) in enumerate(bands, 1):
            import time
            band_start = time.time()
            
            # Get footprints in this band
            fp_lats_band_mask = (fp_lats_all >= band_min) & (fp_lats_all < band_max)
            band_fp_lons = fp_lons_all[fp_lats_band_mask]
            band_fp_lats = fp_lats_all[fp_lats_band_mask]
            band_fp_ids = fp_ids_all[fp_lats_band_mask]
            band_fp_viewing_modes = fp_viewing_modes_all[fp_lats_band_mask]
            band_fp_count = len(band_fp_lons)
            
            if len(band_fp_lons) == 0:
                logger.debug(f"   Band {band_idx}/{len(bands)} [{band_min:.1f}°, {band_max:.1f}°): 0 footprints, skipping")
                continue
            
            # Get clouds in this band with overlap buffer
            cloud_lat_min = band_min - band_overlap_deg # it is okay if this goes below -90, the filtering will handle it
            cloud_lat_max = band_max + band_overlap_deg # it is okay if this goes above 90, the filtering will handle it
            
            # Array-based filtering
            in_band = (cloudy_uncertain_lats >= cloud_lat_min) & (cloudy_uncertain_lats <= cloud_lat_max)
            band_cloud_lons = cloudy_uncertain_lons[in_band]
            band_cloud_lats = cloudy_uncertain_lats[in_band]
            band_cloud_flags = cloudy_uncertain_flags[in_band]
            band_cloud_indices = cloudy_uncertain_indices[in_band]
            
            if len(band_cloud_lons) == 0:
                logger.warning(f"   Band {band_idx}/{len(bands)} [{band_min:.1f}°, {band_max:.1f}°): "
                               f"{band_fp_count:,} footprints but 0 clouds, assigning max distance")
                for i in range(band_fp_count):
                    result = CollocationResult(
                        sounding_id=band_fp_ids[i],
                        granule_id=oco2_granule_id,
                        footprint_lat=band_fp_lats[i],
                        footprint_lon=band_fp_lons[i],
                        viewing_mode=band_fp_viewing_modes[i],
                        nearest_cloud_dist_km=max_distance_km,
                        nearest_cloud_lat=float('nan'),
                        nearest_cloud_lon=float('nan'),
                        cloud_classification='NoCloud',
                    )
                    all_results.append(result)
                band_time = time.time() - band_start
                total_band_time += band_time
                logger.info(f"      ✓ Band completed in {band_time:.2f}s ({band_fp_count/band_time:.0f} fps/s)")
                continue
                        
            logger.info(f"   Band {band_idx}/{len(bands)} [{band_min:.1f}°, {band_max:.1f}°): "
                        f"{band_fp_count:,} footprints, {len(band_cloud_lons):,} clouds")
            
            # Convert clouds to ECEF for this band
            band_cloud_alts = np.zeros(len(band_cloud_lons))
            cloud_ecef = self.convert_to_ecef(band_cloud_lats, band_cloud_lons, band_cloud_alts)
                        
            band_tree = cKDTree(cloud_ecef, leafsize=40)
            
            # Convert footprints to ECEF
            band_fp_alts = np.zeros(band_fp_count)
            fp_ecef = self.convert_to_ecef(band_fp_lats, band_fp_lons, band_fp_alts)
            
            # Query KD-Tree
            distances, indices = band_tree.query(fp_ecef, k=1)
            
            # Build results
            for i in range(band_fp_count):
                footprint_lon = band_fp_lons[i]
                footprint_lat = band_fp_lats[i]
                footprint_id = band_fp_ids[i]
                footprint_viewing_mode = band_fp_viewing_modes[i]
                distance_km = distances[i] / 1000.0
                
                # Cap distance
                if distance_km > max_distance_km:
                    distance_km = max_distance_km
                
                cloud_idx = indices[i]
                
                # Array mode
                nearest_cloud_lat = float(band_cloud_lats[cloud_idx])
                nearest_cloud_lon = float(band_cloud_lons[cloud_idx])
                cloud_flag_val = int(band_cloud_flags[cloud_idx])
                cloud_class = 'Cloudy' if cloud_flag_val == 1 else 'Uncertain'
                
                result = CollocationResult(
                    sounding_id=footprint_id,
                    granule_id=oco2_granule_id,
                    footprint_lat=footprint_lat,
                    footprint_lon=footprint_lon,
                    viewing_mode=footprint_viewing_mode,
                    nearest_cloud_dist_km=distance_km,
                    nearest_cloud_lat=nearest_cloud_lat,
                    nearest_cloud_lon=nearest_cloud_lon,
                    cloud_classification=cloud_class,
                )
                
                all_results.append(result)
            
            band_time = time.time() - band_start
            total_band_time += band_time
            logger.info(f"      ✓ Band completed in {band_time:.2f}s ({band_fp_count/band_time:.0f} fps/s)")
        
        logger.info(f"✓ Total results: {len(all_results):,} in {total_band_time:.2f}s")
        logger.info(f"   Overall throughput: {len(all_results)/total_band_time:.0f} footprints/second")
        
        return all_results
        
    def export_results_hdf5(self, results: List[CollocationResult], 
                           output_path: Path, metadata: Dict = None):
        """
        Export collocation results to HDF5 format.
        
        Args:
            results: List of CollocationResult objects
            output_path: Path to output HDF5 file
            metadata: Optional metadata dictionary
        """
        logger.info(f"\n[Export HDF5] Saving {len(results)} results to {output_path.name}")
        
        with h5py.File(output_path, 'w') as f:
            # Create datasets
            f.create_dataset('sounding_id', data=[r.sounding_id for r in results])
            f.create_dataset('granule_id', data=[r.granule_id.encode() for r in results])
            f.create_dataset('latitude', data=[r.footprint_lat for r in results])
            f.create_dataset('longitude', data=[r.footprint_lon for r in results])
            f.create_dataset('viewing_mode', data=[r.viewing_mode.encode() for r in results])
            f.create_dataset('nearest_cloud_distance_km', data=[r.nearest_cloud_dist_km for r in results])
            f.create_dataset('cloud_latitude', data=[r.nearest_cloud_lat for r in results])
            f.create_dataset('cloud_longitude', data=[r.nearest_cloud_lon for r in results])
            f.create_dataset('cloud_classification', data=[r.cloud_classification.encode() for r in results])
            
            # Add metadata
            f.attrs['num_soundings'] = len(results)
            f.attrs['phase'] = 'Phase 4 - Geometry'
            f.attrs['creation_time'] = datetime.utcnow().isoformat()
            
            if metadata:
                for key, value in metadata.items():
                    f.attrs[key] = value
        
        logger.info(f"✓ HDF5 file saved: {output_path}")
    
    def export_results_csv(self, results: List[CollocationResult], output_path: Path):
        """
        Export collocation results to CSV format.
        
        Args:
            results: List of CollocationResult objects
            output_path: Path to output CSV file
        """
        import csv
        
        logger.info(f"\n[Export CSV] Saving {len(results)} results to {output_path.name}")
        
        with open(output_path, 'w', newline='') as f:
            if results:
                fieldnames = list(asdict(results[0]).keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    writer.writerow(asdict(result))
        
        logger.info(f"✓ CSV file saved: {output_path}")
    
    def get_statistics(self, results: List[CollocationResult]) -> Dict:
        """
        Calculate summary statistics for collocation results.
        
        Args:
            results: List of CollocationResult objects
        
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}
        
        distances = [r.nearest_cloud_dist_km for r in results]
        
        # Count by cloud classification
        cloudy_count = sum(1 for r in results if r.cloud_classification == 'Cloudy')
        uncertain_count = sum(1 for r in results if r.cloud_classification == 'Uncertain')
        
        # Count by distance bins
        dist_0_2 = sum(1 for d in distances if d <= 2)
        dist_2_5 = sum(1 for d in distances if 2 < d <= 5)
        dist_5_10 = sum(1 for d in distances if 5 < d)
        dist_10_20 = sum(1 for d in distances if 10 < d <= 20)
        dist_20_above = sum(1 for d in distances if d > 20)
        
        stats = {
            'total_soundings': len(results),
            'distance_km': {
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'mean': float(np.mean(distances)),
                'median': float(np.median(distances)),
                'std': float(np.std(distances)),
            },
            'cloud_classification': {
                'cloudy': cloudy_count,
                'uncertain': uncertain_count,
            },
            'distance_distribution': {
                '0-2_km': dist_0_2,
                '2-5_km': dist_2_5,
                '5-10_km': dist_5_10,
                '10-20_km': dist_10_20,
                '20+_km': dist_20_above,
            }
        }
        
        return stats
      
    def filter_footprints_by_bounds(self, footprints: Dict, bounds: Tuple) -> Dict:
        """
        Filter OCO-2 footprints by geographic bounding box.
        
        Args:
            footprints: Dictionary of footprints {sounding_id: OCO2Footprint}
            bounds: Tuple of (min_lat, min_lon, max_lat, max_lon)
        
        Returns:
            Filtered dictionary of footprints within bounds
        """
        min_lat, min_lon, max_lat, max_lon = bounds
        
        filtered = {}
        for sounding_id, fp in footprints.items():
            if (min_lat <= fp.latitude <= max_lat and
                min_lon <= fp.longitude <= max_lon and
                -90 <= fp.latitude <= 90 and -180 <= fp.longitude <= 180):
                filtered[sounding_id] = fp
        
        return filtered
        
    def visualize_kdtree_spatial_range(self,
                                      results: List[CollocationResult],
                                      num_cloudy: int,
                                      num_uncertain: int,
                                      output_path: Path,
                                      max_distance: float = 50.0,
                                      dpi: int = 200) -> Optional[Path]:
        """
        Create 3-panel KD-Tree spatial filtering visualization (no cloud plotting for memory efficiency).
        
        Args:
            results: List of collocation results
            num_cloudy: Total count of cloudy pixels
            num_uncertain: Total count of uncertain pixels
            output_path: Path to save visualization
            max_distance: Maximum distance for colorbar
            dpi: DPI for output image
        
        Returns:
            Path to saved visualization or None if failed
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib not available")
            return None
        
        # Extract footprint data
        fp_lats = [r.footprint_lat for r in results]
        fp_lons = [np.where(r.footprint_lon > 180, r.footprint_lon - 360, r.footprint_lon) for r in results]
        fp_distances = [r.nearest_cloud_dist_km for r in results]
        
        distance_threshold = 10.0  # km
        fp_lons_lt_threshold = [lon for lon, d in zip(fp_lons, fp_distances) if d <= distance_threshold]
        fp_lats_lt_threshold = [lat for lat, d in zip(fp_lats, fp_distances) if d <= distance_threshold]
        fp_lons_gt_threshold = [lon for lon, d in zip(fp_lons, fp_distances) if d > distance_threshold]
        fp_lats_gt_threshold = [lat for lat, d in zip(fp_lats, fp_distances) if d > distance_threshold]
        
        nc_lats = [r.nearest_cloud_lat for r in results]
        nc_lons = [np.where(r.nearest_cloud_lon > 180, r.nearest_cloud_lon - 360, r.nearest_cloud_lon) for r in results]
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=dpi)
        
        # Top Left: Footprints colored by distance threshold
        ax = axes[0, 0]
        ax.plot(fp_lons_lt_threshold, fp_lats_lt_threshold, 'go', markersize=4, alpha=0.8,
               label=f'Footprints ≤ {distance_threshold} km', zorder=6)
        ax.plot(fp_lons_gt_threshold, fp_lats_gt_threshold, 'ro', markersize=4, alpha=0.8,
               label=f'Footprints > {distance_threshold} km', zorder=6)
        
        ax.set_xlabel('Longitude (°)', fontsize=12)
        ax.set_ylabel('Latitude (°)', fontsize=12)
        ax.set_title(f'OCO-2 Footprints by Distance Threshold\n{len(fp_lats):,} total footprints',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Top Right: Footprints colored by distance
        ax = axes[0, 1]
        scatter = ax.scatter(fp_lons, fp_lats, c=fp_distances, cmap='jet', s=15,
                           edgecolors=None, linewidths=0.3,
                           vmin=0, vmax=max_distance, zorder=5)
        
        fig.colorbar(scatter, ax=ax, label='Nearest Cloud Distance (km)')
        ax.set_xlabel('Longitude (°)', fontsize=12)
        ax.set_ylabel('Latitude (°)', fontsize=12)
        ax.set_title(f'Footprint-Cloud Distances\nMean: {np.mean(fp_distances):.2f} km, Median: {np.median(fp_distances):.2f} km',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Bottom Left: cloud distance histogram
        ax = axes[1, 0]
        ax.hist(fp_distances, bins=30, range=(0, max_distance), color='steelblue', edgecolor='black')
        ax.set_xlabel('Distance to Nearest Cloud (km)', fontsize=12)
        ax.set_ylabel('Number of Footprints', fontsize=12)
        ax.set_title('Distribution of Footprint-Cloud Distances', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.axis('off')
        fp_distances_array = np.array(fp_distances)
        ax.text(0.5, 0.5, f'Total Footprints: {len(fp_lats):,}\n'
                            f'Footprints ≤ {distance_threshold} km: {len(fp_lats_lt_threshold):,}\n'
                            f'Footprints > {distance_threshold} km: {len(fp_lats_gt_threshold):,}\n'
                            f'Footprint ≤ 4 km percentage: {100 *  np.sum(fp_distances_array <= 4) / len(fp_lats):.3f}%\n'
                            f'Footprint ≤ 10 km percentage: {100 * np.sum(fp_distances_array <= 10) / len(fp_lats):.3f}%\n'
                            f'Footprint ≤ 15 km percentage: {100 * np.sum(fp_distances_array <= 15) / len(fp_lats):.3f}%\n'
                            f'\n'
                            f'Cloud Pixels (total):  {num_cloudy + num_uncertain:,}\n'
                            f'  • Cloudy: {num_cloudy:,}\n'
                            f'  • Uncertain: {num_uncertain:,}\n'
                            f'\n'
                            f'Distance Statistics:\n'
                            f'  • Mean: {np.mean(fp_distances):.2f} km\n'
                            f'  • Median: {np.median(fp_distances):.2f} km'
                            ,
                 fontsize=12, ha='center', va='center', wrap=True)
        
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    def visualize_latband_distance(self,
                                   results: List[CollocationResult],
                                   cloud_lons: np.ndarray,
                                   cloud_lats: np.ndarray,
                                   cloud_flags: np.ndarray,
                                   output_dir: Path,
                                   max_distance: float = 50.0,
                                   lat_band_size: float = 10.0,
                                   max_clouds_per_band: int = 50000,
                                   dpi: int = 200) -> List[Path]:
        """
        Create per-latitude-band distance visualizations showing cloud locations.
        
        Args:
            results: List of collocation results
            cloud_lons: All cloud longitudes (list)
            cloud_lats: All cloud latitudes (list)
            cloud_flags: All cloud flags (list)
            output_dir: Directory to save visualizations
            max_distance: Maximum distance for colorbar
            lat_band_size: Latitude band size in degrees
            max_clouds_per_band: Maximum clouds to plot per band (sample if exceeded)
            dpi: DPI for output images
        
        Returns:
            List of paths to saved visualizations
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib not available")
            return []
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []
        
        lat_bands = np.arange(-90, 90.1, lat_band_size)
        lat_band_keys = [f"{lat_bands[i]:+.0f} to {lat_bands[i+1]:+.0f}" for i in range(len(lat_bands)-1)]
        
        fp_lons = []
        fp_lats = []
        fp_distances = []
        for r in results:
            fp_lats.append(r.footprint_lat)
            fp_lons.append(r.footprint_lon)
            fp_distances.append(r.nearest_cloud_dist_km)
            
        fp_lons = np.array(fp_lons)
        fp_lats = np.array(fp_lats)
        fp_distances = np.array(fp_distances)  
        
        # Group clouds and footprints by band
        clouds_by_band = {}
        footprints_by_band = {}
        for i in range(len(lat_band_keys)):
            lat_band_key = lat_band_keys[i]
            cld_lat_band_mask = (cloud_lats >= lat_bands[i]) & (cloud_lats < lat_bands[i+1])
            fp_lat_band_mask = (fp_lats >= lat_bands[i]) & (fp_lats < lat_bands[i+1])
            

            clouds_by_band[lat_band_key] = {'lons': [], 'lats': [], 'flags': []}
            footprints_by_band[lat_band_key] = {'lons': [], 'lats': [], 'distances': []}
            if not np.any(cld_lat_band_mask) and not any(fp_lat_band_mask):
                continue
            
            clouds_by_band[lat_band_key]['lons'].append(cloud_lons[cld_lat_band_mask])
            clouds_by_band[lat_band_key]['lats'].append(cloud_lats[cld_lat_band_mask])
            clouds_by_band[lat_band_key]['flags'].append(cloud_flags[cld_lat_band_mask])
            
            footprints_by_band[lat_band_key]['lons'].append(fp_lons[fp_lat_band_mask])
            footprints_by_band[lat_band_key]['lats'].append(fp_lats[fp_lat_band_mask])
            footprints_by_band[lat_band_key]['distances'].append(fp_distances[fp_lat_band_mask])
            
        lat_band_keys = sorted(set(clouds_by_band.keys()) | set(footprints_by_band.keys()))
        
        for lat_band_key in lat_band_keys:
            band_clouds = clouds_by_band.get(lat_band_key, {'lons': [], 'lats': [], 'flags': []})
            band_results = footprints_by_band.get(lat_band_key, {'lons': [], 'lats': [], 'distances': []})
            
            if not band_clouds['lons'] and not band_results['lons']:
                continue
            
            fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)
            cross_dateline = False
            xtick_interval = 1
            if len(band_clouds['lons']) > 0:
                cloud_lons_band = np.array(band_clouds['lons'])
                cloud_lats_band = np.array(band_clouds['lats'])
                cloud_flags_band = np.array(band_clouds['flags'])
                       
                if cloud_lons_band.min() < -90 and cloud_lons_band.max() > 90:
                    cross_dateline = True
                    cloud_lons_band = cloud_lons_band % 360
                    
                # Sample if too many
                if len(cloud_lons_band) > max_clouds_per_band:
                    indices = np.random.choice(len(cloud_lons_band), max_clouds_per_band, replace=False)
                    cloud_lons_band = cloud_lons_band[indices]
                    cloud_lats_band = cloud_lats_band[indices]
                    cloud_flags_band = cloud_flags_band[indices]
                
                cloudy_mask = cloud_flags_band == 1
                uncertain_mask = cloud_flags_band == 0
                
                # Count totals
                total_cloudy = np.sum(np.array(band_clouds['flags']) == 1)
                total_uncertain = np.sum(np.array(band_clouds['flags']) == 0)
                
                ax.scatter(cloud_lons_band[cloudy_mask], cloud_lats_band[cloudy_mask],
                          c='lightgray', s=8, alpha=0.35, label=f'Cloudy ({total_cloudy:,})',
                          rasterized=True)
                ax.scatter(cloud_lons_band[uncertain_mask], cloud_lats_band[uncertain_mask],
                          c='darkgray', s=8, alpha=0.35, label=f'Uncertain ({total_uncertain:,})',
                          rasterized=True)
                
                if (cloud_lons_band.max() - cloud_lons_band.min()) > 4:
                    xtick_interval = 2
                elif (cloud_lons_band.max() - cloud_lons_band.min()) > 10:
                    xtick_interval = 5
                elif (cloud_lons_band.max() - cloud_lons_band.min()) > 40:
                    xtick_interval = 10
                elif (cloud_lons_band.max() - cloud_lons_band.min()) > 90:
                    xtick_interval = 30
                xmin = np.floor(cloud_lons_band.min() / xtick_interval) * xtick_interval
                xmax = np.ceil(cloud_lons_band.max() / xtick_interval) * xtick_interval
                xticks = np.arange(xmin, xmax + 1, xtick_interval)
                if cross_dateline:
                    xtick_labels = [f"{(x-360) if x > 180 else x:.0f}°" for x in xticks]
                else:
                    xtick_labels = [f"{x:.0f}°" for x in xticks]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtick_labels, fontsize=10)
            
            if band_results:
                fp_lons_band = np.array(band_results['lons'])
                fp_lats_band = np.array(band_results['lats'])
                fp_distances_band = np.array(band_results['distances'])
                
                if cross_dateline:
                    fp_lons_band = fp_lons_band % 360
                
                
                
                scatter = ax.scatter(fp_lons_band, fp_lats_band, c=fp_distances_band, cmap='jet', 
                                     s=20, edgecolors='black', linewidths=0.2,
                                    vmin=0, vmax=max_distance, zorder=5,
                                    label=f'Footprints ({len(fp_lons_band):,})')
                fig.colorbar(scatter, ax=ax, label='Nearest Cloud Distance (km)')
            
                if len(band_clouds['lons']) == 0:
                    if (fp_lons_band.max() - fp_lons_band.min()) > 4:
                        xtick_interval = 2
                    elif (fp_lons_band.max() - fp_lons_band.min()) > 10:
                        xtick_interval = 5
                    elif (fp_lons_band.max() - fp_lons_band.min()) > 40:
                        xtick_interval = 10
                    elif (fp_lons_band.max() - fp_lons_band.min()) > 90:
                        xtick_interval = 30
                    
                    xmin = np.floor(fp_lons_band.min() / xtick_interval) * xtick_interval
                    xmax = np.ceil(fp_lons_band.max() / xtick_interval) * xtick_interval
                    xticks = np.arange(xmin, xmax + 0.1, xtick_interval)
                    xtick_labels = [f"{x:.0f}°" for x in xticks]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xtick_labels, fontsize=10)
            
            ax.set_xlabel('Longitude (°)', fontsize=12)
            ax.set_ylabel('Latitude (°)', fontsize=12)
            ax.set_title(f'Footprint Cloud Distance by Latitude Band\nBand {lat_band_key}°',
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)

            band_path = output_dir / f'latband_distance_{lat_band_key}.png'
            fig.savefig(band_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            output_paths.append(band_path)
        
        return output_paths
