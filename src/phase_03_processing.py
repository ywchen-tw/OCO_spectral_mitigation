"""
Phase 3: Spatial and Bitmask Processing
========================================

This module extracts and processes OCO-2 footprints and MODIS cloud masks
to prepare datasets for geometric collocation.

Key Functions:
- extract_oco2_footprints: Extract footprint coordinates and timing from L2 Lite files
- extract_modis_cloud_mask: Parse 48-bit cloud mask and unpack bits 1-2
- match_temporal_windows: Align OCO-2 to MODIS granules with drift correction

OCO-2 Processing:
- L2 Lite NetCDF4 files contain: footprint_latitude, footprint_longitude, sounding_time
- Index by sounding_id for later matching
- Extract viewing mode (GL = Glint, ND = Nadir)

MODIS Processing:
- MYD35_L2 HDF4 cloud mask: 48-bit per pixel, organized as Byte 0, Byte 1, Byte 2
- Byte 0 bits 1-2 indicate: 00=Cloudy, 01=Uncertain, 10=Probably Clear, 11=Clear
- MYD03 provides 1 km geolocation: latitude, longitude, height
"""

import logging
import h5py
import numpy as np
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import struct
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import json
import pickle
import json

# Configure logging first
logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import netCDF4 as nc
    NC4_AVAILABLE = True
except ImportError:
    NC4_AVAILABLE = False

try:
    from pyhdf.SD import SD, SDC
    HDF4_AVAILABLE = True
except ImportError:
    HDF4_AVAILABLE = False

# Handle both package and direct imports
try:
    from .config import Config
    from .phase_01_metadata import OCO2Granule
    from .phase_02_ingestion import DownloadedFile
except ImportError:
    from config import Config
    from phase_01_metadata import OCO2Granule
    from phase_02_ingestion import DownloadedFile

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class OCO2Footprint:
    """Represents a single OCO-2 sounding footprint."""
    sounding_id: int
    granule_id: str
    short_orbit_id: str
    latitude: float
    longitude: float
    sounding_time: datetime
    viewing_mode: str  # 'GL' (Glint) or 'ND' (Nadir)
    
    def __repr__(self):
        return (f"OCO2Footprint(sounding_id={self.sounding_id}, "
                f"lat={self.latitude:.4f}, lon={self.longitude:.4f}, "
                f"{self.sounding_time.isoformat()})")

@dataclass
class MODISCloudPixel:
    """Represents a single MODIS cloud pixel with geolocation."""
    granule_id: str
    latitude: float
    longitude: float
    cloud_flag: str  # 'Cloudy' or 'Uncertain'
    observation_time: datetime
    pixel_x: int = None
    pixel_y: int = None
    
    def __repr__(self):
        return (f"MODISCloudPixel(granule={self.granule_id}, "
                f"lat={self.latitude:.4f}, lon={self.longitude:.4f}, "
                f"{self.cloud_flag})")

@dataclass
class MODISCloudMask:
    """Array-based sparse storage format for MODIS cloud mask - efficient vectorized operations."""
    granule_id: str
    observation_time: datetime
    lon: np.ndarray         # Shape: (N,) float32 - longitude values
    lat: np.ndarray         # Shape: (N,) float32 - latitude values
    cloud_flag: np.ndarray  # Shape: (N,) uint8 - 0=Uncertain, 1=Cloudy
    
    def get_pixel_count(self):
        """Return total number of cloudy + uncertain pixels."""
        return len(self.lon) if self.lon is not None else (len(self.pixels) if self.pixels else 0)
    
    def get_coordinates(self):
        """Return (lon, lat) coordinates as Nx2 array for KD-tree operations."""
        return np.column_stack([self.lon, self.lat])
    
    def get_cloudy_mask(self):
        """Return boolean mask for cloudy pixels (cloud_flag == 1)."""
        return self.cloud_flag == 1
    
    def get_uncertain_mask(self):
        """Return boolean mask for uncertain pixels (cloud_flag == 0)."""
        return self.cloud_flag == 0
    

class SpatialProcessor:
    """Processes OCO-2 footprints and MODIS cloud masks for collocation."""
    
    # MODIS cloud mask flag values from Byte 1, bits 1-2
    CLOUD_CLEAR = 0b11      # Bits 1-2 = 11 (binary)
    CLOUD_PROBABLY_CLEAR = 0b10
    CLOUD_UNCERTAIN = 0b01
    CLOUD_CLOUDY = 0b00
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the spatial processor.
        
        Args:
            data_dir: Base directory containing downloaded files
        """
        self.data_dir = Path(data_dir).resolve()
        self.oco2_dir = self.data_dir / "OCO2"
        self.modis_dir = self.data_dir / "MODIS"
        self.processing_dir = self.data_dir / "processing"
        
        # Cache for MYD03 geolocation data to avoid re-reading same files
        self._myd03_cache = {}
        
        logger.info(f"📁 Spatial processor initialized")
        logger.info(f"   OCO2 data: {self.oco2_dir}")
        logger.info(f"   MODIS data: {self.modis_dir}")
        logger.info(f"   Processing cache: {self.processing_dir}")
    
    def _get_cache_path(self, date: datetime, granule_id: str, cache_name: str) -> Path:
        """
        Get cache file path for a processing step.
        
        Args:
            date: Target date
            granule_id: Granule identifier (can be short '22845a' or full filename)
            cache_name: Descriptive cache file name (e.g., 'oco2_footprints', 'myd35_2018291.1230')
        
        Returns:
            Path to cache file: data/processing/YYYY/DOY/short_orbit_id/cache_name.pkl
        """
        year = date.strftime('%Y')
        doy = date.timetuple().tm_yday
        
        # Extract short orbit ID from granule_id
        # OCO-2 format: oco2_L1bScGL_22845a_181018_B11006r_220921185957.h5 → 22845a
        # MODIS format: MYD35_L2.A2018291.0130.061.2018291164841.hdf → A2018291.0130
        short_id = self._extract_short_orbit_id(granule_id)
        
        cache_dir = self.processing_dir / year / f"{doy:03d}" / short_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{cache_name}.pkl"
    
    def _extract_short_orbit_id(self, granule_id: str) -> str:
        """
        Extract a short, clean identifier from a granule ID.
        
        Args:
            granule_id: Full granule identifier (filename or short ID)
        
        Returns:
            Short orbit ID (e.g., '22845a' for OCO-2, 'A2018291.0130' for MODIS)
        """
        # If already short (no file extension), return as-is
        if '.' not in granule_id and len(granule_id) < 20:
            return granule_id
        
        # OCO-2 L1B/L2 format: oco2_L1bScGL_22845a_181018_B11006r_220921185957.h5
        # Include viewing mode so GL and ND orbits with the same orbit_id get separate
        # cache directories (e.g. "22845a_GL", "22845a_ND", "22845a_TG").
        if granule_id.startswith('oco2_'):
            parts = granule_id.split('_')
            if len(parts) >= 3:
                orbit_id = parts[2]  # e.g. "22845a"
                product_str = parts[1].upper() if len(parts) > 1 else ''  # e.g. "L1BSCGL"
                if 'GL' in product_str:
                    return f"{orbit_id}_GL"
                elif 'ND' in product_str:
                    return f"{orbit_id}_ND"
                elif 'TG' in product_str:
                    return f"{orbit_id}_TG"
                return orbit_id  # fallback for unrecognised modes
        
        # MODIS format: MYD35_L2.A2018291.0130.061.2018291164841.hdf
        # Extract time-based ID: A2018291.0130
        match = re.search(r'(A\d{7}\.\d{4})', granule_id)
        if match:
            return match.group(1)
        
        # Fallback: use the granule_id as-is (may be long)
        return granule_id
    
    def _save_cached_result(self, path: Path, data: any):
        """Save data to cache file using pickle."""
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"💾 Cached: {path}")
        except Exception as e:
            logger.warning(f"Failed to cache result to {path}: {e}")
    
    def _load_cached_result(self, path: Path) -> Optional[any]:
        """Load data from cache file."""
        if not path.exists():
            return None
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"📂 Loaded from cache: {path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache from {path}: {e}")
            return None
    
    def extract_oco2_footprints(self,
                               oco2_files: List[DownloadedFile],
                               viewing_mode: str = 'GL',
                               use_cache: bool = True) -> Dict[int, OCO2Footprint]:
        """
        Extract OCO-2 footprint coordinates and timing from L1B files.
        
        Filters footprints to only include sounding_ids that have successful retrievals
        in the L2 Lite file (quality-controlled soundings).
        
        Workflow:
        1. Extract sounding_id list from L2 Lite file (which soundings have XCO2 retrievals)
        2. Extract footprints (lat/lon/time) from L1B Science file
        3. Filter L1B footprints to only include sounding_ids present in Lite file
        
        Args:
            oco2_files: List of downloaded OCO2 file objects (from Phase 2)
            viewing_mode: Filter by viewing mode ('GL'=Glint, 'ND'=Nadir, 'All'=both)
            use_cache: If True, load from cache if available and save results
        
        Returns:
            Dictionary indexed by sounding_id containing OCO2Footprint objects
        """
        footprints = {}
        
        # Separate L2 Lite and L1B files
        lite_files = [f for f in oco2_files if 'L2_Lite' in f.product_type]
        l1b_files = [f for f in oco2_files if 'L1B' in f.product_type]
        
        logger.info(f"\n[Processing] Extracting OCO-2 footprints")
        logger.info(f"  L2 Lite files: {len(lite_files)} (for sounding_id filtering)")
        logger.info(f"  L1B Science files: {len(l1b_files)} (for geolocation)")
        
        # Step 1: Extract sounding_ids from L2 Lite file (quality-controlled soundings)
        valid_sounding_ids = set()
        if lite_files:
            for file_obj in lite_files:
                lite_sounding_ids = self._extract_sounding_ids_from_lite(
                    file_obj.filepath,
                    use_cache=use_cache
                )
                valid_sounding_ids.update(lite_sounding_ids)
                logger.info(f"    L2 Lite: {len(lite_sounding_ids)} valid sounding_ids")
        
        # Step 2: Extract footprints from L1B Science files
        if l1b_files:
            for file_obj in l1b_files:
                granule_id = file_obj.granule_id
                
                # Check cache first
                cache_loaded = False
                if use_cache:
                    # Parse date from filename
                    match = re.search(r'(\d{6})_B', file_obj.filepath.name)
                    if match:
                        date_str = match.group(1)
                        year = 2000 + int(date_str[0:2])
                        month = int(date_str[2:4])
                        day = int(date_str[4:6])
                        file_date = datetime(year, month, day)
                        cache_path = self._get_cache_path(file_date, granule_id, 'footprints')
                        cached_data = self._load_cached_result(cache_path)
                        if cached_data is not None:
                            footprints.update(cached_data)
                            logger.info(f"    📂 Loaded from cache: {granule_id} ({len(cached_data)} footprints)")
                            cache_loaded = True
                
                if not cache_loaded:
                    # Extract footprints from L1B file
                    l1b_footprints = self._extract_footprints_from_l1b(
                        file_obj.filepath,
                        granule_id,
                        viewing_mode
                    )
                    
                    # Filter to only valid sounding_ids (those in L2 Lite)
                    if valid_sounding_ids:
                        filtered_footprints = {
                            sid: fp for sid, fp in l1b_footprints.items()
                            if sid in valid_sounding_ids
                        }
                        logger.info(f"    L1B: {len(l1b_footprints)} total → {len(filtered_footprints)} with L2 Lite retrievals")
                        l1b_footprints = filtered_footprints
                    else:
                        logger.info(f"    L1B: {len(l1b_footprints)} footprints (no Lite filtering)")
                    
                    footprints.update(l1b_footprints)
                    
                    # Save to cache
                    if use_cache and l1b_footprints:
                        match = re.search(r'(\d{6})_B', file_obj.filepath.name)
                        if match:
                            date_str = match.group(1)
                            year = 2000 + int(date_str[0:2])
                            month = int(date_str[2:4])
                            day = int(date_str[4:6])
                            file_date = datetime(year, month, day)
                            cache_path = self._get_cache_path(file_date, granule_id, 'footprints')
                            self._save_cached_result(cache_path, l1b_footprints)
        
        logger.info(f"✓ Total footprints extracted: {len(footprints)}")
        return footprints
    
    def filter_footprints_by_date(self,
                                  footprints: Dict[int, OCO2Footprint],
                                  target_date: datetime,
                                  buffer_hours: int = 12) -> Dict[int, OCO2Footprint]:
        """
        Filter footprints to only include those from a specific date.
        
        Useful when L2 Lite files contain multiple dates (e.g., yearly aggregates).
        
        Args:
            footprints: Dictionary of footprints indexed by sounding_id
            target_date: Target date to filter for
            buffer_hours: Hours before/after target date to include (default: ±12 hours)
        
        Returns:
            Filtered dictionary containing only footprints from target date
        """
        # Create date range
        start_time = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = start_time - timedelta(hours=buffer_hours)
        end_time = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        end_time = end_time + timedelta(hours=buffer_hours)
        
        # Filter footprints
        filtered = {
            sid: fp for sid, fp in footprints.items()
            if start_time <= fp.sounding_time <= end_time
        }
        
        logger.info(f"Filtered {len(footprints)} footprints to {len(filtered)} within date range")
        logger.debug(f"  Date range: {start_time} to {end_time}")
        
        return filtered
    
    def group_footprints_by_granule(self,
                                    footprints: Dict[int, OCO2Footprint]) -> Dict[str, List[OCO2Footprint]]:
        """
        Organize OCO-2 footprints by granule ID.
        
        Useful for Phase 04 processing where each OCO-2 granule is processed separately
        for nearest-cloud distance calculations.
        
        Args:
            footprints: Dictionary of footprints indexed by sounding_id
        
        Returns:
            Dictionary indexed by granule_id containing lists of footprints
        """
        footprints_by_granule = {}
        
        for sounding_id, footprint in footprints.items():
            granule_id = footprint.granule_id
            if granule_id not in footprints_by_granule:
                footprints_by_granule[granule_id] = []
            footprints_by_granule[granule_id].append(footprint)
        
        logger.info(f"Organized {len(footprints)} footprints into {len(footprints_by_granule)} granule(s)")
        for granule_id, granule_footprints in footprints_by_granule.items():
            logger.debug(f"  {granule_id}: {len(granule_footprints)} footprints")
        
        return footprints_by_granule
    
    def get_granule_statistics(self,
                               footprints_by_granule: Dict[str, List[OCO2Footprint]]) -> Dict[str, Dict]:
        """
        Calculate statistics for each OCO-2 granule.
        
        Args:
            footprints_by_granule: Dictionary of footprints organized by granule
        
        Returns:
            Dictionary with statistics for each granule
        """
        stats = {}
        
        for granule_id, footprints in footprints_by_granule.items():
            if not footprints:
                continue
            
            lats = [fp.latitude for fp in footprints]
            lons = [fp.longitude for fp in footprints]
            times = [fp.sounding_time for fp in footprints]
            
            stats[granule_id] = {
                'count': len(footprints),
                'lat_range': (min(lats), max(lats)),
                'lon_range': (min(lons), max(lons)),
                'time_range': (min(times), max(times)),
                'viewing_mode': footprints[0].viewing_mode if footprints else 'Unknown'
            }
        
        return stats
    
    def combine_OCO_fp_cloud_masks_by_granule(self,
                                              granule_cloud_masks: Dict[str, MODISCloudMask],
                                              granule_footprints: Dict[int, OCO2Footprint],
                                              oco2_granule_id: str,
                                              cache_dir: Path) -> Dict:
        """
        Combine MODIS cloud masks and OCO-2 footprints for a single OCO-2 granule.
        
        Takes multiple MODIS cloud mask granules that match to one OCO-2 granule,
        aggregates their pixels into a single combined result, and saves everything
        to a single cache file.
        
        Args:
            granule_cloud_masks: Dict mapping MODIS granule IDs to MODISCloudMask objects
                                 (typically the output from extract_modis_cloud_mask for one OCO-2 granule)
            granule_footprints: Dict mapping sounding_ids to OCO2Footprint objects for this granule
            oco2_granule_id: OCO-2 granule ID (e.g., 'oco2_L1bScGL_22845a_181018_...')
            cache_dir: Cache directory path where to save the combined data
        
        Returns:
            Dict containing combined data:
                - 'lon': np.ndarray of longitudes (cloud mask)
                - 'lat': np.ndarray of latitudes (cloud mask)
                - 'cloud_flag': np.ndarray of cloud flags (0=uncertain, 1=cloudy)
                - 'modis_granules': List of MODIS granule IDs that were combined
                - 'oco2_footprints': Dict of OCO-2 footprints
                - 'footprint_count': Number of OCO-2 footprints
        """
        combined_data = {
            'lon': [],
            'lat': [],
            'cloud_flag': [],
            'modis_granules': [],
            'oco2_footprints': None,
            'footprint_count': 0
        }
        
        # Aggregate all MODIS masks for this OCO-2 granule
        for modis_granule_id, cloud_mask in granule_cloud_masks.items():
            if hasattr(cloud_mask, 'lon') and cloud_mask.lon is not None:
                # New format: numpy arrays
                combined_data['lon'].append(cloud_mask.lon)
                combined_data['lat'].append(cloud_mask.lat)
                combined_data['cloud_flag'].append(cloud_mask.cloud_flag)
                combined_data['modis_granules'].append(modis_granule_id)
        
        # Concatenate arrays if we have data
        if combined_data['lon']:
            combined_data['lon'] = np.concatenate(combined_data['lon'])
            combined_data['lat'] = np.concatenate(combined_data['lat'])
            combined_data['cloud_flag'] = np.concatenate(combined_data['cloud_flag'])
        else:
            # No cloud mask data, set to None
            combined_data['lon'] = None
            combined_data['lat'] = None
            combined_data['cloud_flag'] = None
        
        # Add OCO-2 footprints
        if granule_footprints:
            fp_lats = np.array([fp.latitude for fp in granule_footprints.values()])
            fp_lons = np.array([fp.longitude for fp in granule_footprints.values()])
            fp_sounding_ids = np.array(list(granule_footprints.keys()))
            fp_viewing_modes = np.array([fp.viewing_mode for fp in granule_footprints.values()])
            combined_data['oco2_fp_lons'] = fp_lons
            combined_data['oco2_fp_lats'] = fp_lats
            combined_data['oco2_fp_sounding_ids'] = fp_sounding_ids
            combined_data['oco2_fp_viewing_modes'] = fp_viewing_modes
            combined_data['footprint_count'] = len(granule_footprints)
            
            
        # Remove cloud mask data which are for away from the OCO-2 footprints (e.g., > 2 degrees lon and 1 degree lat)
        lat_threshold = 0.5
        lon_threshold_base = 0.5
        # do the filtering every 5 degrees of latitude
        lat_degree_interval = 0.25
        if combined_data['lon'] is not None and combined_data['lat'] is not None and granule_footprints:
            footprint_lats = np.array([fp.latitude for fp in granule_footprints.values()])
            footprint_lons = np.array([fp.longitude for fp in granule_footprints.values()])
            
            # Create a mask to keep only cloud mask pixels within the threshold of any footprint
            keep_mask = np.zeros_like(combined_data['lon'], dtype=bool)
            for lat in np.arange(min(footprint_lats)-lat_threshold, max(footprint_lats)+lat_threshold, lat_degree_interval):
                fp_lat_mask = (footprint_lats >= lat) & (footprint_lats < lat + lat_degree_interval)
                if not np.any(fp_lat_mask):
                    continue
                
                
                lat_mask = (combined_data['lat'] >= lat - lat_threshold) & (combined_data['lat'] < lat + lat_degree_interval + lat_threshold)
                if not np.any(lat_mask):
                    continue
                
                # Adjust longitude threshold by latitude
                # At higher latitudes, the same degree of longitude corresponds to a smaller distance, so we need to increase the threshold in degrees
                lon_threshold_bottom = lon_threshold_base / np.cos(np.radians(lat-lat_threshold))
                lon_threshold_top = lon_threshold_base / np.cos(np.radians(lat+lat_degree_interval+lat_threshold))
                lon_threshold = max(lon_threshold_bottom, lon_threshold_top)
                # For this latitude band, check longitude proximity to footprints
                # # loop through each footprint and mark cloud mask pixels within the longitude threshold
                # for fp_lat, fp_lon in zip(footprint_lats, footprint_lons):
                #     lon_mask = (combined_data['lon'] >= fp_lon - lon_threshold) & (combined_data['lon'] <= fp_lon + lon_threshold)
                #     keep_mask |= (lat_mask & lon_mask)
                # Use min and max longitude of footprints in this latitude band to create a longitude mask
                fp_lon_min = np.min(footprint_lons[fp_lat_mask])
                fp_lon_max = np.max(footprint_lons[fp_lat_mask])
                # if fp_lon_min - lon_threshold and fp_lon_max + lon_threshold cross the dateline, we need to adjust the longitude mask accordingly
                fp_potentially_across_dateline = (fp_lon_min - lon_threshold*5 < -180) or (fp_lon_max + lon_threshold*5 > 180)
                
                cloud_lon_in_band = combined_data['lon'][lat_mask]
                cld_lon_across_dateline = (cloud_lon_in_band.min() < -90) and (cloud_lon_in_band.max() > 90)
                if fp_potentially_across_dateline or cld_lon_across_dateline:
                    tmp_fp_lon = footprint_lons[fp_lat_mask] % 360
                    tmp_fp_lon_min = np.min(tmp_fp_lon)
                    tmp_fp_lon_max = np.max(tmp_fp_lon)
                    tmp_combined_lon = cloud_lon_in_band % 360
                    lon_mask = (tmp_combined_lon >= tmp_fp_lon_min - lon_threshold) & (tmp_combined_lon <= tmp_fp_lon_max + lon_threshold)
                else:
                    lon_mask = (cloud_lon_in_band >= fp_lon_min - lon_threshold) & (cloud_lon_in_band <= fp_lon_max + lon_threshold)
                keep_mask[lat_mask] |= lon_mask
            
            # Apply the mask to filter cloud mask data
            combined_data['lon'] = combined_data['lon'][keep_mask]
            combined_data['lat'] = combined_data['lat'][keep_mask]
            combined_data['cloud_flag'] = combined_data['cloud_flag'][keep_mask]
        
        # Extract short orbit ID for display and cache filename
        short_orbit = self._extract_short_orbit_id(oco2_granule_id)
        
        # Log what we're combining
        pixel_count = len(combined_data['lon']) if combined_data['lon'] is not None else 0
        modis_count = len(combined_data['modis_granules'])
        footprint_count = combined_data['footprint_count']
        
        logger.info(f"Combining data for orbit {short_orbit}:")
        logger.info(f"  - {pixel_count} cloud pixels from {modis_count} MODIS granule(s)")
        logger.info(f"  - {footprint_count} OCO-2 footprints")
        
        # Save everything to a single cache file
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"granule_combined_{short_orbit}.pkl"
        self._save_cached_result(cache_file, combined_data)
        logger.info(f"  Saved combined data to: {cache_file.name}")
        
        # Delete individual MODIS cloud mask cache files (myd35_*.pkl) after combining
        individual_cache_files = list(cache_dir.glob("myd35_*.pkl"))
        for cache_file_to_delete in individual_cache_files:
            try:
                cache_file_to_delete.unlink()
                logger.debug(f"  Deleted individual cache file: {cache_file_to_delete.name}")
            except Exception as e:
                logger.warning(f"  Failed to delete {cache_file_to_delete.name}: {e}")
        
        return combined_data
    
    def _extract_footprints_from_file(self,
                                     filepath: Path,
                                     granule_id: str,
                                     viewing_mode: str = 'GL') -> Dict[int, OCO2Footprint]:
        """
        Extract footprints from a single OCO-2 L2 Lite file.
        
        NOTE: This method is currently NOT CALLED. L2 Lite file processing is disabled.
        
        Args:
            filepath: Path to HDF5 or NetCDF4 file
            granule_id: OCO-2 granule identifier
            viewing_mode: Filter by viewing mode
        
        Returns:
            Dictionary of footprints indexed by sounding_id
        """
        footprints = {}
        short_orbit_id = self._extract_short_orbit_id(granule_id)
        
        # L2 Lite files are NetCDF4 (.nc4 extension)
        if str(filepath).endswith('.nc4'):
            if not NC4_AVAILABLE:
                logger.error(f"    netCDF4 not available - install with: pip install netCDF4")
                return footprints
            
            try:
                with nc.Dataset(filepath, 'r') as ds:
                    # NetCDF4 variables are at top level
                    sounding_ids = ds.variables['sounding_id'][:]
                    latitudes = ds.variables['latitude'][:]
                    longitudes = ds.variables['longitude'][:]
                    times = ds.variables['time'][:]
                    
                    # Create footprints
                    for i in range(len(sounding_ids)):
                        sounding_id = int(sounding_ids[i])
                        
                        # Convert time (seconds since TAI93 epoch)
                        try:
                            sounding_time = self._convert_oco2_time(times[i])
                        except:
                            sounding_time = datetime.utcnow()
                        
                        footprints[sounding_id] = OCO2Footprint(
                            sounding_id=sounding_id,
                            granule_id=granule_id,
                            short_orbit_id=short_orbit_id,
                            latitude=float(latitudes[i]),
                            longitude=float(longitudes[i]),
                            sounding_time=sounding_time,
                            viewing_mode=viewing_mode
                        )
                    
                    logger.debug(f"    Extracted {len(footprints)} soundings from NetCDF4")
                    return footprints
                    
            except Exception as e:
                logger.error(f"    Failed to read NetCDF4 file: {e}")
                return footprints
        
        # Try HDF5 for other files
        try:
            with h5py.File(filepath, 'r') as f:
                # Navigate to the geolocation group
                if 'Geolocation' in f:
                    geo_group = f['Geolocation']
                    lat = geo_group.get('footprint_latitude', geo_group.get('latitude', None))
                    lon = geo_group.get('footprint_longitude', geo_group.get('longitude', None))
                    time = geo_group.get('time', geo_group.get('sounding_time', None))
                    
                    if lat is not None and lon is not None and time is not None:
                        lat_data = lat[:]
                        lon_data = lon[:]
                        time_data = time[:]
                        
                        # Create footprints
                        for i in range(len(lat_data)):
                            try:
                                sounding_time = self._convert_oco2_time(time_data[i])
                            except:
                                sounding_time = datetime.utcnow()
                            
                            footprints[i] = OCO2Footprint(
                                sounding_id=i,
                                granule_id=granule_id,
                                short_orbit_id=short_orbit_id,
                                latitude=float(lat_data[i]),
                                longitude=float(lon_data[i]),
                                sounding_time=sounding_time,
                                viewing_mode=viewing_mode
                            )
        
        except Exception as e:
            logger.debug(f"    Could not read as HDF5: {e}")
        
        return footprints
    
    def _convert_oco2_time(self, time_value: float) -> datetime:
        """
        Convert OCO-2 time value to datetime.
        
        OCO-2 L2 Lite files store time as seconds since Unix epoch (1970-01-01 00:00:00).
        Other OCO-2 products may use TAI93 (1993-01-01 00:00:00).
        
        Args:
            time_value: Time in seconds since reference epoch
        
        Returns:
            datetime object
        """
        # Try Unix epoch first (most L2 Lite files)
        unix_epoch = datetime(1970, 1, 1, 0, 0, 0)
        
        try:
            dt = unix_epoch + timedelta(seconds=float(time_value))
            
            # Sanity check: should be between 2009 (OCO-2 launch prep) and 2030
            if 2009 <= dt.year <= 2030:
                return dt
                
            # If out of range, try TAI93
            tai93_epoch = datetime(1993, 1, 1, 0, 0, 0)
            dt = tai93_epoch + timedelta(seconds=float(time_value))
            
            if 2009 <= dt.year <= 2030:
                return dt
            
            # Still out of range, return current time
            logger.warning(f"Time value {time_value} produced out-of-range date: {dt}")
            return datetime.utcnow()
        except Exception as e:
            logger.debug(f"Time conversion error: {e}")
            return datetime.utcnow()
    
    def _extract_sounding_ids_from_lite(self, filepath: Path, use_cache: bool = True) -> set:
        """
        Extract sounding_id list from L2 Lite file.
        
        This identifies which soundings have successful XCO2 retrievals.
        Uses day-level caching to avoid re-opening the same Lite file for multiple granules.
        
        Args:
            filepath: Path to L2 Lite NetCDF4 file
            use_cache: If True, load from/save to cache
        
        Returns:
            Set of sounding_ids present in the Lite file
        """
        sounding_ids = set()
        
        if not NC4_AVAILABLE:
            logger.warning(f"    netCDF4 not available - cannot extract sounding_ids from Lite file")
            return sounding_ids
        
        # Try to load from cache first
        cache_path = None
        if use_cache:
            # Parse date from filename to create day-level cache path
            # Format: oco2_LtCO2_YYMMDD_B11004Ar_*.nc4
            match = re.search(r'_(\d{6})_B', filepath.name)
            if match:
                date_str = match.group(1)
                year = 2000 + int(date_str[0:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                file_date = datetime(year, month, day)
                
                # Create day-level cache path (not orbit-specific)
                # data/processing/2018/291/lite_sounding_ids.pkl
                year_str = file_date.strftime('%Y')
                doy = file_date.timetuple().tm_yday
                cache_dir = self.processing_dir / year_str / f"{doy:03d}"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / "lite_sounding_ids.pkl"
                
                # Check if cache exists
                cached_data = self._load_cached_result(cache_path)
                if cached_data is not None:
                    logger.info(f"    📂 Loaded Lite sounding_ids from cache: {len(cached_data)} ids")
                    return cached_data
        
        # Cache miss or disabled - extract from file
        try:
            with nc.Dataset(filepath, 'r') as ds:
                if 'sounding_id' in ds.variables:
                    ids = ds.variables['sounding_id'][:]
                    sounding_ids = set(int(sid) for sid in ids)
                    logger.debug(f"    Extracted {len(sounding_ids)} sounding_ids from {filepath.name}")
                else:
                    logger.warning(f"    No sounding_id variable in {filepath.name}")
        except Exception as e:
            logger.error(f"    Failed to read Lite file {filepath.name}: {e}")
            return sounding_ids
        
        # Save to cache
        if use_cache and cache_path and sounding_ids:
            self._save_cached_result(cache_path, sounding_ids)
            logger.info(f"    💾 Cached Lite sounding_ids: {len(sounding_ids)} ids")
        
        return sounding_ids
    
    def _extract_footprints_from_l1b(self,
                                    filepath: Path,
                                    granule_id: str,
                                    viewing_mode=None) -> Dict[int, OCO2Footprint]:
        """
        Extract footprints from L1B Science HDF5 file.
        
        L1B files contain high-accuracy geolocation data in the /SoundingGeometry group.
        
        Args:
            filepath: Path to L1B Science HDF5 file
            granule_id: OCO-2 granule identifier
            viewing_mode: Expected viewing mode ('GL' or 'ND' or 'TG') for filtering (optional)
        
        Returns:
            Dictionary of footprints indexed by sounding_id
        """
        footprints = {}
        short_orbit_id = self._extract_short_orbit_id(granule_id)
        
        try:
            with h5py.File(filepath, 'r') as f:
                viewing_mode = filepath.name.split('_')[1].upper()[-2:]  # e.g. "L1BScGL" -> "L1BSCGL"
                # L1B Science files have /SoundingGeometry group
                if 'SoundingGeometry' not in f:
                    logger.warning(f"    No SoundingGeometry group in {filepath.name}")
                    return footprints
                
                geo_group = f['SoundingGeometry']
                
                # Extract required datasets
                if 'sounding_id' not in geo_group:
                    logger.warning(f"    No sounding_id in SoundingGeometry")
                    return footprints
                
                sounding_ids = geo_group['sounding_id'][:]
                sounding_latitude = geo_group['sounding_latitude'][:]
                sounding_longitude = geo_group['sounding_longitude'][:]
                sounding_time_tai93 = geo_group['sounding_time_tai93'][:]
                
                # L1B data is typically 2D: (frames, footprints)
                # Flatten to 1D for processing
                sounding_ids = sounding_ids.flatten()
                sounding_latitude = sounding_latitude.flatten()
                sounding_longitude = sounding_longitude.flatten()
                sounding_time_tai93 = sounding_time_tai93.flatten()
                
                logger.debug(f"    L1B: {len(sounding_ids)} soundings in file")
                
                # Create footprints
                for i in range(len(sounding_ids)):
                    sounding_id = int(sounding_ids[i])
                    
                    # Skip invalid soundings (sounding_id = 0)
                    if sounding_id == 0:
                        continue
                    
                    # Convert TAI93 time to datetime
                    try:
                        tai93_epoch = datetime(1993, 1, 1, 0, 0, 0)
                        sounding_time = tai93_epoch + timedelta(seconds=float(sounding_time_tai93[i]))
                    except:
                        sounding_time = datetime.utcnow()
                    
                    footprints[sounding_id] = OCO2Footprint(
                        sounding_id=sounding_id,
                        granule_id=granule_id,
                        short_orbit_id=short_orbit_id,
                        latitude=float(sounding_latitude[i]),
                        longitude=float(sounding_longitude[i]),
                        sounding_time=sounding_time,
                        viewing_mode=viewing_mode
                    )
                
                logger.debug(f"    Successfully extracted {len(footprints)} footprints from L1B")
        
        except Exception as e:
            logger.error(f"    Failed to read L1B file {filepath.name}: {e}")
        
        return footprints
    
    def extract_modis_cloud_mask(self,
                                 modis_files: List[DownloadedFile],
                                 myd03_files: List[DownloadedFile] = None,
                                 use_cache: bool = True,
                                 oco2_orbit_id: str = None) -> Dict[str, List[MODISCloudMask]]:
        """
        Extract cloud pixels from MODIS MYD35_L2 cloud mask.
        
        Unpacks the 48-bit cloud mask (Byte 0, bits 1-2) to identify cloudy/uncertain pixels.
        Uses MYD03 geolocation data for lat/lon coordinates.
        
        Args:
            modis_files: List of MYD35_L2 files (from Phase 2)
            myd03_files: List of MYD03 geolocation files (optional, for high-res coords)
            use_cache: If True, load from/save to cache
            oco2_orbit_id: OCO-2 orbit ID (e.g., '22845a') to organize cache under
        
        Returns:
            Dictionary indexed by granule_id with list of cloud pixels
        """
        cloud_pixels_by_granule = {}
        
        if use_cache:
            logger.info(f"  Cache enabled: checking for cached cloud masks")
        
        # Build MYD03 lookup by time-based granule ID (extract A2018291.HHMM part)
        # This is needed because MYD35_L2 and MYD03 files have different download timestamps
        # Ensure 1:1 mapping: each time ID should map to exactly one MYD03 file
        myd03_lookup = {}
        myd03_duplicates = []  # Track duplicate time IDs
        
        logger.info(f"  Building MYD03 lookup from {len(myd03_files) if myd03_files else 0} files...")
        if myd03_files:
            for file_obj in myd03_files:
                # Extract time-based ID: A2018291.HHMM from filename
                match = re.search(r'A(\d{4}\d{3}\.\d{4})', file_obj.filepath.name)
                if match:
                    time_id = 'A' + match.group(1)
                    
                    # Check for duplicates
                    if time_id in myd03_lookup:
                        myd03_duplicates.append((time_id, myd03_lookup[time_id].filepath.name, file_obj.filepath.name))
                        logger.warning(f"  ⚠️  Duplicate MYD03 time ID {time_id}: {myd03_lookup[time_id].filepath.name} vs {file_obj.filepath.name}")
                    
                    myd03_lookup[time_id] = file_obj
                    logger.debug(f"    Registered MYD03: {time_id} -> {file_obj.filepath.name}")
            
            logger.info(f"  MYD03 lookup created with {len(myd03_lookup)} unique time ID(s)")
            if myd03_duplicates:
                logger.warning(f"  ⚠️  Found {len(myd03_duplicates)} duplicate MYD03 time ID(s) - using last occurrence")
        
        # Filter to MYD35_L2 files and check for duplicates
        myd35_files = [f for f in modis_files if f.product_type == 'MYD35_L2']
        
        # Validate 1:1 mapping: check for duplicate MYD35_L2 time IDs
        myd35_time_ids = {}
        myd35_duplicates = []
        for file_obj in myd35_files:
            match = re.search(r'A(\d{4}\d{3}\.\d{4})', file_obj.filepath.name)
            if match:
                time_id = 'A' + match.group(1)
                if time_id in myd35_time_ids:
                    myd35_duplicates.append((time_id, myd35_time_ids[time_id], file_obj.filepath.name))
                    logger.warning(f"  ⚠️  Duplicate MYD35_L2 time ID {time_id}: {myd35_time_ids[time_id]} vs {file_obj.filepath.name}")
                myd35_time_ids[time_id] = file_obj.filepath.name
        
        if myd35_duplicates:
            logger.warning(f"  ⚠️  Found {len(myd35_duplicates)} duplicate MYD35_L2 time ID(s) - multiple files share the same observation time")
        
        logger.info(f"\n[Processing] Extracting MODIS cloud masks from {len(myd35_files)} MYD35_L2 file(s)")
        logger.info(f"  MYD35_L2 unique time IDs: {len(myd35_time_ids)}")
        logger.info(f"  MYD03 available time IDs: {len(myd03_lookup)}")
        
        # Track actual pairings used
        myd35_myd03_pairings = []
        
        for file_obj in myd35_files:
            try:
                granule_id = file_obj.granule_id  # Use actual granule ID (e.g., full MODIS filename)
                logger.debug(f"  Processing: {file_obj.filepath.name}")
                
                # Try to load from cache first
                cache_loaded = False
                if use_cache:
                    # Parse date from filename (A2018291... = year 2018, doy 291)
                    match = re.search(r'A(\d{4})(\d{3})\.(\d{4})', file_obj.filepath.name)
                    if match:
                        year = int(match.group(1))
                        doy = int(match.group(2))
                        time_hhmm = match.group(3)
                        file_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
                        # Use MYD35-specific cache name with time ID
                        cache_name = f'myd35_{year}{doy:03d}.{time_hhmm}'
                        
                        # Use OCO-2 orbit ID for cache folder if provided, otherwise fall back to MODIS time ID
                        cache_folder_id = oco2_orbit_id if oco2_orbit_id else granule_id
                        cache_path = self._get_cache_path(file_date, cache_folder_id, cache_name)
                        cached_data = self._load_cached_result(cache_path)
                        if cached_data is not None:
                            if hasattr(cached_data, 'lon') and cached_data.lon is not None:
                                pixel_count = len(cached_data.lon)
                                cloud_pixels_by_granule[granule_id] = cached_data
                                logger.info(f"    📂 Loaded from cache: {granule_id} ({pixel_count} pixels, array format)")
                            else:
                                pixel_count = 0
                                logger.warning(f"    ⚠️  Cached data has unrecognized format")
                            cache_loaded = True
                
                if not cache_loaded:
                    # Find corresponding MYD03 file by matching time-based ID
                    myd03_file = None
                    if myd03_lookup:
                        match = re.search(r'A(\d{4}\d{3}\.\d{4})', granule_id)
                        if match:
                            time_id = 'A' + match.group(1)
                            myd03_file = myd03_lookup.get(time_id, None)
                            if myd03_file:
                                logger.info(f"    ✓ Using MYD03 match for {time_id}: {myd03_file.filepath.name}")
                                # Track pairing
                                myd35_myd03_pairings.append({
                                    'time_id': time_id,
                                    'myd35': file_obj.filepath.name,
                                    'myd03': myd03_file.filepath.name
                                })
                            else:
                                logger.warning(f"    ✗ No MYD03 match for {time_id} - extracting without geolocation")
                    else:
                        logger.debug(f"    No MYD03 files provided")
                    
                    # Extract cloud mask
                    cloud_pixels = self._extract_cloud_pixels_from_file(
                        file_obj.filepath,
                        granule_id,
                        myd03_file
                    )
                    
                    cloud_pixels_by_granule[granule_id] = cloud_pixels
                    
                    # Provide feedback on success
                    # Handle both new (array-based) and old (tuple-based) formats
                    if hasattr(cloud_pixels, 'lon') and cloud_pixels.lon is not None:
                        pixel_count = len(cloud_pixels.lon)
                        cloudy_count = np.sum(cloud_pixels.cloud_flag == 1)
                        uncertain_count = np.sum(cloud_pixels.cloud_flag == 0)
                        logger.info(f"    ✓ Extracted {pixel_count} cloud pixel(s): {cloudy_count} cloudy, {uncertain_count} uncertain")
                    else:
                        logger.info(f"    ✓ Extracted 0 cloud pixel(s)")
                    
                    # Save to cache
                    if use_cache and cloud_pixels:
                        match = re.search(r'A(\d{4})(\d{3})\.(\d{4})', file_obj.filepath.name)
                        if match:
                            year = int(match.group(1))
                            doy = int(match.group(2))
                            time_hhmm = match.group(3)
                            file_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
                            # Use MYD35-specific cache name with time ID
                            cache_name = f'myd35_{year}{doy:03d}.{time_hhmm}'
                            
                            # Use OCO-2 orbit ID for cache folder if provided, otherwise fall back to MODIS time ID
                            cache_folder_id = oco2_orbit_id if oco2_orbit_id else granule_id
                            cache_path = self._get_cache_path(file_date, cache_folder_id, cache_name)
                            self._save_cached_result(cache_path, cloud_pixels)
            
            except Exception as e:
                logger.error(f"  ✗ Failed to process {file_obj.filepath.name}: {e}")
        
        # Calculate total clouds handling both new (array-based) and old (tuple-based) formats
        total_clouds = 0
        for granule_pixels in cloud_pixels_by_granule.values():
            if hasattr(granule_pixels, 'lon') and granule_pixels.lon is not None:
                # New array format: MODISCloudMask with numpy arrays
                total_clouds += len(granule_pixels.lon)
        
        logger.info(f"✓ Total cloud pixels extracted: {total_clouds}")
        
        # Log MYD03 cache statistics
        if self._myd03_cache:
            logger.info(f"✓ MYD03 geolocation cache: {len(self._myd03_cache)} file(s) loaded")
        
        # Report MYD35_L2 <-> MYD03 pairing summary
        if myd35_myd03_pairings:
            logger.info(f"\n✓ MYD35_L2 <-> MYD03 Pairing Summary ({len(myd35_myd03_pairings)} pairs):")
            for pairing in myd35_myd03_pairings:
                logger.info(f"  {pairing['time_id']}: MYD35={pairing['myd35']} <-> MYD03={pairing['myd03']}")
            
            # Verify 1:1 mapping
            myd03_used = [p['myd03'] for p in myd35_myd03_pairings]
            myd35_used = [p['myd35'] for p in myd35_myd03_pairings]
            
            if len(myd03_used) != len(set(myd03_used)):
                logger.warning(f"  ⚠️  Warning: Some MYD03 files are used multiple times!")
                from collections import Counter
                myd03_counts = Counter(myd03_used)
                for filename, count in myd03_counts.items():
                    if count > 1:
                        logger.warning(f"    {filename}: used {count} times")
            else:
                logger.info(f"  ✓ Verified: Each MYD03 file used exactly once")
            
            if len(myd35_used) != len(set(myd35_used)):
                logger.warning(f"  ⚠️  Warning: Some MYD35_L2 files mapped multiple times!")
            else:
                logger.info(f"  ✓ Verified: Each MYD35_L2 file mapped exactly once")
        
        return cloud_pixels_by_granule
    
    def _extract_cloud_pixels_from_file(self,
                                       filepath: Path,
                                       granule_id: str,
                                       myd03_file: DownloadedFile = None) -> List[MODISCloudMask]:
        """
        Extract cloud pixels from a single MODIS MYD35_L2 file.
        
        Args:
            filepath: Path to MYD35_L2 HDF4 file
            granule_id: MODIS granule identifier
            myd03_file: Optional MYD03 file object for geolocation
        
        Returns:
            MODISCloudMask object
        """
        cloud_pixels = []
        
        # MODIS files are HDF4 format
        if not HDF4_AVAILABLE:
            logger.error(f"    pyhdf not available - install with: pip install python-hdf4")
            return cloud_pixels
        
        try:
            # Load geolocation data from MYD03 if available
            geolocation = None
            if myd03_file:
                geolocation = self._load_myd03_geolocation(myd03_file.filepath)
                if not geolocation:
                    logger.warning(f"    ⚠️  Could not load geolocation from {myd03_file.filepath.name} - cloud pixels will use pixel indices")
            
            # Open HDF4 file
            hdf = SD(str(filepath), SDC.READ)
            
            # Get Cloud_Mask dataset
            if 'Cloud_Mask' in hdf.datasets():
                cloud_mask_sds = hdf.select('Cloud_Mask')
                cloud_mask_data = cloud_mask_sds.get()
                
                logger.debug(f"    Cloud mask shape: {cloud_mask_data.shape}")
                logger.debug(f"    Cloud mask dtype: {cloud_mask_data.dtype}")
                
                # Unpack the mask (returns efficient MODISCloudMask object with sparse pixels)
                cloud_mask_obj = self._unpack_cloud_mask(
                    cloud_mask_data,
                    granule_id,
                    geolocation
                )
                
                cloud_mask_sds.endaccess()
            else:
                logger.warning(f"    No Cloud_Mask dataset found in {filepath.name}")
                logger.debug(f"    Available datasets: {list(hdf.datasets().keys())[:10]}")
            
            hdf.end()
        
        except Exception as e:
            logger.error(f"    Failed to read HDF4 file: {e}")
        
        return cloud_mask_obj
    
    def _load_myd03_geolocation(self, myd03_filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load latitude and longitude arrays from MYD03 geolocation file.
        
        Uses caching to avoid re-reading the same file multiple times.
        
        Args:
            myd03_filepath: Path to MYD03 HDF4 file
        
        Returns:
            Tuple of (latitude_array, longitude_array), or None if loading failed
        """
        if not HDF4_AVAILABLE:
            return None
        
        # Check cache first
        cache_key = str(myd03_filepath)
        if cache_key in self._myd03_cache:
            logger.info(f"    📋 Using cached MYD03 geolocation: {myd03_filepath.name}")
            return self._myd03_cache[cache_key]
        
        try:
            logger.info(f"    📂 Loading MYD03 geolocation from disk: {myd03_filepath.name}")
            hdf = SD(str(myd03_filepath), SDC.READ)
            
            # Try to load latitude and longitude
            lat_sds = hdf.select('Latitude')
            lon_sds = hdf.select('Longitude')
            
            latitude = lat_sds.get()
            longitude = lon_sds.get()
            
            lat_sds.endaccess()
            lon_sds.endaccess()
            hdf.end()
            
            logger.info(f"    ✓ Loaded & cached MYD03: lat shape={latitude.shape}, lon shape={longitude.shape}")
            
            # Cache the result
            geolocation_data = (latitude, longitude)
            self._myd03_cache[cache_key] = geolocation_data
            
            return geolocation_data
        
        except Exception as e:
            logger.warning(f"    Failed to load MYD03 geolocation from {myd03_filepath.name}: {e}")
            logger.info(f"    ⚠️  File may be corrupted. Attempting to re-download {myd03_filepath.name}...")
            
            # Try to re-download the corrupted file
            try:
                from phase_02_ingestion import DataIngestionManager
                
                # Extract year, doy, and granule info from filename
                # Format: MYD03.AYYYYDDD.HHMM.061.*.hdf
                import re
                match = re.search(r'A(\d{4})(\d{3})\.(\d{4})', myd03_filepath.name)
                if match:
                    year = int(match.group(1))
                    doy = int(match.group(2))
                    
                    manager = DataIngestionManager(output_dir=str(self.data_dir))
                    
                    # Delete corrupted file
                    logger.info(f"    Deleting corrupted file: {myd03_filepath.name}")
                    myd03_filepath.unlink()
                    
                    # Re-download
                    logger.info(f"    Re-downloading {myd03_filepath.name}...")
                    result = manager.download_modis_granule(
                        granule_filename=myd03_filepath.name,
                        product_type='MYD03',
                        year=year,
                        doy=doy
                    )
                    
                    if result and result.filepath.exists():
                        logger.info(f"    ✓ Successfully re-downloaded {myd03_filepath.name}")
                        # Try again
                        try:
                            hdf = SD(str(myd03_filepath), SDC.READ)
                            lat_sds = hdf.select('Latitude')
                            lon_sds = hdf.select('Longitude')
                            latitude = lat_sds.get()
                            longitude = lon_sds.get()
                            lat_sds.endaccess()
                            lon_sds.endaccess()
                            hdf.end()
                            logger.debug(f"    Successfully loaded re-downloaded geolocation: lat shape={latitude.shape}, lon shape={longitude.shape}")
                            
                            # Cache the re-downloaded data
                            geolocation_data = (latitude, longitude)
                            self._myd03_cache[cache_key] = geolocation_data
                            
                            return geolocation_data
                        except Exception as retry_error:
                            logger.error(f"    Failed to load re-downloaded file: {retry_error}")
                    else:
                        logger.error(f"    Failed to re-download {myd03_filepath.name}")
                else:
                    logger.debug(f"    Could not extract year/doy from {myd03_filepath.name}")
            except Exception as download_error:
                logger.error(f"    Re-download attempt failed: {download_error}")
            
            return None
    
    def _unpack_cloud_mask(self,
                          mask_data: np.ndarray,
                          granule_id: str,
                          geolocation: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Optional[MODISCloudMask]:
        """
        Unpack MODIS 48-bit cloud mask to identify cloudy/uncertain pixels.
        
        The mask is stored as 6 bytes (48 bits) per pixel:
        - Byte 0: bit 0 = cloud mask determined (0=not determined, 1=determined)
                  bits 1-2 = cloud flags (00=Cloudy, 01=Uncertain, 10=Probably Clear, 11=Clear)
        - Bytes 1-5: Additional flags
        
        Args:
            mask_data: 3D numpy array with uint8 dtype
                      Shape is (6, rows, cols) or (rows, cols, 6)
            granule_id: MODIS granule identifier
            geolocation: Optional tuple of (latitude_array, longitude_array) from MYD03
        
        Returns:
            MODISCloudMask object for cloudy/uncertain pixels
        """
        cloud_pixels = []
        
        # Handle different data shapes - MYD35 uses (bytes, rows, cols)
        if mask_data.ndim == 3:
            if mask_data.shape[0] == 6:
                # Shape: (6 bytes, rows, cols) - MODIS MYD35_L2 format
                byte0 = mask_data[0, :, :]  # First byte: bit 0 = determined, bits 1-2 = cloud flags
                logger.debug(f"    Cloud mask shape: (6 bytes, {mask_data.shape[1]} rows, {mask_data.shape[2]} cols)")
            elif mask_data.shape[2] == 6:
                # Shape: (rows, cols, 6 bytes) - alternative format
                byte0 = mask_data[:, :, 0]
                logger.debug(f"    Cloud mask shape: ({mask_data.shape[0]} rows, {mask_data.shape[1]} cols, 6 bytes)")
            else:
                logger.warning(f"    Unexpected cloud mask shape: {mask_data.shape}")
                # Try assuming last dimension is bytes
                byte0 = mask_data[:, :, 0] if mask_data.shape[2] < mask_data.shape[0] else mask_data[0, :, :]
        elif mask_data.ndim == 2:
            # If 2D, assume it's already processed or single byte
            logger.warning(f"    Unexpected 2D cloud mask shape: {mask_data.shape}")
            byte0 = mask_data
        else:
            logger.warning(f"    Unsupported cloud mask shape: {mask_data.shape}")
            return cloud_pixels
        
        # Extract bits from Byte 0 (first byte)
        # Bit 0: Cloud mask determined (0=not determined, 1=determined)
        # Bits 1-2: Cloud flags (00=Cloudy, 01=Uncertain, 10=Probably Clear, 11=Clear)
        # Bit 3: Day/Night Flag (1=Day/Ascending, 0=Night/Descending)
        # Bits 4-7: Reserved
        # Bits are counted from right (LSB=0, MSB=7)
        
        cloud_flags = (byte0 >> 1) & 0b11
        day_night_flag = (byte0 >> 3) & 0b1  # Extract Bit 3
        
        # Calculate day/night granule classification
        day_pixels = np.sum(day_night_flag == 1)
        night_pixels = np.sum(day_night_flag == 0)
        is_day_pass = day_pixels > night_pixels  # Majority vote
        
        logger.debug(f"    Day/Night pixels - Day: {day_pixels}, Night: {night_pixels} (Pass: {'Day' if is_day_pass else 'Night'})")
        
        # Identify all cloud mask categories
        cloudy_mask = cloud_flags == self.CLOUD_CLOUDY
        uncertain_mask = cloud_flags == self.CLOUD_UNCERTAIN
        # probably_clear_mask = cloud_flags == self.CLOUD_PROBABLY_CLEAR  # Commented out to save storage
        # clear_mask = cloud_flags == self.CLOUD_CLEAR  # Commented out to save storage
        
        # Only include cloudy and uncertain pixels (to save storage)
        target_pixels = cloudy_mask | uncertain_mask
        # target_pixels = cloudy_mask | uncertain_mask | probably_clear_mask | clear_mask  # Full version
        
        # Get pixel coordinates
        rows, cols = np.where(target_pixels)
        
        # Debug: Output bit pattern statistics
        unique_flags, counts = np.unique(cloud_flags, return_counts=True)
        logger.debug(f"    Cloud flag distribution:")
        for flag, count in zip(unique_flags, counts):
            flag_name = {
                self.CLOUD_CLEAR: "Clear",
                self.CLOUD_PROBABLY_CLEAR: "Probably Clear",
                self.CLOUD_UNCERTAIN: "Uncertain",
                self.CLOUD_CLOUDY: "Cloudy"
            }.get(flag, f"Unknown({flag})")
            logger.debug(f"      {flag_name} (bits={flag:02b}): {count} pixels")
        
        logger.debug(f"    Target pixels (Cloudy+Uncertain): {len(rows)} out of {cloud_flags.size}")
        
        # Get observation time from granule ID (format: MYD35_L2.AYYYYDDD.HHMM.*.hdf)
        try:
            import re
            match = re.search(r'A(\d{4})(\d{3})\.(\d{4})', granule_id)
            if match:
                year = int(match.group(1))
                doy = int(match.group(2))
                hhmm = match.group(3)
                hour = int(hhmm[:2])
                minute = int(hhmm[2:4])
                
                # Create datetime
                obs_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
                obs_time = obs_date.replace(hour=hour, minute=minute, second=0)
            else:
                obs_time = datetime.utcnow()
        except:
            obs_time = datetime.utcnow()
        
        # Log summary statistics
        cloudy_count = np.sum(cloudy_mask)
        uncertain_count = np.sum(uncertain_mask)
        total_count = cloudy_count + uncertain_count
        
        logger.debug(f"    Cloudy pixels: {cloudy_count}, Uncertain pixels: {uncertain_count}, Total: {total_count}")
        
        # Log geolocation array info if available
        if geolocation is not None:
            latitude_arr, longitude_arr = geolocation
            logger.debug(f"    Geolocation available: lat shape={latitude_arr.shape}, lon shape={longitude_arr.shape}")
            # Validate geolocation arrays are compatible
            if latitude_arr.shape != longitude_arr.shape:
                logger.warning(f"    Geolocation shape mismatch: lat {latitude_arr.shape} vs lon {longitude_arr.shape}")
                geolocation = None
        else:
            logger.debug(f"    No geolocation data available")
            latitude_arr = None
            longitude_arr = None
        
        # Always return array-based sparse storage format (NEW, efficient)
        # Store lon, lat as float32 arrays and cloud_flag as uint8 (0=Uncertain, 1=Cloudy)
        # This provides direct geolocation data and enables vectorized KD-tree operations
        
        # Check if we have geolocation data
        if latitude_arr is None or longitude_arr is None:
            logger.warning(f"    No geolocation data available - cannot create cache")
            return MODISCloudMask(
                granule_id=granule_id,
                observation_time=obs_time,
                lon=np.array([], dtype=np.float32),
                lat=np.array([], dtype=np.float32),
                cloud_flag=np.array([], dtype=np.uint8)
            )
        
        # Build arrays by collecting indices first, then constructing arrays
        lon_list = []
        lat_list = []
        flag_list = []
        
        # Process cloudy pixels (flag=1)
        cloudy_indices = np.where(cloudy_mask)
        for y, x in zip(cloudy_indices[0], cloudy_indices[1]):
            lon_list.append(float(longitude_arr[y, x]))
            lat_list.append(float(latitude_arr[y, x]))
            flag_list.append(1)  # Cloudy
        
        # Process uncertain pixels (flag=0)
        uncertain_indices = np.where(uncertain_mask)
        for y, x in zip(uncertain_indices[0], uncertain_indices[1]):
            lon_list.append(float(longitude_arr[y, x]))
            lat_list.append(float(latitude_arr[y, x]))
            flag_list.append(0)  # Uncertain
        
        # Convert lists to numpy arrays for efficient storage
        lon_array = np.array(lon_list, dtype=np.float32)
        lat_array = np.array(lat_list, dtype=np.float32)
        flag_array = np.array(flag_list, dtype=np.uint8)
        
        logger.debug(f"    Sparse cloud mask: {len(lon_list)} pixels (arrays format)")
        
        # Return MODISCloudMask object with array-based sparse storage
        cloud_mask_obj = MODISCloudMask(
            granule_id=granule_id,
            observation_time=obs_time,
            lon=lon_array,
            lat=lat_array,
            cloud_flag=flag_array
        )
        return cloud_mask_obj
    
    def match_temporal_windows(self,
                              oco2_footprints: Dict[int, OCO2Footprint],
                              modis_files: List[DownloadedFile],
                              buffer_minutes: int = 20) -> Dict[int, List[str]]:
        """
        Match OCO-2 soundings to MODIS granules based on temporal proximity.
        
        Uses adaptive buffer: ±10 minutes for years < 2023, ±20 minutes for 2023+
        (Aqua orbital drift increased after 2023).
        
        Args:
            oco2_footprints: Dictionary of OCO2Footprint objects by sounding_id
            modis_files: List of MODIS file objects
            buffer_minutes: Temporal buffer for matching (±minutes, default=20)
        
        Returns:
            Dictionary: sounding_id -> list of matching MODIS granule IDs
        """
        matching = {}
        
        logger.info(f"\n[Temporal Matching] Matching {len(oco2_footprints)} OCO-2 soundings to MODIS granules")
        logger.info(f"                   Buffer: ±{buffer_minutes} minutes")
        
        # Extract granule times from filenames and determine year for adaptive buffer
        modis_granule_times = {}
        observation_year = None
        
        for file_obj in modis_files:
            granule_id = file_obj.granule_id
            try:
                import re
                # Format: MYD35_L2.AYYYYDDD.HHMM.061.*.hdf or MYD03.AYYYYDDD.HHMM.061.*.hdf
                match = re.search(r'A(\d{4})(\d{3})\.(\d{4})', granule_id)
                if match:
                    year = int(match.group(1))
                    doy = int(match.group(2))
                    hhmm = match.group(3)
                    hour = int(hhmm[:2])
                    minute = int(hhmm[2:4])
                    
                    # Track observation year for adaptive buffer
                    if observation_year is None:
                        observation_year = year
                    
                    # Check Day/Night flag from cloud mask (Bit 3 of Byte 0)
                    # Only include day passes (ascending tracks)
                    if file_obj.product_type == 'MYD35_L2':
                        try:
                            from pyhdf.SD import SD, SDC
                            hdf_file = SD(str(file_obj.filepath))
                            cloud_mask_data = hdf_file.select('Cloud_Mask')
                            mask_data = cloud_mask_data[:, :]
                            hdf_file.end()
                            
                            # Extract Byte 0 and check Bit 3 (Day/Night flag)
                            if mask_data.ndim == 3 and mask_data.shape[0] == 6:
                                byte0 = mask_data[0, :, :]
                            elif mask_data.ndim == 3 and mask_data.shape[2] == 6:
                                byte0 = mask_data[:, :, 0]
                            else:
                                byte0 = mask_data
                            
                            day_night_flag = (byte0 >> 3) & 0b1
                            day_pixels = np.sum(day_night_flag == 1)
                            night_pixels = np.sum(day_night_flag == 0)
                            is_day_pass = day_pixels > night_pixels
                            
                            if not is_day_pass:
                                logger.debug(f"    Skipping night pass (descending): {granule_id} (Day: {day_pixels}, Night: {night_pixels})")
                                continue
                        except Exception as e:
                            logger.debug(f"    Could not check day/night flag for {granule_id}: {e}")
                            # Continue anyway if we can't determine
                    
                    # Create datetime
                    granule_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
                    granule_time = granule_date.replace(hour=hour, minute=minute, second=0)
                    modis_granule_times[granule_id] = granule_time
            except Exception as e:
                logger.debug(f"  Could not parse time from {granule_id}: {e}")
        
        # Adjust buffer based on observation year (Aqua drift increased after 2023)
        effective_buffer = buffer_minutes
        if observation_year is not None and observation_year < 2023:
            effective_buffer = 10  # Use ±10 minutes for pre-2023 data
            logger.info(f"                   Year {observation_year} < 2023: Using reduced buffer of ±{effective_buffer} minutes")
        
        logger.info(f"  Found {len(modis_granule_times)} MODIS granules with timing (ascending tracks only)")
        
        # Match each OCO-2 sounding to MODIS granules
        for sounding_id, footprint in oco2_footprints.items():
            sounding_time = footprint.sounding_time
            
            # Find MODIS granules within temporal window
            matches = []
            for granule_id, granule_time in modis_granule_times.items():
                time_diff = abs((sounding_time - granule_time).total_seconds() / 60)  # minutes
                if time_diff <= effective_buffer:
                    matches.append((granule_id, time_diff))
            
            # Sort by time difference (closest first)
            matches.sort(key=lambda x: x[1])
            matching[sounding_id] = [g[0] for g in matches]
        
        # Statistics
        with_matches = sum(1 for m in matching.values() if len(m) > 0)
        logger.info(f"✓ {with_matches}/{len(oco2_footprints)} soundings matched to MODIS granules")
        
        return matching
    
    def get_processing_summary(self,
                              oco2_footprints: Dict[int, OCO2Footprint],
                              modis_cloud_pixels: Dict[str, List[MODISCloudMask]],
                              temporal_matching: Dict[int, List[str]]) -> Dict:
        """
        Generate summary statistics for processed data.
        
        Args:
            oco2_footprints: Extracted OCO-2 footprints
            modis_cloud_pixels: Extracted MODIS cloud pixels by granule
                               Can contain MODISCloudMask objects or lists
            temporal_matching: OCO-2 to MODIS temporal matching
        
        Returns:
            Summary dictionary
        """
        # Calculate total clouds handling both formats
        total_clouds = 0
        for granule_pixels in modis_cloud_pixels.values():
            if hasattr(granule_pixels, 'get_pixel_count'):
                total_clouds += granule_pixels.get_pixel_count()
        
        matched_soundings = sum(1 for matches in temporal_matching.values() if len(matches) > 0)
        
        # Count unique OCO-2 granules
        unique_granules = set(fp.granule_id for fp in oco2_footprints.values())
        
        return {
            'oco2_footprints': len(oco2_footprints),
            'oco2_granules': len(unique_granules),
            'modis_granules': len(modis_cloud_pixels),
            'total_cloud_pixels': total_clouds,
            'matched_soundings': matched_soundings,
            'unmatched_soundings': len(oco2_footprints) - matched_soundings
        }
    
    
    def plot_by_latitude_bands(self,
                              cache_dir: Path,
                              output_dir: str = "./visualizations",
                              lat_band_size: float = 10.0,
                              figsize: Tuple[int, int] = (16, 10),
                              dpi: int = 100) -> Dict[str, Path]:
        """
        Load combined cache files and plot by latitude bands.
        
        Instead of plotting individual granules, this aggregates all matched MODIS
        cloud pixels from a day and groups them into latitude bands for visualization.
        Also overlays OCO-2 footprints from the combined cache.
        
        Args:
            cache_dir: Directory containing granule_combined_*.pkl files (e.g., data/processing/2018/291/)
            output_dir: Directory to save visualizations
            lat_band_size: Latitude band size in degrees (default: 10°)
            figsize: Figure size (width, height) in inches
            dpi: Resolution in dots per inch
        
        Returns:
            Dictionary mapping latitude band to Path of saved visualization
        
        Features:
        - Aggregates MODIS cloud pixels from combined cache files
        - Groups pixels by 10° latitude bands
        - Shows cloudy (red squares) and uncertain (orange triangles) pixels
        - Overlays OCO-2 footprints (blue circles=Glint, green diamonds=Nadir)
        - Automatically detects and handles date line crossing
        - One plot per latitude band with cumulative statistics
        """
        from pathlib import Path
        import glob
        
        cache_path = Path(cache_dir).resolve()
        
        # Find all granule_combined_*.pkl files
        cache_files = sorted(cache_path.glob("granule_combined_*.pkl"))
        if not cache_files:
            logger.warning(f"No combined cache files found in {cache_path}")
            return {}
        
        logger.info(f"\n[Latitude Band Visualization] Loading {len(cache_files)} combined cache file(s)")
        
        # Load all cloud pixels and footprints from combined caches
        
        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    combined_data = pickle.load(f)
                
                # Extract cloud mask data
                if combined_data['lon'] is not None and combined_data['lat'] is not None:
                    all_cloud_lon = combined_data['lon']
                    all_cloud_lat = combined_data['lat']
                    all_cloudy_mask = combined_data['cloud_flag'] == 1
                    all_uncertain_mask = combined_data['cloud_flag'] == 0
                else:
                    all_cloud_lon = np.array([], dtype=np.float32)
                    all_cloud_lat = np.array([], dtype=np.float32)
                    all_cloudy_mask = np.array([], dtype=bool)
                    all_uncertain_mask = np.array([], dtype=bool)
                
                # Extract OCO-2 footprints
                if combined_data['oco2_fp_lons'] is not None and combined_data['oco2_fp_lats'] is not None:
                    oco_fp_lons = combined_data['oco2_fp_lons']
                    oco_fp_lats = combined_data['oco2_fp_lats']
                    oco_fp_ND_mask = combined_data['oco2_fp_viewing_modes'] == 'ND'
                    oco_fp_GL_mask = combined_data['oco2_fp_viewing_modes'] == 'GL'
                else:
                    oco_fp_lons = np.array([], dtype=np.float32)
                    oco_fp_lats = np.array([], dtype=np.float32)
                    oco_fp_ND_mask = np.array([], dtype=bool)
                    oco_fp_GL_mask = np.array([], dtype=bool)
                    
                print("combined_data['oco2_fp_view_modes'] set:", set(combined_data.get('oco2_fp_viewing_modes', [])))
            
            except Exception as e:
                logger.warning(f"Failed to load {cache_file.name}: {e}")
                continue
        
        if all_cloud_lon.size == 0 or all_cloud_lat.size == 0:
            logger.warning(f"No cloud pixels loaded from cache files")
            return {}
        
        logger.info(f"✓ Loaded {len(all_cloud_lon)} total cloud pixels")
        
        if len(oco_fp_lons) > 0 and len(oco_fp_lats) > 0:
            logger.info(f"✓ Loaded {len(oco_fp_lons)} OCO-2 footprints from combined cache")
        
        
        # Group pixels by latitude band
        lat_bands = {}
        # for pixel in all_pixels:
        #     # Determine latitude band (e.g., -90 to -80, -80 to -70, etc.)
        #     lat_band_idx = int(np.floor(pixel.latitude / lat_band_size))
        #     lat_band_min = lat_band_idx * lat_band_size
        #     lat_band_max = lat_band_min + lat_band_size
        #     lat_band_key = f"{lat_band_min:+.0f}to{lat_band_max:+.0f}"
            
        #     if lat_band_key not in lat_bands:
        #         lat_bands[lat_band_key] = {
        #             'lat_min': lat_band_min,
        #             'lat_max': lat_band_max,
        #             'cloudy': [],
        #             'uncertain': [],
        #             'all': []
        #         }
            
        #     lat_bands[lat_band_key]['all'].append(pixel)
        #     if pixel.cloud_flag == 'Cloudy':
        #         lat_bands[lat_band_key]['cloudy'].append(pixel)
        #     elif pixel.cloud_flag == 'Uncertain':
        #         lat_bands[lat_band_key]['uncertain'].append(pixel)
                
        latband_levels = np.arange(-90, 90 + lat_band_size, lat_band_size)
        latband_indices = np.digitize(all_cloud_lat, latband_levels) - 1  # Get band index for each pixel
        latband_groupnames = [f"{latband_levels[i]:+.0f}to{latband_levels[i+1]:+.0f}" for i in range(len(latband_levels)-1)]
        for i in range(len(latband_levels)-1):
            lat_band_key = latband_groupnames[i]
            lat_band_min = latband_levels[i]
            lat_band_max = latband_levels[i+1]
            
            band_mask = (latband_indices == i)
            if not np.any(band_mask):
                continue
            
            if lat_band_key not in lat_bands:
                lat_bands[lat_band_key] = {
                    'lat_min': lat_band_min,
                    'lat_max': lat_band_max,
                    'cld_lon': all_cloud_lon[band_mask],
                    'cld_lat': all_cloud_lat[band_mask],
                    'cloudy_mask': all_cloudy_mask[band_mask],
                    'uncertain_mask': all_uncertain_mask[band_mask],
                }
        
        # Create visualization for each latitude band
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n[Latitude Band Visualization] Creating plots for {len(lat_bands)} latitude band(s)")
        
        results = {}
        for lat_band_key in sorted(lat_bands.keys()):
            lat_band_data = lat_bands[lat_band_key]
            
            try:
                # Create figure
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                
                crosses_dateline_final = False  # Flag to track if we need to adjust for date line crossing
                
                # Plot cloudy pixels
                if lat_band_data['cloudy_mask'].any():
                    cloudy_lons = lat_band_data['cld_lon'][lat_band_data['cloudy_mask']]
                    cloudy_lats = lat_band_data['cld_lat'][lat_band_data['cloudy_mask']]
                    cloudy_lon_min, cloudy_lon_max = min(cloudy_lons), max(cloudy_lons)
                    print(f"Latitude band {lat_band_key}: Cloudy pixel longitude range [{cloudy_lon_min:.1f}, {cloudy_lon_max:.1f}]")
                    
                    # Check for date line crossing
                    crosses_dateline_cloud = min(cloudy_lons) < -90 and max(cloudy_lons) > 90
                
                else:
                    crosses_dateline_cloud = False
                
                
                # Plot uncertain pixels
                if lat_band_data['uncertain_mask'].any():
                    uncertain_lons = lat_band_data['cld_lon'][lat_band_data['uncertain_mask']]
                    uncertain_lats = lat_band_data['cld_lat'][lat_band_data['uncertain_mask']]
                    uncertain_lon_min, uncertain_lon_max = min(uncertain_lons), max(uncertain_lons)
                    print(f"Latitude band {lat_band_key}: Uncertain pixel longitude range [{uncertain_lon_min:.1f}, {uncertain_lon_max:.1f}]")
                    
                    # Check for date line crossing
                    crosses_dateline_uncertain = min(uncertain_lons) < -90 and max(uncertain_lons) > 90
                else:
                    crosses_dateline_uncertain = False
                        
                crosses_dateline_final = crosses_dateline_cloud or crosses_dateline_uncertain
                
                if lat_band_data['cloudy_mask'].any():
                    
                    # Check for date line crossing
                    crosses_dateline = min(cloudy_lons) < -90 and max(cloudy_lons) > 90
                    
                    if crosses_dateline_final:
                        cloudy_lons = [lon + 360 if lon < 0 else lon for lon in cloudy_lons]
                    
                    ax.scatter(cloudy_lons, cloudy_lats, c='red', s=30, alpha=0.7,
                              label=f'Cloudy ({np.sum(lat_band_data["cloudy_mask"])})', marker='s')
                
                # Plot uncertain pixels
                if lat_band_data['uncertain_mask'].any():                    
                    # Check for date line crossing
                    if crosses_dateline_final:
                        uncertain_lons = [lon + 360 if lon < 0 else lon for lon in uncertain_lons]
                    
                    ax.scatter(uncertain_lons, uncertain_lats, c='orange', s=30, alpha=0.7,
                              label=f'Uncertain ({np.sum(lat_band_data["uncertain_mask"])})', marker='^')
                
                # Plot OCO-2 footprints for this latitude band
                if len(oco_fp_lons) > 0 and len(oco_fp_lats) > 0:
                    # Filter footprints by latitude band
                    band_lat_mask = (oco_fp_lats >= lat_band_data['lat_min']) & (oco_fp_lats < lat_band_data['lat_max'])
                    band_fp_lons = oco_fp_lons[band_lat_mask]
                    band_fp_lats = oco_fp_lats[band_lat_mask]
                    band_fp_ND_mask = oco_fp_ND_mask[band_lat_mask]
                    band_fp_GL_mask = oco_fp_GL_mask[band_lat_mask]
                    
                    
                    if band_lat_mask.any():
                        
                        gl_fp_lons = band_fp_lons[band_fp_GL_mask].copy()  # GL mode
                        gl_fp_lats = band_fp_lats[band_fp_GL_mask].copy()
                        nd_fp_lons = band_fp_lons[band_fp_ND_mask].copy()  # ND mode
                        nd_fp_lats = band_fp_lats[band_fp_ND_mask].copy()
                        
                        # Plot glint footprints
                        if len(gl_fp_lons):
                            if crosses_dateline_final:
                                gl_fp_lons = [gl_fp_lons[i] + 360 if gl_fp_lons[i] < 0 else gl_fp_lons[i] for i in range(len(gl_fp_lons))]
                            ax.scatter(gl_fp_lons, gl_fp_lats, c='blue', s=100, alpha=0.8,
                                      marker='o', edgecolors='darkblue', linewidth=2,
                                      label=f'OCO-2 Glint ({len(gl_fp_lons)})', zorder=5)
                        
                        # Plot nadir footprints
                        if len(nd_fp_lons):
                            if crosses_dateline_final:
                                nd_fp_lons = [nd_fp_lons[i] + 360 if nd_fp_lons[i] < 0 else nd_fp_lons[i] for i in range(len(nd_fp_lons))]
                            ax.scatter(nd_fp_lons, nd_fp_lats, c='green', s=100, alpha=0.8,
                                      marker='D', edgecolors='darkgreen', linewidth=2,
                                      label=f'OCO-2 Nadir ({len(nd_fp_lons)})', zorder=5)
                
                # Set title and labels
                ax.set_title(f'MODIS Cloud Pixels: Latitude Band [{lat_band_data["lat_min"]:.0f}° to {lat_band_data["lat_max"]:.0f}°]',
                            fontsize=14, fontweight='bold', pad=20)
                
                xtick_interval = 1  # degrees
                if crosses_dateline_final:
                    print("xmin before adjustment:", min(cloudy_lons), "xmax before adjustment:", max(all_cloud_lon))
                    xmin = np.floor(min(cloudy_lons) / xtick_interval) * xtick_interval
                    xmax = np.ceil(max(cloudy_lons) / xtick_interval) * xtick_interval
                    print("xmin after adjustment:", xmin, "xmax after adjustment:", xmax)
                    xticks = np.arange(xmin, xmax + 0.1, xtick_interval)
                    xtick_labels = [f"{(x if x <= 180 else x - 360):.0f}°" for x in xticks]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xtick_labels, fontsize=10)
                    ax.set_xlabel('Longitude (degrees)', fontsize=12)
                    
                else:
                    xmin = np.floor(min(cloudy_lons) / xtick_interval) * xtick_interval
                    xmax = np.ceil(max(cloudy_lons) / xtick_interval) * xtick_interval
                    xticks = np.arange(xmin, xmax + 0.1, xtick_interval)
                    ax.set_xticks(xticks)
                    ax.set_xlabel('Longitude (degrees)', fontsize=12)
                ax.set_ylabel('Latitude (degrees)', fontsize=12)
                
                ax.legend(loc='best', fontsize=11, framealpha=0.9)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Add statistics text box
                band_footprints_count = 0
                if band_lat_mask.any():
                    band_footprints_count = np.sum(band_lat_mask)
                
                stats_text = f"""Latitude Band: [{lat_band_data["lat_min"]:.0f}°, {lat_band_data["lat_max"]:.0f}°]

MODIS Cloud Pixels:
  Cloudy:     {np.sum(lat_band_data["cloudy_mask"])}
  Uncertain:  {np.sum(lat_band_data["uncertain_mask"])}

OCO-2 Footprints:
  Total:      {band_footprints_count}"""
                fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                        verticalalignment='bottom')
                
                # Save figure
                filename = f"latband_{lat_band_key}.png"
                filepath = output_path / filename
                plt.savefig(filepath, bbox_inches='tight', dpi=dpi)
                logger.info(f"✓ Saved visualization: {filename}")
                
                plt.close(fig)
                results[lat_band_key] = filepath
                
            except Exception as e:
                logger.error(f"Failed to create visualization for {lat_band_key}: {e}")
                plt.close('all')
        
        logger.info(f"✓ Created {len(results)} latitude band visualization(s) in {output_dir}")
        return results
    

def main():
    """Demo script for Phase 3."""
    from phase_02_ingestion import DataIngestionManager
    
    # Initialize processors
    ingestion = DataIngestionManager(output_dir="../data")
    processor = SpatialProcessor(data_dir="../data")
    
    # Get downloaded files from Phase 2 (would normally use real data)
    print("\nPhase 3: Spatial and Bitmask Processing")
    print("=" * 70)
    
    logger.info("\n[Demo] Phase 3 workflow initialized")
    logger.info("  To use with real data:")
    logger.info("    1. Run Phase 2 to download files")
    logger.info("    2. Extract footprints with: processor.extract_oco2_footprints(oco2_files)")
    logger.info("    3. Extract cloud mask with: processor.extract_modis_cloud_mask(modis_files)")
    logger.info("    4. Temporal matching with: processor.match_temporal_windows(footprints, modis_files)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
