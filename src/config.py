"""
Configuration file for OCO-2/MODIS collocation analysis
========================================================

Contains URLs, constants, and configuration parameters used across all phases.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:
    """Configuration for a specific satellite dataset."""
    short_name: str
    version: str
    base_url: str
    description: str


class Config:
    """Global configuration for the OCO-2/MODIS collocation workflow."""
    
    # ========================
    # NASA Earthdata Settings
    # ========================
    EARTHDATA_LOGIN_URL = "urs.earthdata.nasa.gov"
    
    # ========================
    # OCO-2 Dataset Settings
    # ========================
    OCO2_L1B_SCIENCE = DatasetConfig(
        short_name="OCO2_L1B_Science_11r",
        version="11r (before 2024-04-01, DOY < 92) / 11.2r (from 2024-04-01, DOY >= 92)",
        base_url="https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L1B_Science.{VERSION}/",
        description="OCO-2 Level 1B calibrated, geolocated science spectra"
    )
    
    OCO2_L2_LITE = DatasetConfig(
        short_name="OCO2_L2_Lite_FP_11.1r",
        version="11.1r (before 2024-04-01, DOY < 92) / 11.2r (from 2024-04-01, DOY >= 92)",
        base_url="https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L2_Lite_FP.{VERSION}/",
        description="OCO-2 Level 2 bias-corrected XCO2 and other variables"
    )
    
    OCO2_L2_MET = DatasetConfig(
        short_name="OCO2_L2_Met_11r",
        version="11r (before 2024-04-01, DOY < 92) / 11.2r (from 2024-04-01, DOY >= 92)",
        base_url="https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L2_Met.{VERSION}/",
        description="OCO-2 Level 2 meteorological data"
    )
    
    OCO2_L2_CO2PRIOR = DatasetConfig(
        short_name="OCO2_L2_CO2Prior_11r",
        version="11r (before 2024-04-01, DOY < 92) / 11.2r (from 2024-04-01, DOY >= 92)",
        base_url="https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L2_CO2Prior.{VERSION}/",
        description="OCO-2 Level 2 CO2 prior profiles"
    )
    
    # ========================
    # MODIS Dataset Settings
    # ========================
    MODIS_MYD35_L2 = DatasetConfig(
        short_name="MYD35_L2",
        version="6.1",
        base_url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MYD35_L2/",
        description="MODIS/Aqua Cloud Mask (5 km resolution)"
    )
    
    MODIS_MYD03 = DatasetConfig(
        short_name="MYD03",
        version="6.1",
        base_url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MYD03/",
        description="MODIS/Aqua Geolocation (1 km resolution)"
    )
    
    # ========================
    # CMR Search Settings
    # ========================
    CMR_GRANULE_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.xml"
    CMR_COLLECTION_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/collections.xml"
    
    # ========================
    # Temporal Settings
    # ========================
    # Aqua is in free-drift phase (2023+), requiring larger temporal windows
    MODIS_TEMPORAL_BUFFER_MINUTES = 20  # ±20 minutes for MODIS granule matching
    
    # ========================
    # Spatial Settings
    # ========================
    EARTH_RADIUS_KM = 6371.0  # Mean Earth radius in kilometers
    
    # MODIS resolution
    MODIS_MYD03_RESOLUTION_KM = 1.0  # 1 km nadir resolution
    MODIS_MYD35_RESOLUTION_KM = 5.0  # 5 km resolution (insufficient for our needs)
    
    # OCO-2 footprint size
    OCO2_FOOTPRINT_SIZE_KM = 2.25  # ~1.29 x 2.25 km footprint (along-track x across-track)
    
    # ========================
    # Processing Settings
    # ========================
    # MODIS cloud mask bit positions (Byte 1)
    MODIS_CLOUD_MASK_BYTE = 0  # First byte of 48-bit mask
    MODIS_CLOUD_BITS = (1, 2)  # Bits 1-2 indicate cloud status
    
    # Cloud classification
    CLOUD_CONFIDENT = 0b11  # Bits 1-2 = 11 (cloudy)
    CLOUD_PROBABLY = 0b10   # Bits 1-2 = 10 (probably cloudy)
    CLOUD_UNCERTAIN = 0b01  # Bits 1-2 = 01 (uncertain)
    CLOUD_CLEAR = 0b00      # Bits 1-2 = 00 (confident clear)
    
    # KD-Tree settings
    KDTREE_LEAF_SIZE = 40  # Optimize for memory/speed tradeoff
    
    # ========================
    # Output Settings
    # ========================
    OUTPUT_VARIABLES = [
        'sounding_id',
        'latitude',
        'longitude',
        'time',
        'orbit_number',
        'viewing_mode',
        'nearest_cloud_dist_km',
        'modis_time_diff_minutes',
        'cloud_classification'
    ]
    
    # ========================
    # File Paths
    # ========================
    DEFAULT_CACHE_DIR = "./cache"
    DEFAULT_OUTPUT_DIR = "./output"
    
    # System-specific data paths (override with environment variables)
    # Examples:
    #   macOS: /Volumes/ExternalDrive/oco2_data
    #   Linux: /mnt/data/oco2_data
    #   Windows: D:/oco2_data
    #   CURC: /projects/$USER/oco2_data or /pl/active/<project>/oco2_data
    DATA_ROOT_DIRS = {
        'default': './data',
        'local': './data',
        'external': os.environ.get('OCO2_DATA_ROOT', './data'),
        'scratch': os.environ.get('SCRATCH_DIR', './workspace/scratch_data'),
        'shared': os.environ.get('SHARED_DATA_ROOT', '/shared/oco2_data'),
        'curc': os.environ.get('CURC_DATA_ROOT', os.environ.get('OCO2_DATAROOT', './data'))
    }
    
    @classmethod
    def get_data_path(cls, storage_type: str = 'default') -> str:
        """
        Get the data storage path based on storage type.
        
        Args:
            storage_type: 'default', 'local', 'external', 'scratch', 'shared', or 'curc'
        
        Returns:
            Absolute path to data directory
        
        Environment Variables:
            OCO2_DATA_ROOT: Override for 'external' storage
            SCRATCH_DIR: Override for 'scratch' storage
            SHARED_DATA_ROOT: Override for 'shared' storage
            CURC_DATA_ROOT: Override for 'curc' storage (defaults to PROJECTS env var)
        """
        return cls.DATA_ROOT_DIRS.get(storage_type, cls.DATA_ROOT_DIRS['default'])
    
    @classmethod
    def get_dataset_config(cls, dataset_name: str) -> Optional[DatasetConfig]:
        """
        Get configuration for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'OCO2_L1B', 'MYD35')
        
        Returns:
            DatasetConfig object or None if not found
        """
        dataset_map = {
            'OCO2_L1B': cls.OCO2_L1B_SCIENCE,
            'OCO2_L2_LITE': cls.OCO2_L2_LITE,
            'OCO2_L2_MET': cls.OCO2_L2_MET,
            'OCO2_L2_CO2PRIOR': cls.OCO2_L2_CO2PRIOR,
            'MYD35': cls.MODIS_MYD35_L2,
            'MYD03': cls.MODIS_MYD03
        }
        return dataset_map.get(dataset_name.upper())


# Coordinate system constants
class CoordinateSystem:
    """Constants for coordinate transformations."""
    
    # WGS84 ellipsoid parameters
    WGS84_A = 6378137.0  # Semi-major axis (meters)
    WGS84_B = 6356752.314245  # Semi-minor axis (meters)
    WGS84_F = (WGS84_A - WGS84_B) / WGS84_A  # Flattening
    WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # Eccentricity squared
    
    @staticmethod
    def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0):
        """
        Convert geodetic coordinates (lat, lon, alt) to ECEF (X, Y, Z).
        
        This avoids polar distortions in distance calculations.
        
        Args:
            lat_deg: Latitude in degrees
            lon_deg: Longitude in degrees
            alt_m: Altitude above ellipsoid in meters (default: 0)
        
        Returns:
            Tuple of (X, Y, Z) in meters
        """
        import numpy as np
        
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)
        
        # Radius of curvature in prime vertical
        N = CoordinateSystem.WGS84_A / np.sqrt(
            1 - CoordinateSystem.WGS84_E2 * np.sin(lat_rad) ** 2
        )
        
        X = (N + alt_m) * np.cos(lat_rad) * np.cos(lon_rad)
        Y = (N + alt_m) * np.cos(lat_rad) * np.sin(lon_rad)
        Z = (N * (1 - CoordinateSystem.WGS84_E2) + alt_m) * np.sin(lat_rad)
        
        return X, Y, Z
