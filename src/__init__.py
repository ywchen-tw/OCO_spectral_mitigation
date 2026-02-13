"""
OCO-2/MODIS Footprint Analysis Package
=======================================

A phased workflow for collocating OCO-2 glint-mode footprints 
with Aqua-MODIS cloud masks.

Phases:
1. Metadata Acquisition - Retrieve OCO-2 orbital and temporal metadata
2. Data Ingestion - Download OCO-2 and MODIS datasets
3. Data Processing - Extract and filter cloud mask information
4. Geometry Computation - Calculate distances to nearest cloud pixels
5. Synthesis - Merge and export final results
"""

__version__ = '0.1.0'
__author__ = 'OCO-2 Analysis Team'

from .phase_01_metadata import OCO2MetadataRetriever, OCO2Granule
from .config import Config, DatasetConfig, CoordinateSystem

__all__ = [
    'OCO2MetadataRetriever',
    'OCO2Granule',
    'Config',
    'DatasetConfig',
    'CoordinateSystem',
]
