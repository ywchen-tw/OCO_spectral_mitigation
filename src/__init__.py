"""
OCO-2/MODIS Footprint Analysis Package
=======================================

A phased workflow for collocating OCO-2 glint-mode footprints
with Aqua-MODIS cloud masks.
"""

__version__ = '0.1.0'
__author__  = 'OCO-2 Analysis Team'

from .pipeline.phase_01_metadata import OCO2MetadataRetriever, OCO2Granule
from .config import Config, DatasetConfig, CoordinateSystem

__all__ = [
    'OCO2MetadataRetriever',
    'OCO2Granule',
    'Config',
    'DatasetConfig',
    'CoordinateSystem',
]
