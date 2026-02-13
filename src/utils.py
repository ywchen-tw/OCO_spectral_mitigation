"""
Utility functions for OCO-2/MODIS collocation analysis
======================================================

Common helper functions used across different phases.
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def format_bytes(size_bytes: int) -> str:
    """
    Format byte size as human-readable string.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def datetime_to_unix_timestamp(dt: datetime) -> float:
    """
    Convert datetime to Unix timestamp.
    
    Args:
        dt: datetime object
    
    Returns:
        Unix timestamp (seconds since epoch)
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def unix_timestamp_to_datetime(timestamp: float) -> datetime:
    """
    Convert Unix timestamp to datetime.
    
    Args:
        timestamp: Unix timestamp (seconds since epoch)
    
    Returns:
        datetime object in UTC
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, 
                       lat2: np.ndarray, lon2: np.ndarray,
                       earth_radius_km: float = 6371.0) -> np.ndarray:
    """
    Calculate great circle distance between points using Haversine formula.
    
    Note: For precise distance calculations avoiding polar distortions,
    use ECEF coordinates instead.
    
    Args:
        lat1, lon1: First point(s) in degrees
        lat2, lon2: Second point(s) in degrees
        earth_radius_km: Earth radius in kilometers
    
    Returns:
        Distance in kilometers
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = (np.sin(dlat / 2) ** 2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return earth_radius_km * c


def ecef_distance(x1: np.ndarray, y1: np.ndarray, z1: np.ndarray,
                  x2: np.ndarray, y2: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distance in ECEF coordinates.
    
    This method avoids polar distortions present in lat/lon calculations.
    
    Args:
        x1, y1, z1: First point(s) in ECEF meters
        x2, y2, z2: Second point(s) in ECEF meters
    
    Returns:
        Distance in meters
    """
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    return np.sqrt(dx**2 + dy**2 + dz**2)


def extract_bits(value: int, start_bit: int, num_bits: int = 1) -> int:
    """
    Extract specific bits from an integer value.
    
    Args:
        value: Integer value to extract from
        start_bit: Starting bit position (0-indexed, LSB = 0)
        num_bits: Number of bits to extract
    
    Returns:
        Extracted bits as integer
    """
    mask = (1 << num_bits) - 1
    return (value >> start_bit) & mask


def parse_modis_cloud_mask(byte_value: int) -> dict:
    """
    Parse MODIS cloud mask byte to extract cloud information.
    
    The MODIS MYD35_L2 cloud mask uses a 48-bit structure.
    Byte 1, bits 1-2 indicate cloud determination:
    - 00: Confident clear
    - 01: Probably clear
    - 10: Probably cloudy
    - 11: Confident cloudy
    
    Args:
        byte_value: First byte of the cloud mask
    
    Returns:
        Dictionary with cloud classification
    """
    # Extract bits 1-2 (cloud mask)
    cloud_bits = extract_bits(byte_value, 1, 2)
    
    classifications = {
        0b00: 'confident_clear',
        0b01: 'probably_clear',
        0b10: 'probably_cloudy',
        0b11: 'confident_cloudy'
    }
    
    is_cloudy = cloud_bits in [0b10, 0b11]
    
    return {
        'cloud_bits': cloud_bits,
        'classification': classifications.get(cloud_bits, 'unknown'),
        'is_cloudy': is_cloudy,
        'confidence': 'confident' if cloud_bits in [0b00, 0b11] else 'probable'
    }


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            description: Description of the task
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, n: int = 1):
        """Update progress by n items."""
        self.current += n
        self._log_progress()
    
    def _log_progress(self):
        """Log current progress."""
        if self.total > 0:
            pct = (self.current / self.total) * 100
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            if self.current > 0:
                rate = self.current / elapsed if elapsed > 0 else 0
                eta_seconds = (self.total - self.current) / rate if rate > 0 else 0
                eta_str = f"{eta_seconds:.0f}s"
            else:
                eta_str = "N/A"
            
            logger.info(
                f"{self.description}: {self.current}/{self.total} "
                f"({pct:.1f}%) - ETA: {eta_str}"
            )
    
    def finish(self):
        """Mark as completed."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(
            f"{self.description} completed: {self.total} items in {elapsed:.1f}s"
        )
