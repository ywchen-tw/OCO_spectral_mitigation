"""Shared helpers for the demo_combined pipeline (banners, Lite-file
version handling and downstream-cache invalidation, CLI validation,
storage-dir resolution, MODIS cleanup).

Split out of demo_combined.py (2026-07, review §7.4); demo_combined
re-imports every name, so its behaviour and CLI are unchanged.
"""

import logging
import platform
import re
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def print_banner(text: str):
    """Print a formatted banner."""
    width = 70
    logger.info("=" * width)
    logger.info(text.center(width))
    logger.info("=" * width)


def print_step_header(step_num: int, title: str):
    """Print step header."""
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP {step_num}: {title}".ljust(70))
    logger.info(f"{'='*70}")


LITE_VERSION_RANK = {
    "10r": 0,
    "11r": 1,
    "11.1r": 2,
    "11.2r": 3,
    "11.3r": 4,
}


def _decode_hdf5_attr(value) -> str:
    """Return a readable string for scalar or one-element HDF5 attributes."""
    arr = np.asarray(value)
    if arr.shape:
        value = arr.flat[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def infer_lite_version(filepath: Path) -> str:
    """Infer the Lite collection version from metadata, then filename."""
    text_parts = [filepath.name]
    try:
        with h5py.File(filepath, "r") as h5f:
            for attr_name in (
                "gesdisc_collection",
                "BuildId",
                "CollectionLabel",
                "lite_definition_module",
                "bc_function",
            ):
                if attr_name in h5f.attrs:
                    text_parts.append(_decode_hdf5_attr(h5f.attrs[attr_name]))
    except (OSError, RuntimeError, ValueError):
        return "unreadable"

    text = " ".join(text_parts).lower()
    version_markers = (
        ("11.3r", ("11.3r", "b11.3", "b113", "lite_b113", "b11_3")),
        ("11.2r", ("11.2r", "b11.2", "b112", "lite_b112", "b11_2")),
        ("11.1r", ("11.1r", "b11.1", "b111", "lite_b111", "b11_1")),
        ("11r", ("11r", "b11.0", "b110", "lite_b110")),
        ("10r", ("10r", "b10")),
    )
    for version, markers in version_markers:
        if any(marker in text for marker in markers):
            return version
    return "unknown"


def lite_version_is_before(version: Optional[str], minimum_version: str) -> bool:
    """Treat unknown/unreadable Lite versions as stale for recompute decisions."""
    if not version or version in {"unknown", "unreadable"}:
        return True
    return LITE_VERSION_RANK.get(version, -1) < LITE_VERSION_RANK[minimum_version]


def select_local_lite_file(data_dir: Path, target_date: datetime) -> Tuple[Optional[Path], Optional[str]]:
    """Select the local Lite file using the same date-aware policy as fitting.py."""
    lite_dir = data_dir / "OCO2" / str(target_date.year) / f"{target_date.timetuple().tm_yday:03d}"
    nc4_files = sorted(lite_dir.glob("*.nc4"))
    if not nc4_files:
        return None, None

    preferred_versions = (
        ["11.3r", "11.2r", "11.1r", "11r", "10r"]
        if target_date.year >= 2024
        else ["11.2r", "11.1r", "11r", "10r", "11.3r"]
    )
    version_priority = {version: rank for rank, version in enumerate(preferred_versions)}
    version_priority["unknown"] = 99
    version_priority["unreadable"] = 100

    candidates = []
    for path in nc4_files:
        version = infer_lite_version(path)
        candidates.append((version_priority.get(version, 99), path.name, path, version))
    candidates.sort()
    _, _, selected_path, selected_version = candidates[0]
    return selected_path, selected_version


def invalidate_lite_downstream_cache(data_dir: Path, target_date: datetime) -> int:
    """Remove derived per-date caches that depend on Lite sounding IDs/vertices."""
    processing_day_dir = data_dir / "processing" / str(target_date.year) / f"{target_date.timetuple().tm_yday:03d}"
    if not processing_day_dir.exists():
        return 0

    patterns = (
        "lite_sounding_ids.pkl",
        "lite_vertex_data.pkl",
        "*/footprints.pkl",
        "*/granule_combined_*.pkl",
        "*/phase4_results.pkl",
    )
    removed = 0
    for pattern in patterns:
        for path in processing_day_dir.glob(pattern):
            try:
                path.unlink()
                removed += 1
                logger.info("Removed Lite-stale cache: %s", path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning("Could not remove stale cache %s: %s", path, exc)
    return removed


def delete_lite_files_before(data_dir: Path, target_date: datetime, minimum_version: str) -> int:
    """Delete recognized local Lite files older than the requested minimum version."""
    lite_dir = data_dir / "OCO2" / str(target_date.year) / f"{target_date.timetuple().tm_yday:03d}"
    if not lite_dir.exists():
        return 0

    removed = 0
    minimum_rank = LITE_VERSION_RANK[minimum_version]
    for path in sorted(lite_dir.glob("*.nc4")):
        version = infer_lite_version(path)
        if version not in LITE_VERSION_RANK:
            logger.warning(
                "Keeping Lite file with unknown/unreadable version: %s (%s)",
                path,
                version,
            )
            continue
        if LITE_VERSION_RANK[version] >= minimum_rank:
            continue

        try:
            path.unlink()
            removed += 1
            logger.info("Deleted old Lite file: %s (%s)", path, version)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Could not delete old Lite file %s: %s", path, exc)

    return removed


def validate_date(date_str: str) -> datetime:
    """Validate and parse date string."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def parse_orbit_arg(orbit_str: str) -> Tuple[int, Optional[str]]:
    """Parse --orbit argument into (orbit_number, granule_suffix).

    Accepts plain orbit numbers ('31017') or orbit+suffix ('31017b').
    Returns the integer orbit number and the optional lowercase suffix letter.
    """
    m = re.match(r'^(\d+)([a-z]?)$', orbit_str)
    if not m:
        raise ValueError(
            f"Invalid --orbit value '{orbit_str}'. "
            "Use a plain orbit number (e.g. 31017) or add a suffix letter (e.g. 31017b)."
        )
    return int(m.group(1)), m.group(2) or None


def get_storage_dir() -> Path:
    """Determine storage directory based on platform."""
    if platform.system() == "Darwin":
        logger.info("Detected macOS - using local data directory")
        return Path(Config.get_data_path('local'))
    elif platform.system() == "Linux":
        logger.info("Detected Linux - using CURC storage directory")
        return Path(Config.get_data_path('curc'))
    else:
        logger.warning(f"Unknown platform: {platform.system()}. Using default.")
        return Path(Config.get_data_path('default'))


def cleanup_modis_data(target_date: datetime, data_dir: Path) -> bool:
    """Delete MODIS data files for a specific date to save disk space.
    
    Args:
        target_date: Target date for cleanup
        data_dir: Data storage directory
    
    Returns:
        Success flag
    """
    logger.info("\n" + "="*70)
    logger.info("CLEANUP: Deleting MODIS Data")
    logger.info("="*70)
    
    try:
        data_dir = Path(data_dir)
        doy = target_date.timetuple().tm_yday
        year = target_date.year
        
        # MODIS directories to delete
        myd35_dir = data_dir / "MODIS" / "MYD35_L2" / str(year) / f"{doy:03d}"
        myd03_dir = data_dir / "MODIS" / "MYD03" / str(year) / f"{doy:03d}"
        
        deleted_size = 0
        deleted_files = 0
        
        for modis_dir in [myd35_dir, myd03_dir]:
            if modis_dir.exists():
                # Calculate size before deletion
                for file in modis_dir.glob("*.hdf"):
                    deleted_size += file.stat().st_size
                    deleted_files += 1
                
                # Delete directory
                shutil.rmtree(modis_dir)
                logger.info(f"✓ Deleted: {modis_dir}")
            else:
                logger.info(f"⊘ Not found: {modis_dir}")
        
        if deleted_files > 0:
            deleted_size_mb = deleted_size / (1024 * 1024)
            logger.info(f"\n✓ Cleanup Complete: Deleted {deleted_files} file(s), freed {deleted_size_mb:.1f} MB")
        else:
            logger.info("\n⊘ No MODIS files found to delete")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Cleanup Failed: {e}")
        traceback.print_exc()
        return False

