"""
Phase 2: Targeted Data Ingestion
=================================

This module handles the download of OCO-2 and MODIS data products for the
temporal window identified in Phase 1.

Key Functions:
- download_oco2_products: Downloads OCO-2 L1B, L2 Lite, L2 Met, L2 CO2Prior
- download_modis_products: Downloads MODIS MYD35_L2 (cloud mask) and MYD03 (geolocation)
- organize_downloads: Manages file organization by date and orbit

OCO-2 Products:
- L1B Science: Calibrated radiances with geolocation
- L2 Lite: Bias-corrected XCO2 retrievals
- L2 Met: Meteorological data
- L2 CO2Prior: A priori CO2 profiles

MODIS Products:
- MYD35_L2: Cloud mask (48-bit) at 5 km resolution
- MYD03: Geolocation fields at 1 km resolution (required for accurate distance calculations)
"""

import requests
import os
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

# Handle both package and direct imports
try:
    from .config import Config
    from .phase_01_metadata import OCO2Granule, OCO2MetadataRetriever
except ImportError:
    from config import Config
    from phase_01_metadata import OCO2Granule, OCO2MetadataRetriever

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DownloadedFile:
    """Represents a successfully downloaded file."""
    filepath: Path
    product_type: str  # e.g., "OCO2_L1B", "MYD35_L2"
    target_year: int
    target_doy: int
    granule_id: str
    file_size_mb: float
    download_time_seconds: float
    
    def __repr__(self):
        return (f"DownloadedFile({self.product_type}, {self.granule_id}, "
                f"{self.file_size_mb:.2f} MB in {self.download_time_seconds:.1f}s)")


class DataIngestionManager:
    """Manages download and organization of OCO-2 and MODIS data products."""
    
    # LAADS DAAC URL for MODIS data
    LAADS_BASE_URL = "https://ladsweb.modaps.eosdis.nasa.gov"
    LAADS_ARCHIVE_URL = f"{LAADS_BASE_URL}/archive/allData"
    
    # MODIS version and collection
    MODIS_VERSION = "61"  # Collection 6.1
    
    def __init__(self,
                 output_dir: str = "./data",
                 earthdata_username: Optional[str] = None,
                 earthdata_password: Optional[str] = None,
                 laads_token: Optional[str] = None,
                 dry_run: bool = False,
                 storage_type: str = 'default'):
        """
        Initialize the data ingestion manager.
        
        Args:
            output_dir: Base directory for downloaded files (overrides storage_type)
            earthdata_username: NASA Earthdata username (for OCO-2 from GES DISC)
            earthdata_password: NASA Earthdata password
            laads_token: LAADS DAAC application token (for MODIS)
            dry_run: If True, only check file existence without downloading
            storage_type: Storage location type ('default', 'local', 'external', 'scratch', 'shared', 'curc')
        """
        # Set output directory based on storage type or explicit path
        if output_dir == "./data" and storage_type != 'default':
            output_dir = Config.get_data_path(storage_type)
        
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.storage_type = storage_type
        
        if not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if dry_run:
            logger.info(f"🔍 DRY RUN MODE: Will only check file existence (no downloads)")
        logger.info(f"📁 Data directory: {self.output_dir.absolute()}")
        
        # Set up authentication
        self.username = earthdata_username or os.environ.get('EARTHDATA_USERNAME')
        self.password = earthdata_password or os.environ.get('EARTHDATA_PASSWORD')
        self.laads_token = laads_token or os.environ.get('LAADS_TOKEN')
        
        # Create authenticated session for GES DISC (OCO-2)
        self.gesdisc_session = self._create_earthdata_session()
        
        # LAADS session (uses token in headers)
        self.laads_session = requests.Session()
        if self.laads_token:
            self.laads_session.headers.update({'Authorization': f'Bearer {self.laads_token}'})
            logger.info("LAADS DAAC authentication configured")
        else:
            logger.warning("No LAADS token - MODIS downloads may be rate-limited or fail")
        
        # Initialize Phase 1 retriever for metadata
        self.metadata_retriever = OCO2MetadataRetriever(
            earthdata_username=self.username,
            earthdata_password=self.password
        )
        
        # Statistics
        self.download_stats = {
            'total_files': 0,
            'total_bytes': 0,
            'total_time_seconds': 0,
            'failed_downloads': []
        }
    
    def _create_earthdata_session(self) -> requests.Session:
        """
        Create a session with Earthdata authentication.
        
        GES DISC uses OAuth redirects, so we need to handle authentication
        properly with session cookies.
        
        Returns:
            Authenticated requests.Session
        """
        session = requests.Session()
        
        if self.username and self.password:
            # Set up authentication
            session.auth = (self.username, self.password)
            
            # Enable cookie handling for OAuth redirects
            session.trust_env = True
            
            # Test authentication with a simple request
            try:
                # Use a lightweight endpoint to test auth
                test_url = "https://urs.earthdata.nasa.gov/api/users/user"
                response = session.get(test_url, timeout=10)
                if response.status_code == 200:
                    logger.info("GES DISC authentication configured and verified")
                else:
                    logger.warning(f"GES DISC authentication test returned status {response.status_code}")
            except Exception as e:
                logger.warning(f"Could not verify GES DISC authentication: {e}")
                logger.info("GES DISC authentication configured (not verified)")
        else:
            logger.warning("No Earthdata credentials - OCO-2 downloads may fail")
        
        return session
    
    def _check_file_exists_remote(self,
                                   url: str,
                                   session: requests.Session) -> Tuple[bool, float]:
        """
        Check if a file exists on the remote server without downloading.
        
        Args:
            url: URL to check
            session: Requests session to use (for authentication)
        
        Returns:
            Tuple of (exists, file_size_mb)
        """
        try:
            # Use HEAD request to check existence without downloading
            response = session.head(url, timeout=10, allow_redirects=True)
            
            if response.status_code == 200:
                file_size = int(response.headers.get('content-length', 0))
                file_size_mb = file_size / (1024 * 1024)
                return True, file_size_mb
            elif response.status_code == 404:
                return False, 0.0
            else:
                # Try GET with range to check if file exists
                response = session.get(url, headers={'Range': 'bytes=0-0'}, timeout=10)
                if response.status_code in [200, 206]:  # 206 = Partial Content
                    # Try to get content length from Content-Range header
                    content_range = response.headers.get('content-range', '')
                    if content_range:
                        import re
                        match = re.search(r'/(\d+)$', content_range)
                        if match:
                            file_size = int(match.group(1))
                            return True, file_size / (1024 * 1024)
                    return True, 0.0
            # Allow redirects for OAuth authentication
            response = session.get(url, stream=True, timeout=30, allow_redirects=True)
            
            # Check if we got redirected to login page (authentication failed)
            if 'urs.earthdata.nasa.gov' in response.url and 'oauth' in response.url.lower():
                raise requests.exceptions.HTTPError(
                    "Authentication required. Please set EARTHDATA_USERNAME and EARTHDATA_PASSWORD."
                )
            
        except Exception as e:
            logger.debug(f"Error checking remote file: {e}")
            return False, 0.0
    
    def _download_file(self,
                      url: str,
                      output_path: Path,
                      session: requests.Session,
                      chunk_size: int = 8192) -> Tuple[bool, float, float]:
        """
        Download a single file with progress tracking.
        
        Args:
            url: URL to download from
            output_path: Local path to save file
            session: Requests session to use (for authentication)
            chunk_size: Download chunk size in bytes
        
        Returns:
            Tuple of (success, file_size_mb, download_time_seconds)
        """
        # In dry-run mode, only check if file exists remotely
        if self.dry_run:
            exists, file_size_mb = self._check_file_exists_remote(url, session)
            if exists:
                logger.info(f"✓ Remote file exists: {output_path.name} ({file_size_mb:.2f} MB)")
                return True, file_size_mb, 0.0
            else:
                logger.warning(f"✗ Remote file not found: {output_path.name}")
                return False, 0.0, 0.0
        
        try:
            start_time = time.time()
            
            # Make request with stream=True for large files
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Detect HTML auth redirect: GES DISC returns 200 + HTML login page
            # when the session cookie has expired instead of a proper 401/403.
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                raise requests.exceptions.RequestException(
                    f"Server returned HTML instead of data (authentication may have failed "
                    f"or URL is incorrect): {url}"
                )

            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress
            downloaded = 0
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress for large files
                        if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:  # Every 10 MB
                            progress = (downloaded / total_size) * 100
                            logger.debug(f"  {progress:.1f}% ({downloaded / 1e6:.1f} MB)")
            
            download_time = time.time() - start_time
            file_size_mb = downloaded / (1024 * 1024)
            
            logger.info(f"✓ Downloaded {output_path.name} ({file_size_mb:.2f} MB in {download_time:.1f}s)")
            
            return True, file_size_mb, download_time
            
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Failed to download {url}: {e}")
            return False, 0.0, 0.0
    
    def _extract_auth_token_from_url(self, url: str) -> str:
        """
        Extract the authentication token path from a GES DISC URL.
        
        Format: https://oco2.gesdisc.eosdis.nasa.gov/data/{TOKEN}/OCO2_DATA/...
        
        Args:
            url: Full GES DISC URL
        
        Returns:
            Auth token (e.g., '.A4ft5bK203Mc2nXI/')
        """
        import re
        match = re.search(r'/data(/\.[A-Za-z0-9]+/)', url)
        if match:
            return match.group(1)
        return '/'
    
    def _query_ges_disc_directory(self, 
                                  product_url: str,
                                  year: int, 
                                  doy: int,
                                  filename_pattern: str = None,
                                  granule_id: str = None) -> Optional[str]:
        """
        Query GES DISC directory to find the actual filename for a product.
        
        Args:
            product_url: Base URL for the product (with {VERSION} placeholder if needed)
            year: Year
            doy: Day of year
            filename_pattern: Pattern to match in directory listing
            granule_id: OCO-2 granule ID for constructing specific filenames
        
        Returns:
            Full URL to the first matching file, or None if not found
        """
        # Replace version placeholder with actual version
        if '{VERSION}' in product_url:
            # For dates before DOY 92 (2024-04-01), use base version
            # For dates on/after DOY 92, use .2r version
            # Check if year is 2024 and doy >= 92
            if year == 2024 and doy >= 92:
                version = '11.2r'
            elif year > 2024:
                version = '11.2r'
            else:
                # For years before 2024, determine version
                # L2_Lite uses 11.1r before DOY 92
                if 'L2_Lite_FP' in product_url:
                    version = '11.1r'
                else:
                    version = '11r'
            
            product_url = product_url.replace('{VERSION}', version)
        
        # Build directory URL
        if 'L2_Lite' in product_url:
            # L2_Lite has different directory structure: /YYYY/ instead of /YYYY/DOY/
            dir_url = f"{product_url}{year}/"
        else:
            dir_url = f"{product_url}{year}/{doy:03d}/"
        
        logger.debug(f"Querying directory: {dir_url}")
        
        try:
            response = self.gesdisc_session.get(dir_url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML directory listing for .hdf or .nc4 files
            import re
            from datetime import datetime, timedelta
            
            # Match href attributes
            if 'L2_Lite' in product_url:
                # L2_Lite uses .nc4 format
                # Try to construct a date-specific pattern if we have granule info
                if granule_id:
                    # Extract date from granule ID (format: oco2_L1bScGL_22845a_181018_*)
                    # Date is YYMMDD in the granule ID
                    parts = granule_id.split('_')
                    if len(parts) >= 4:
                        date_str = parts[3]  # e.g., "181018"
                        # Look for files with this date - match only valid filename characters
                        pattern = rf'href=["\']+(oco2_LtCO2[\w._-]*{date_str}[\w._-]*\.nc4)["\']+'
                    else:
                        # Fallback to generic pattern
                        pattern = r'href=["\']+(oco2_LtCO2[\w._-]+\.nc4)["\']+'
                else:
                    # Generic pattern
                    pattern = r'href=["\']+(oco2_LtCO2[\w._-]+\.nc4)["\']+'
            else:
                # Other products use .h5 format - match only valid filename characters
                pattern = r'href=["\']+([\w._-]+\.h5)["\']+'
            
            matches = re.findall(pattern, response.text)
            # remove duplicates and sort
            matches = sorted(set(matches))
            
            if matches:
                # For L2_Lite with specific date pattern, prefer exact match
                filename = None
                if 'L2_Lite' in product_url and granule_id:
                    # Check if we found a date-specific file
                    parts = granule_id.split('_')
                    if len(parts) >= 4:
                        date_str = parts[3]
                        date_matches = [m for m in matches if date_str in m]
                        if date_matches:
                            filename = date_matches[0]
                            logger.debug(f"Found date-specific L2 Lite file: {filename}")
                        else:
                            # No date-specific file, use first match (might be yearly aggregate)
                            filename = matches[0]
                            logger.warning(f"No date-specific L2 Lite file found, using: {filename}")
                            logger.warning(f"  This file may contain data from multiple dates")
                    else:
                        filename = matches[0]
                elif granule_id:
                    # For Met/CO2Prior: filter by orbit ID (e.g., "22845a") so each orbit
                    # gets its own file instead of always getting the first in the directory
                    parts = granule_id.split('_')
                    if len(parts) >= 3:
                        orbit_id = parts[2]  # e.g., "22845a"
                        viewing_mode = parts[1][-2:]  # e.g., "GL", "ND", "TG"
                        if viewing_mode not in ['GL', 'ND', 'TG']:
                            viewing_mode = None
                        orbit_matches = [m for m in matches if orbit_id in m]
                        if orbit_matches and viewing_mode:          
                            mode_matches = [m for m in orbit_matches if viewing_mode+'_' in m.upper()]
                            if not mode_matches:
                                logger.warning(
                                    f"No {viewing_mode}-mode file found for orbit {orbit_id} mode {viewing_mode} in:\n"
                                    f"  URL: {dir_url}\n"
                                    f"  Skipping — file may be unavailable or in a different collection."
                                )
                                return None
                            filename = mode_matches[0]
                            logger.debug(f"Found orbit-specific file for {orbit_id} mode {viewing_mode}: {filename}")
                else:
                    logger.debug(f"No matching files found for {granule_id}")
                    return None
                
                # Construct full URL
                file_url = f"{dir_url}{filename}"
                return file_url
            else:
                logger.debug(f"No matching files found in {dir_url}")
                return None
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"Unable to query directory {dir_url}: {e}")
            return None
    
    def download_oco2_granule(self,
                              granule: OCO2Granule,
                              target_year: int,
                              target_doy: int,
                              product_types: List[str] = None,) -> List[DownloadedFile]:
        """
        Download OCO-2 products for a specific granule.
        
        Args:
            granule: OCO2Granule object with metadata
            product_types: List of product types to download
                          ['L1B', 'L2_Lite', 'L2_Met', 'L2_CO2Prior']
                          Default: all products
        
        Returns:
            List of DownloadedFile objects
        """
        if product_types is None:
            product_types = 'L1B', 'L2_Lite', 'L2_Met', 'L2_CO2Prior'

        downloaded_files = []
        
        # Map product types to their base URLs and expected filename patterns
        product_configs = {
            'L1B': {
                'base_url': Config.OCO2_L1B_SCIENCE.base_url,
                'has_direct_url': True,  # Phase 1 provides direct download URL
            },
            'L2_Lite': {
                'base_url': Config.OCO2_L2_LITE.base_url,
                'has_direct_url': False,
            },
            'L2_Met': {
                'base_url': Config.OCO2_L2_MET.base_url,
                'has_direct_url': False,
            },
            'L2_CO2Prior': {
                'base_url': Config.OCO2_L2_CO2PRIOR.base_url,
                'has_direct_url': False,
            }
        }
        
        # Create output directory structure based on product type:
        # - L1B, Met, CO2Prior: data/OCO2/{YYYY}/{DOY}/{granule_id}/
        # - L2_Lite: data/OCO2/{YYYY}/{DOY}/
        year = granule.start_time.year
        doy = granule.start_time.timetuple().tm_yday
        orbit_str = granule.orbit_str  # e.g., "22845a"
        viewing_mode = granule.viewing_mode  # "GL", "ND", or "TG"
        
        for product_type in product_types:
            if product_type not in product_configs:
                logger.warning(f"Unknown product type: {product_type}")
                continue
            
            config = product_configs[product_type]
            
            # Get download URL
            if product_type == 'L1B' and granule.download_url:
                # L1B URL comes directly from Phase 1 metadata
                url = granule.download_url
                # Extract filename from URL
                filename = url.split('/')[-1]
            else:
                # Query directory to find actual filename
                url = self._query_ges_disc_directory(
                    config['base_url'],
                    year,
                    doy,
                    granule_id=granule.granule_id
                )
                
                if not url:
                    logger.warning(f"✗ Could not find {product_type} file in directory")
                    self.download_stats['failed_downloads'].append({
                        'url': 'N/A',
                        'product_type': product_type,
                        'granule_id': granule.granule_id
                    })
                    continue
                
                filename = url.split('/')[-1]
            
            # Determine output path based on product type
            if product_type == 'L2_Lite':
                # L2_Lite: data/OCO2/{YYYY}/{DOY}/
                output_subdir = self.output_dir / "OCO2" / str(year) / f"{doy:03d}"
            else:
                # L1B, Met, CO2Prior: data/OCO2/{YYYY}/{DOY}/{orbit_id}_{mode}/
                # Parse orbit_id and viewing_mode directly from the downloaded filename.
                # Files whose viewing mode is not GL/ND/TG are skipped.
                fname_parts = filename.split('_')
                if len(fname_parts) < 3:
                    logger.warning(f"Cannot parse orbit_id/viewing_mode from '{filename}'; skipping")
                    self.download_stats['failed_downloads'].append({
                        'url': url, 'product_type': product_type, 'granule_id': granule.granule_id
                    })
                    continue
                short_orbit_id = fname_parts[2]
                file_mode = fname_parts[1].upper()[-2:]  # Get last 2 characters for mode
                if file_mode not in ['GL', 'ND', 'TG']:
                    logger.warning(f"Unrecognised viewing mode '{file_mode}' in '{filename}'; skipping")
                    self.download_stats['failed_downloads'].append({
                        'url': url, 'product_type': product_type, 'granule_id': granule.granule_id
                    })
                    continue
                folder_name = f"{short_orbit_id}_{file_mode}"
                output_subdir = self.output_dir / "OCO2" / str(target_year) / f"{target_doy:03d}" / folder_name
            print(f"Output subdir for {product_type}: {output_subdir}")
            sys.exit()               
            if not self.dry_run:
                output_subdir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_subdir / filename
            
            # Check if already downloaded (local file exists)
            if output_path.exists():
                # For HDF5/NetCDF4 files, validate the header to catch corrupted or
                # partially-downloaded files (e.g. HTML auth redirects saved as .nc4).
                if output_path.suffix in ('.nc4', '.h5', '.hdf5') and not self.dry_run:
                    import h5py
                    if not h5py.is_hdf5(str(output_path)):
                        logger.warning(
                            f"  ⚠️  {filename} exists but is not valid HDF5 "
                            f"(possibly corrupted or incomplete download) — re-downloading"
                        )
                        output_path.unlink()
                        # Fall through to download below
                    else:
                        file_size_mb = output_path.stat().st_size / (1024 * 1024)
                        logger.info(f"  ⏭️  File exists locally: {filename} ({file_size_mb:.2f} MB)")
                        downloaded_files.append(
                            DownloadedFile(
                                filepath=output_path,
                                product_type=f"OCO2_{product_type}",
                                target_year=target_year,
                                target_doy=target_doy,
                                granule_id=granule.granule_id,
                                file_size_mb=file_size_mb,
                                download_time_seconds=0.0
                            )
                        )
                        continue
                else:
                    file_size_mb = output_path.stat().st_size / (1024 * 1024)
                    status = "📋" if self.dry_run else "⏭️"
                    logger.info(f"  {status} File exists locally: {filename} ({file_size_mb:.2f} MB)")
                    downloaded_files.append(
                        DownloadedFile(
                            filepath=output_path,
                            product_type=f"OCO2_{product_type}",
                            target_year=target_year,
                            target_doy=target_doy,
                            granule_id=granule.granule_id,
                            file_size_mb=file_size_mb,
                            download_time_seconds=0.0
                        )
                    )
                    continue
            
            # Download the file (or check if it exists remotely in dry-run mode)
            action = "Checking" if self.dry_run else "Downloading"
            logger.info(f"{action} {product_type}: {filename}")
            success, file_size_mb, download_time = self._download_file(
                url=url,
                output_path=output_path,
                session=self.gesdisc_session
            )
            
            if success:
                downloaded_files.append(
                    DownloadedFile(
                        filepath=output_path,
                        product_type=f"OCO2_{product_type}",
                        target_year=target_year,
                        target_doy=target_doy,
                        granule_id=granule.granule_id,
                        file_size_mb=file_size_mb,
                        download_time_seconds=download_time
                    )
                )
                self.download_stats['total_files'] += 1
                self.download_stats['total_bytes'] += file_size_mb * 1024 * 1024
                self.download_stats['total_time_seconds'] += download_time
            else:
                self.download_stats['failed_downloads'].append({
                    'url': url,
                    'product_type': product_type,
                    'granule_id': granule.granule_id
                })
        
        return downloaded_files
    
    def find_modis_granules(self,
                           start_time: datetime,
                           end_time: datetime,
                           buffer_minutes: int = 20,
                           product: str = 'MYD35_L2') -> List[str]:
        """
        Find MODIS granules within the temporal window.
        
        Uses adaptive buffer: ±10 minutes for years < 2023, ±20 minutes for 2023+
        (Aqua orbital drift increased after 2023).
        
        MODIS granules are named with year and day-of-year.
        Each granule is ~5 minutes, so we need to search across the time range.
        
        Args:
            start_time: Start of temporal window
            end_time: End of temporal window
            buffer_minutes: Default temporal buffer (±20 min)
            product: Product to search ('MYD35_L2' or 'MYD03')
        
        Returns:
            List of MODIS granule identifiers
        """
        # Determine observation year for adaptive buffer
        observation_year = start_time.year
        
        # Apply adaptive buffer (Aqua drift increased after 2023)
        effective_buffer = buffer_minutes
        if observation_year < 2022:
            effective_buffer = 10
            logger.info(f"Year {observation_year} < 2022: Using reduced temporal buffer of ±{effective_buffer} minutes")
        else:
            logger.info(f"Year {observation_year} >= 2022: Using standard buffer of ±{effective_buffer} minutes")
        
        # Add buffer
        search_start = start_time - timedelta(minutes=effective_buffer)
        search_end = end_time + timedelta(minutes=effective_buffer)
        
        logger.info(f"Searching {product} granules from {search_start} to {search_end} ({effective_buffer}±min buffer)")
        
        # MODIS granules are organized by date and time
        # Format: MYD35_L2.AYYYYDDD.HHMM.VVV.YYYYDDDHHMMSS.hdf or MYD03.AYYYYDDD.HHMM.061.*.hdf
        # We'll need to query the LAADS archive for the date range
        
        granule_ids = []
        current_date = search_start.date()
        end_date = search_end.date()
        
        import re
        
        while current_date <= end_date:
            year = current_date.year
            doy = current_date.timetuple().tm_yday
            
            # Query LAADS directory for this date
            try:
                granules = self._query_laads_directory(year, doy, product)
                
                # Filter granules by time (extract HHMM from filename)
                # Format: MYD35_L2.AYYYYDDD.HHMM.061.*.hdf
                for granule in granules:
                    # Extract just the filename if it's a URL or path
                    filename = granule.split('/')[-1] if '/' in granule else granule
                    
                    # Extract time from granule filename: A(\d{4})(\d{3}).(\d{4})
                    match = re.search(r'A(\d{4})(\d{3})\.(\d{4})', filename)
                    if match:
                        g_year = int(match.group(1))
                        g_doy = int(match.group(2))
                        hhmm = match.group(3)
                        
                        # Parse time: HHMM format
                        hour = int(hhmm[:2])
                        minute = int(hhmm[2:4])
                        
                        # Create datetime for this granule (UTC timezone, matching MODIS data)
                        from datetime import timezone
                        granule_date = datetime(g_year, 1, 1, tzinfo=timezone.utc) + timedelta(days=g_doy-1)
                        granule_time = granule_date.replace(hour=hour, minute=minute)
                        
                        # Check if within search window (handle both timezone-aware and naive comparisons)
                        if search_start <= granule_time <= search_end:
                            granule_ids.append(filename)  # Store just the filename
                    else:
                        logger.debug(f"Could not parse time from granule: {granule}")
            except Exception as e:
                logger.error(f"Failed to query {product} granules for {year}/{doy:03d}: {e}")
            
            current_date += timedelta(days=1)
        
        # Deduplicate granule IDs (in case same file appears across date boundaries)
        granule_ids = list(set(granule_ids))
        
        logger.info(f"Found {len(granule_ids)} {product} granule(s) within temporal window")
        return granule_ids
    
    def _query_laads_directory(self, year: int, doy: int, product: str) -> List[str]:
        """
        Query LAADS directory listing for available granules.
        
        Args:
            year: Year
            doy: Day of year
            product: Product name (MYD35_L2 or MYD03)
        
        Returns:
            List of granule filenames
        """
        # LAADS directory structure: /archive/allData/{VERSION}/{PRODUCT}/{YEAR}/{DOY}/
        url = f"{self.LAADS_ARCHIVE_URL}/{self.MODIS_VERSION}/{product}/{year}/{doy:03d}/"
        
        logger.debug(f"Querying LAADS directory: {url}")
        
        try:
            response = self.laads_session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML directory listing for .hdf files
            import re
            pattern = r'href=["\']?([^"\'>\s]*\.hdf)["\']?'
            matches = re.findall(pattern, response.text)
            
            # Filter for actual granule files (not XML or metadata) and deduplicate
            granules = list(set([m for m in matches if product in m and m.endswith('.hdf')]))
            
            logger.info(f"  Found {len(granules)} {product} file(s) for {year}/{doy:03d}")
            return granules
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to query LAADS directory {url}: {e}")
            return []
    
    def _check_day_night_from_api(self, granule_filename: str, year: int, doy: int) -> Optional[bool]:
        """
        Check if a MODIS MYD35_L2 file is a day pass using LAADS API metadata.
        
        This avoids downloading the file just to check the day/night flag.
        Uses the LAADS searchGranule API to query metadata.
        
        Args:
            granule_filename: Granule filename (e.g., MYD35_L2.A2018291.0045.*.hdf)
            year: Year
            doy: Day of year
        
        Returns:
            True if day pass, False if night pass, None if unable to determine
        """
        # Extract time from filename for API query
        # Format: MYD35_L2.AYYYYDDD.HHMM.*.hdf
        import re
        match = re.search(r'A(\d{4})(\d{3})\.(\d{4})', granule_filename)
        if not match:
            logger.debug(f"    Could not parse granule filename: {granule_filename}")
            return None
        
        # Construct search API URL
        # LAADS API: https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details
        api_url = f"https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details"
        
        params = {
            'product': 'MYD35_L2',
            'collection': self.MODIS_VERSION,
            'startTime': f"{year}-01-01",  # API expects YYYY-MM-DD
            'endTime': f"{year}-12-31",
            'dateOfYear': doy,
        }
        
        try:
            response = self.laads_session.get(api_url, params=params, timeout=15)
            response.raise_for_status()
            metadata = response.json()
            
            # Search for our specific granule in the results
            if 'content' in metadata:
                for item in metadata['content']:
                    if item.get('name') == granule_filename:
                        day_night = item.get('dayNightFlag', '').upper()
                        if day_night == 'DAY':
                            logger.debug(f"    ✓ API: {granule_filename} is DAY pass")
                            return True
                        elif day_night == 'NIGHT':
                            logger.debug(f"    ⏭️ API: {granule_filename} is NIGHT pass - skipping download")
                            return False
            
            # If not found in API results, return None (will fall back to file check)
            logger.debug(f"    API metadata not available for {granule_filename}")
            return None
            
        except Exception as e:
            logger.debug(f"    Failed to query API metadata: {e}")
            return None
    
    def _is_day_pass(self, filepath: Path) -> Optional[bool]:
        """
        Check if a MODIS MYD35_L2 file is a day pass (ascending track).
        
        Uses Bit 3 of Byte 0 from the cloud mask (1=Day/Ascending, 0=Night/Descending).
        Results are cached to avoid re-reading HDF4 files on subsequent runs.
        
        Args:
            filepath: Path to MYD35_L2 HDF4 file
        
        Returns:
            True if day pass, False if night pass, None if unable to determine
        """
        # Check cache first to avoid re-reading HDF4 file
        cache_file = filepath.parent / f"{filepath.name}.daynight.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    is_day = cache_data.get('is_day_pass')
                    day_pct = cache_data.get('day_percent', 0)
                    pass_type = "DAY (ascending)" if is_day else "NIGHT (descending)"
                    logger.debug(f"    📋 Cached day/night: {pass_type} ({day_pct:.1f}% day pixels)")
                    return is_day
            except Exception as e:
                logger.debug(f"    Cache read failed, will re-check: {e}")
        
        # Cache miss - read HDF4 file
        try:
            from pyhdf.SD import SD, SDC
            import numpy as np
            
            hdf = SD(str(filepath), SDC.READ)
            
            if 'Cloud_Mask' not in hdf.datasets():
                logger.debug(f"    No Cloud_Mask dataset in {filepath.name}")
                return None
            
            cloud_mask_sds = hdf.select('Cloud_Mask')
            mask_data = cloud_mask_sds.get()
            cloud_mask_sds.endaccess()
            hdf.end()
            
            # Extract Byte 0 (first byte)
            if mask_data.ndim == 3:
                if mask_data.shape[0] == 6:
                    byte0 = mask_data[0, :, :]  # Shape: (6, rows, cols)
                elif mask_data.shape[2] == 6:
                    byte0 = mask_data[:, :, 0]  # Shape: (rows, cols, 6)
                else:
                    return None
            else:
                byte0 = mask_data
            
            # Extract Bit 3 (Day/Night flag)
            day_night_flag = (byte0 >> 3) & 0b1
            day_pixels = np.sum(day_night_flag == 1)
            night_pixels = np.sum(day_night_flag == 0)
            
            # Majority vote
            is_day_pass = day_pixels > night_pixels
            
            day_pct = (day_pixels / (day_pixels + night_pixels) * 100) if (day_pixels + night_pixels) > 0 else 0
            pass_type = "DAY (ascending)" if is_day_pass else "NIGHT (descending)"
            logger.info(f"    ✓ Day/Night check: {pass_type} ({day_pct:.1f}% day pixels)")
            
            # Save result to cache for future runs
            try:
                cache_data = {
                    'is_day_pass': bool(is_day_pass),
                    'day_percent': float(day_pct),
                    'day_pixels': int(day_pixels),
                    'night_pixels': int(night_pixels),
                    'checked_at': datetime.utcnow().isoformat()
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                logger.debug(f"    💾 Cached day/night result to {cache_file.name}")
            except Exception as e:
                logger.debug(f"    Failed to cache day/night result: {e}")
            
            return is_day_pass
        
        except Exception as e:
            logger.debug(f"    Could not check day/night status: {e}")
            return None
    
    def download_modis_granule(self,
                              granule_filename: str,
                              product_type: str,
                              year: int,
                              doy: int,
                              target_year: int,
                              target_doy: int,
                              skip_night_passes: bool = False) -> Optional[DownloadedFile]:
        """
        Download a single MODIS granule.
        
        Args:
            granule_filename: Granule filename (e.g., MYD35_L2.A2018291.1835.061.*.hdf)
            product_type: 'MYD35_L2' or 'MYD03'
            year: Year
            doy: Day of year
            skip_night_passes: If True and product is MYD35_L2, skip night/descending passes
        
        Returns:
            DownloadedFile object or None if failed
        """
        # Construct URL
        url = f"{self.LAADS_ARCHIVE_URL}/{self.MODIS_VERSION}/{product_type}/{year}/{doy:03d}/{granule_filename}"
        
        # Create output directory: data/MODIS/{PRODUCT}/{YEAR}/{DOY}/
        output_subdir = self.output_dir / "MODIS" / product_type / str(target_year) / f"{target_doy:03d}"
        if not self.dry_run:
            output_subdir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_subdir / granule_filename
        
        # Check if already downloaded (local file exists)
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            # If filtering night passes and this is MYD35_L2, check if it's a day pass
            if skip_night_passes and product_type == 'MYD35_L2' and not self.dry_run:
                is_day = self._is_day_pass(output_path)
                if is_day == False:  # Use == not 'is' for numpy.bool_ compatibility
                    logger.info(f"  🗑️  Deleting existing night pass: {granule_filename}")
                    try:
                        output_path.unlink()
                        return None
                    except Exception as e:
                        logger.error(f"    Failed to delete night pass file: {e}")
                        return None
            
            status = "📋" if self.dry_run else "⏭️"
            logger.info(f"  {status} File exists locally: {granule_filename} ({file_size_mb:.2f} MB)")
            return DownloadedFile(
                filepath=output_path,
                product_type=product_type,
                target_year=target_year,
                target_doy=target_doy,
                granule_id=granule_filename,
                file_size_mb=file_size_mb,
                download_time_seconds=0.0
            )
        
        # Check if night pass BEFORE downloading (if filtering enabled)
        is_day_api = None
        if skip_night_passes and product_type == 'MYD35_L2' and not self.dry_run:
            # Try API metadata check first (faster, doesn't require download)
            is_day_api = self._check_day_night_from_api(granule_filename, year, doy)
            if is_day_api == False:  # Definitively a night pass
                logger.info(f"  ⏭️ Skipping night pass (API check): {granule_filename}")
                return None
            # If is_day_api is None (API unavailable), continue with download and check afterward
            # If is_day_api is True, we know it's a day pass - skip post-download check
        
        # Download (or check if it exists remotely in dry-run mode)
        action = "Checking" if self.dry_run else "Downloading"
        logger.info(f"{action} {product_type}: {granule_filename}")
        success, file_size_mb, download_time = self._download_file(
            url=url,
            output_path=output_path,
            session=self.laads_session
        )
        
        if success:
            # Check if this is a night pass and should be skipped
            # Only do file-based check if API didn't already confirm it's a day pass
            if skip_night_passes and product_type == 'MYD35_L2' and not self.dry_run and is_day_api is None:
                is_day = self._is_day_pass(output_path)
                if is_day == False:  # Use == not 'is' for numpy.bool_ compatibility
                    logger.info(f"  🗑️  Deleting downloaded night pass: {granule_filename}")
                    try:
                        output_path.unlink()
                        return None
                    except Exception as e:
                        logger.error(f"    Failed to delete night pass file: {e}")
            
            self.download_stats['total_files'] += 1
            self.download_stats['total_bytes'] += file_size_mb * 1024 * 1024
            self.download_stats['total_time_seconds'] += download_time
            
            return DownloadedFile(
                filepath=output_path,
                product_type=product_type,
                target_year=target_year,
                target_doy=target_doy,
                granule_id=granule_filename,
                file_size_mb=file_size_mb,
                download_time_seconds=download_time
            )
        else:
            self.download_stats['failed_downloads'].append({
                'url': url,
                'product_type': product_type,
                'granule_id': granule_filename
            })
            return None
    
    def _check_download_status(self, target_date: datetime) -> Optional[Dict]:
        """
        Check if data for a date has already been downloaded.
        
        Args:
            target_date: Target date to check
        
        Returns:
            Status info if downloaded, None otherwise
        """
        # Construct path to status file directory
        doy = target_date.timetuple().tm_yday
        status_dir = Path(self.output_dir) / "OCO2" / str(target_date.year) / f"{doy:03d}"
        
        if not status_dir.exists():
            return None
        
        # Look for any status file in this directory
        for status_file in status_dir.glob("*/sat_data_status.json"):
            try:
                with open(status_file, 'r') as f:
                    status = json.load(f)
                    if status.get('downloading_completed'):
                        logger.info(f"✓ Found existing download status: {status_file}")
                        return status
            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Could not read status file {status_file}: {e}")
        
        return None
    
    def _list_existing_files(self, target_date: datetime, granule_ids: List[str]) -> Tuple[List, List]:
        """
        List existing downloaded files for a date instead of re-downloading.
        
        Args:
            target_date: Target date
            granule_ids: List of granule IDs that were intended to download
        
        Returns:
            Tuple of (oco2_files, modis_files) lists
        """
        logger.info("Loading existing files from disk...")
        oco2_files = []
        modis_files = []
        doy = target_date.timetuple().tm_yday
        
        # List OCO-2 files
        oco2_dir = Path(self.output_dir) / "OCO2" / str(target_date.year) / f"{doy:03d}"
        if oco2_dir.exists():
            for granule_dir in oco2_dir.iterdir():
                if granule_dir.is_dir():
                    # Skip directories that are named like granule IDs (e.g., oco2_L1bScGL_*.h5/)
                    # These contain status files only. Process only orbit ID directories (e.g., 22845a/)
                    if granule_dir.name.endswith('.h5') or granule_dir.name.endswith('.hdf5') or granule_dir.name.endswith('.nc4'):
                        continue
                    
                    # Look for OCO-2 files (both .h5 and .hdf5 extensions)
                    for hdf_file in list(granule_dir.glob("*.h5")) + list(granule_dir.glob("*.hdf5")):
                        # Skip status files and other non-data files
                        if "status" in hdf_file.name.lower():
                            continue
                        
                        # Determine product type from filename (case-insensitive)
                        filename_upper = hdf_file.name.upper()
                        if 'L1B' in filename_upper:
                            ptype = "L1B"
                        elif 'LTC' in filename_upper or 'LTCO2' in filename_upper:  # L2 Lite CO2
                            ptype = "L2_Lite"
                        elif 'L2MET' in filename_upper:
                            ptype = "L2_Met"
                        elif 'L2CPR' in filename_upper or 'L2CO2' in filename_upper:  # CO2 Prior
                            ptype = "L2_CO2Prior"
                        else:
                            ptype = "Unknown"  # Default fallback
                        
                        # For non-L2_Lite files: granule_id is the actual filename (contains full granule info)
                        # For L2_Lite files: granule_id should be reconstructed or use stem
                        if ptype == "L2_Lite":
                            granule_id = hdf_file.stem  # e.g., "oco2_LtCO2_181018_01B"
                        else:
                            granule_id = hdf_file.stem  # Use filename stem (without .h5)
                        
                        oco2_files.append(DownloadedFile(
                            filepath=hdf_file,
                            product_type=ptype,
                            target_year=target_date.year,
                            target_doy=doy,
                            granule_id=granule_id,
                            file_size_mb=hdf_file.stat().st_size / (1024 * 1024),
                            download_time_seconds=0
                        ))

            # L2 Lite files are stored at the day level (data/OCO2/YYYY/DOY/*.nc4)
            for nc4_file in oco2_dir.glob("*.nc4"):
                if "status" in nc4_file.name.lower():
                    continue

                oco2_files.append(DownloadedFile(
                    filepath=nc4_file,
                    product_type="L2_Lite",
                    target_year=target_date.year,
                    target_doy=doy,
                    granule_id=nc4_file.stem,
                    file_size_mb=nc4_file.stat().st_size / (1024 * 1024),
                    download_time_seconds=0
                ))
        
        # List MODIS files
        modis_myd35_dir = Path(self.output_dir) / "MODIS" / "MYD35_L2" / str(target_date.year) / f"{doy:03d}"
        modis_myd03_dir = Path(self.output_dir) / "MODIS" / "MYD03" / str(target_date.year) / f"{doy:03d}"
        
        for modis_dir in [modis_myd35_dir, modis_myd03_dir]:
            if modis_dir.exists():
                for hdf_file in modis_dir.glob("*.hdf"):
                    product_type = "MYD35_L2" if "MYD35_L2" in hdf_file.name else "MYD03"
                    modis_files.append(DownloadedFile(
                        filepath=hdf_file,
                        product_type=product_type,
                        target_year=target_date.year,
                        target_doy=doy,
                        granule_id=hdf_file.stem,
                        file_size_mb=hdf_file.stat().st_size / (1024 * 1024),
                        download_time_seconds=0
                    ))
        
        logger.info(f"✓ Listed {len(oco2_files)} OCO-2 files and {len(modis_files)} MODIS files")
        return oco2_files, modis_files
    
    def _write_download_status(self, target_date: datetime, granule_ids: List[str], 
                               oco2_files: List, modis_files: List) -> bool:
        """
        Write status file after successful download.
        
        Args:
            target_date: Target date
            granule_ids: List of granule IDs
            oco2_files: List of downloaded OCO-2 files
            modis_files: List of downloaded MODIS files
        
        Returns:
            True if status file written successfully
        """
        try:
            doy = target_date.timetuple().tm_yday
            
            # Write one status file per granule
            for granule_id in granule_ids:
                # Extract short orbit ID (e.g., "22845a") from granule_id
                # Format: oco2_L1bScGL_22845a_181018_B11006r_220921185957.h5
                # This matches the folder structure used in download_oco2_granule()
                gid_parts = granule_id.split('_')
                if len(gid_parts) < 3:
                    logger.warning(f"Cannot parse orbit_id/viewing_mode from granule_id '{granule_id}'; skipping status file")
                    continue
                short_orbit_id = gid_parts[2]
                product_str = gid_parts[1].upper()
                if 'GL' in product_str:
                    folder_name = f"{short_orbit_id}_GL"
                elif 'ND' in product_str:
                    folder_name = f"{short_orbit_id}_ND"
                elif 'TG' in product_str:
                    folder_name = f"{short_orbit_id}_TG"
                else:
                    logger.warning(f"Unrecognised viewing mode in granule_id '{granule_id}'; skipping status file")
                    continue

                status_dir = Path(self.output_dir) / "OCO2" / str(target_date.year) / f"{doy:03d}" / folder_name
                status_dir.mkdir(parents=True, exist_ok=True)

                status_file = status_dir / "sat_data_status.json"
                status_data = {
                    'downloading_completed': True,
                    'download_timestamp': datetime.now().isoformat(),
                    'target_date': target_date.isoformat(),
                    'granule_id': granule_id,
                    'short_orbit_id': short_orbit_id,
                    'oco2_file_count': sum(1 for f in oco2_files if f.filepath.parent.name == folder_name),
                    'modis_file_count': len(modis_files),
                    'total_oco2_files': len(oco2_files),
                    'total_modis_files': len(modis_files)
                }
                
                with open(status_file, 'w') as f:
                    json.dump(status_data, f, indent=2)
                
                logger.info(f"✓ Wrote status file: {status_file}")
            
            return True
        except Exception as e:
            logger.warning(f"Could not write download status file: {e}")
            return False

    def download_all_for_date(self,
                             target_date: datetime,
                             orbit_filter: Optional[str] = None,
                             mode_filter: Optional[str] = None,
                             include_modis: bool = True,
                             limit_granules: Optional[int] = None,
                             skip_existing: bool = True) -> Dict:
        """
        Download all OCO-2 and MODIS products for a specific date.

        Args:
            target_date: Target date for data acquisition
            orbit_filter: Optional specific orbit number to download
            granule_suffix: Optional single-letter suffix to narrow orbit filter (e.g. 'b'
                            selects only the '31017b' granule when orbit_filter=31017)
            mode_filter: Optional viewing mode filter ('GL' or 'ND')
            include_modis: Whether to download MODIS products
            limit_granules: Optional limit to download only first N granules (useful for testing)
            skip_existing: If True, check for existing downloads and skip if already completed

        Returns:
            Dictionary with download summary
        """
        logger.info(f"{'='*70}")
        logger.info(f"Phase 2: Data Ingestion for {target_date.date()}")
        logger.info(f"{'='*70}")

        # Check for existing downloads if requested
        if skip_existing:
            logger.info("\n[Step 0] Checking for existing downloads...")
            existing_status = self._check_download_status(target_date)
            if existing_status:
                logger.info("✓ Previous download status found. Verifying files and downloading any missing ones...")
            # Always fall through to the download loop so per-file existence checks
            # catch any files missing from orbit folders (e.g. after a partial run).

        # Phase 1: Get metadata
        logger.info("\n[Step 1] Retrieving OCO-2 metadata...")
        target_date_1d_prior = target_date - timedelta(days=1)
        xml_contents_1d_prior = self.metadata_retriever.fetch_oco2_xml(target_date_1d_prior)
        granules_1d_prior = self.metadata_retriever.parse_orbit_info(xml_contents_1d_prior)
        xml_contents = self.metadata_retriever.fetch_oco2_xml(target_date)
        granules = self.metadata_retriever.parse_orbit_info(xml_contents)

        granules = granules_1d_prior + granules  # Combine granules from target date and 1 day prior to catch orbits crossing midnight
        
        # Only keep granules that start or end on target date  
        target_date_only = target_date.date()
        target_doy = target_date.timetuple().tm_yday
        granules = [g for g in granules if g.start_time.date() == target_date_only or g.end_time.date() == target_date_only]  
        
        # Filter granules if requested
        if orbit_filter:
            granules = [g for g in granules if g.orbit_str == orbit_filter]
        if mode_filter:
            granules = [g for g in granules if g.viewing_mode == mode_filter]
        
        # Limit granules for testing
        if limit_granules:
            logger.info(f"Limiting to first {limit_granules} granule(s) for testing")
            granules = granules[:limit_granules]

        # Drop granules whose start_time is not on target_date.
        # CMR temporal-overlap queries can return the last orbit of the previous day
        # when it crosses midnight into target_date; that orbit's L1B file and L2 Lite
        # file belong to the previous DOY and should not be downloaded here.
        # target_date_only = target_date.date()
        # off_day = [g for g in granules if g.start_time.date() != target_date_only]
        # if off_day:
        #     logger.warning(
        #         f"Dropping {len(off_day)} granule(s) whose start_time is not on "
        #         f"{target_date_only}: {[g.granule_id for g in off_day]}"
        #     )
        #     granules = [g for g in granules if g.start_time.date() == target_date_only]

        logger.info(f"✓ Found {len(granules)} granule(s) to download")

        if not granules:
            logger.warning("No granules found. Exiting.")
            return {'granules': [], 'oco2_files': [], 'modis_files': []}
        
        # Download OCO-2 products
        logger.info("\n[Step 2] Downloading OCO-2 products...")
        oco2_files = []
        for i, granule in enumerate(granules, 1):
            logger.info(f"\nGranule {i}/{len(granules)}: {granule.granule_id}")
            files = self.download_oco2_granule(granule, 
                                               target_year=target_date.year,
                                               target_doy=target_doy)
            oco2_files.extend(files)
        
        # Download MODIS products
        modis_files = []
        if include_modis:
            logger.info("\n[Step 3] Downloading MODIS products...")
            
            # Get per-granule temporal windows for targeted MODIS searching
            granule_windows = self.metadata_retriever.extract_granule_temporal_windows(granules)
            logger.info(f"Searching MODIS granules for {len(granule_windows)} OCO-2 granule(s)")
            
            # Search for MODIS granules separately for each OCO-2 granule to reduce search space
            myd35_granule_ids = set()
            myd03_granule_ids = set()
            
            for window in granule_windows:
                logger.info(f"\n  OCO-2 granule: {window['granule_id']}")
                logger.info(f"    Time window: {window['start_time']} to {window['end_time']}")
                
                # Find MYD35_L2 granules for this specific OCO-2 granule's time window
                myd35_ids = self.find_modis_granules(window['start_time'], window['end_time'], product='MYD35_L2')
                logger.info(f"    Found {len(myd35_ids)} MYD35_L2 granule(s)")
                myd35_granule_ids.update(myd35_ids)
                
                # Find MYD03 granules for this specific OCO-2 granule's time window
                myd03_ids = self.find_modis_granules(window['start_time'], window['end_time'], product='MYD03')
                logger.info(f"    Found {len(myd03_ids)} MYD03 granule(s)")
                myd03_granule_ids.update(myd03_ids)
            
            # Convert sets back to lists for iteration
            myd35_granule_ids = list(myd35_granule_ids)
            myd03_granule_ids = list(myd03_granule_ids)
            
            # Download MYD35_L2 files (skip night passes for ascending tracks only)
            import re
            kept_myd35_times = set()  # Track which MYD35_L2 files were kept (not deleted)
            
            logger.info(f"\nDownloading {len(myd35_granule_ids)} MYD35_L2 granule(s)")
            for granule_id in myd35_granule_ids:
                match = re.search(r'A(\d{4})(\d{3})\.(\d{4})', granule_id)
                if match:
                    year = int(match.group(1))
                    doy = int(match.group(2))
                    time_str = match.group(3)  # HHMM
                    
                    file_obj = self.download_modis_granule(granule_id, 'MYD35_L2', year, doy, 
                                                           target_year=target_date.year,
                                                           target_doy=target_doy, skip_night_passes=False)
                    if file_obj:
                        modis_files.append(file_obj)
                        # Track this time stamp for MYD03 matching
                        kept_myd35_times.add(f"A{year}{doy:03d}.{time_str}")
            
            # Download MYD03 files ONLY for granules where MYD35_L2 was kept
            logger.info(f"Downloading MYD03 files for {len(kept_myd35_times)} day-pass granule(s)")
            for granule_id in myd03_granule_ids:
                match = re.search(r'A(\d{4})(\d{3})\.(\d{4})', granule_id)
                if match:
                    year = int(match.group(1))
                    doy = int(match.group(2))
                    time_str = match.group(3)  # HHMM
                    time_id = f"A{year}{doy:03d}.{time_str}"
                    
                    # Only download MYD03 if corresponding MYD35_L2 was kept
                    if time_id in kept_myd35_times:
                        file_obj = self.download_modis_granule(granule_id, 'MYD03', year, doy, 
                                                               target_year=target_date.year,
                                                               target_doy=target_doy)
                        if file_obj:
                            modis_files.append(file_obj)
                    else:
                        logger.debug(f"  Skipping MYD03 {granule_id} (corresponding MYD35_L2 was night pass)")
        
        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info("Download Summary")
        logger.info(f"{'='*70}")
        logger.info(f"OCO-2 files: {len(oco2_files)}")
        logger.info(f"MODIS files: {len(modis_files)}")
        logger.info(f"Total size: {self.download_stats['total_bytes'] / 1e9:.2f} GB")
        logger.info(f"Total time: {self.download_stats['total_time_seconds']:.1f} seconds")
        
        if self.download_stats['failed_downloads']:
            logger.warning(f"\nFailed downloads: {len(self.download_stats['failed_downloads'])}")
            for failure in self.download_stats['failed_downloads']:
                logger.warning(f"  - {failure['granule_id']} ({failure['product_type']})")
        
        # Write status file to indicate successful download
        if oco2_files or modis_files:
            granule_ids = [g.granule_id for g in granules]
            self._write_download_status(target_date, granule_ids, oco2_files, modis_files)
        
        return {
            'granules': granules,
            'oco2_files': oco2_files,
            'modis_files': modis_files,
            'stats': self.download_stats
        }
    
    def get_download_summary(self) -> Dict:
        """Get summary of download statistics."""
        return {
            'total_files': self.download_stats['total_files'],
            'total_size_gb': self.download_stats['total_bytes'] / 1e9,
            'total_time_seconds': self.download_stats['total_time_seconds'],
            'average_speed_mbps': (self.download_stats['total_bytes'] * 8 / 1e6) / 
                                  max(self.download_stats['total_time_seconds'], 1),
            'failed_count': len(self.download_stats['failed_downloads'])
        }


def main():
    """Demo script for Phase 2."""
    from datetime import datetime
    
    # Example usage
    target_date = datetime(2018, 10, 18)
    
    # Initialize ingestion manager
    manager = DataIngestionManager(output_dir="./data")
    
    # Download all products for the date (all modes: GL and ND)
    result = manager.download_all_for_date(
        target_date=target_date,
        limit_granules=1,  # Download only first granule for testing
        include_modis=True
    )
    
    print("\nDownload complete!")
    print(f"OCO-2 files: {len(result['oco2_files'])}")
    print(f"MODIS files: {len(result['modis_files'])}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
