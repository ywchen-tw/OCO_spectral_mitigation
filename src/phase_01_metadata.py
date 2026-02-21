"""
Phase 1: Metadata Acquisition and Temporal Filtering
=====================================================

This module handles the acquisition and parsing of OCO-2 L1B Science metadata
from the GES DISC archive to establish temporal and orbital boundaries.

Key Functions:
- fetch_oco2_xml: Retrieves XML metadata for a specific date
- parse_orbit_info: Extracts orbit number, viewing mode, and version
- extract_temporal_window: Identifies overall start/end timestamps across all matching granules
- extract_granule_temporal_windows: Identifies separate start/end timestamps for each granule
"""

import requests
import xml.etree.ElementTree as ET
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OCO2Granule:
    """Represents an OCO-2 granule with its metadata."""
    granule_id: str
    orbit_str: str # e.g., "22845a"
    viewing_mode: str  # "GL" (Glint) or "ND" (Nadir) or "TG" (Target)
    version: str  # e.g., "B11"
    start_time: datetime
    end_time: datetime
    download_url: str
    spatial_bounds: List[List[Tuple[float, float]]] = None  # List of GPolygons: [[(lat, lon), ...], ...]
    
    def __repr__(self):
        num_polygons = len(self.spatial_bounds) if self.spatial_bounds else 0
        return (f"OCO2Granule(orbit={self.orbit_str}, mode={self.viewing_mode}, "
                f"version={self.version}, polygons={num_polygons}, start={self.start_time.isoformat()})")


class OCO2MetadataRetriever:
    """Handles retrieval and parsing of OCO-2 L1B Science metadata."""
    
    # GES DISC direct access URL structure for OCO-2 L1B Science
    # Format: https://oco2.gesdisc.eosdis.nasa.gov/data/{session}/OCO2_DATA/OCO2_L1B_Science.{VERSION}/{YEAR}/{DOY}/
    GESDISC_BASE_URL = "https://oco2.gesdisc.eosdis.nasa.gov/data"
    GESDISC_COLLECTION_BASE = "OCO2_DATA/OCO2_L1B_Science"
    
    # Version changed from 11r to 11.2r after 2024 DOY=92
    VERSION_CHANGE_DATE = datetime(2024, 1, 1).replace(
        month=1, day=1
    ) + timedelta(days=91)  # 2024-04-01 (DOY=92)
    L1B_VERSION_OLD = "11r"
    L1B_VERSION_NEW = "11.2r"
    
    # CMR search URL as fallback
    CMR_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.xml"
    CMR_COLLECTION_OLD = "OCO2_L1B_Science_11r"
    CMR_COLLECTION_NEW = "OCO2_L1B_Science_11.2r"
    
    def __init__(self, earthdata_username: Optional[str] = None, 
                 earthdata_password: Optional[str] = None):
        """
        Initialize the metadata retriever.
        
        Args:
            earthdata_username: NASA Earthdata username (optional for metadata)
            earthdata_password: NASA Earthdata password (optional for metadata)
        """
        self.username = earthdata_username
        self.password = earthdata_password
        self.session = requests.Session()
        
        # Set up authentication if provided or available in environment
        if self.username is None or self.password is None:
            self.username = os.environ.get('EARTHDATA_USERNAME')
            self.password = os.environ.get('EARTHDATA_PASSWORD')
            if self.username and self.password:
                logger.info("Using credentials from environment variables")
        
        if self.username and self.password:
            self.session.auth = (self.username, self.password)
            logger.info("Authentication initialized")
        else:
            logger.warning("No credentials provided. Some features may be unavailable.")
    
    def _get_collection_version(self, target_date: datetime) -> str:
        """
        Get the appropriate collection version based on the date.
        
        Version changed from 11r to 11.2r on 2024-04-01 (DOY=92).
        
        Args:
            target_date: The date to check
        
        Returns:
            Version string ("11r" or "11.2r")
        """
        if target_date >= self.VERSION_CHANGE_DATE:
            return self.L1B_VERSION_NEW
        else:
            return self.L1B_VERSION_OLD
    
    def _get_cmr_collection(self, target_date: datetime) -> str:
        """
        Get the appropriate CMR collection short name based on the date.
        
        Args:
            target_date: The date to check
        
        Returns:
            CMR collection name
        """
        if target_date >= self.VERSION_CHANGE_DATE:
            return self.CMR_COLLECTION_NEW
        else:
            return self.CMR_COLLECTION_OLD
    
    def fetch_oco2_xml_from_directory(self, target_date: datetime) -> List[str]:
        """
        Fetch OCO-2 L1B Science XML files from GES DISC directory.
        
        Accesses the GES DISC file system directly using the year/day-of-year
        directory structure: /data/{SESSION}/OCO2_DATA/OCO2_L1B_Science.{VERSION}/{YEAR}/{DOY}/
        
        Version: 11r (before 2024-04-01) or 11.2r (after 2024-04-01)
        
        Args:
            target_date: The observation date to query
        
        Returns:
            List of XML file contents
        """
        # Get the appropriate version for this date
        version = self._get_collection_version(target_date)
        
        # Calculate year and day-of-year
        year = target_date.year
        day_of_year = target_date.timetuple().tm_yday
        doy_str = str(day_of_year).zfill(3)  # Zero-padded, e.g., "008"
        
        # Build directory URL (without session ID, relying on direct HTTP access)
        collection_path = f"{self.GESDISC_COLLECTION_BASE}.{version}"
        directory_url = f"{self.GESDISC_BASE_URL}/{collection_path}/{year}/{doy_str}/"
        
        logger.info(f"Querying GES DISC directory: {directory_url}")
        logger.info(f"Target date: {target_date.date()} (DOY: {doy_str}, Version: {version})")
        
        xml_files = []
        
        try:
            # Try to list directory contents
            response = self.session.get(directory_url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML directory listing to find .xml files
            xml_pattern = r'href=["\']?([^"\'>\s]*\.xml)["\']?'
            matches = re.findall(xml_pattern, response.text, re.IGNORECASE)
            # Remove duplicates and ensure full URLs with sorrting
            matches = list(sorted(set(matches)))
            
            for xml_filename in matches:
                try:
                    xml_url = directory_url.rstrip('/') + '/' + xml_filename.strip('/')
                    logger.debug(f"Fetching: {xml_url}")
                    
                    xml_response = self.session.get(xml_url, timeout=10)
                    xml_response.raise_for_status()
                    
                    xml_files.append(xml_response.text)
                    logger.debug(f"Retrieved {xml_filename} ({len(xml_response.text)} bytes)")
                    
                except requests.RequestException as e:
                    logger.warning(f"Failed to fetch {xml_filename}: {e}")
                    continue
            
            if xml_files:
                logger.info(f"Successfully retrieved {len(xml_files)} XML files")
            else:
                logger.warning("No XML files retrieved from directory")
            
            return xml_files
            
        except requests.RequestException as e:
            logger.error(f"Failed to access GES DISC directory: {e}")
            logger.info("Falling back to CMR API query...")
            xml_content = self.fetch_oco2_xml_from_cmr(target_date)
            return [xml_content] if xml_content else []
    
    def fetch_oco2_xml_from_cmr(self, target_date: datetime) -> Optional[str]:
        """
        Fetch OCO-2 L1B Science XML metadata from CMR API (fallback method).
        
        Args:
            target_date: The observation date to query
        
        Returns:
            XML string containing granule metadata, or None if failed
        """
        # CMR requires short_name and version as separate parameters.
        # The collection short name is always "OCO2_L1B_Science"; the version
        # ("11r" or "11.2r") selects the correct dataset vintage.
        version = self._get_collection_version(target_date)

        # Set up temporal bounds (full day)
        start_time = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)

        # Build CMR search parameters
        params = {
            'short_name': 'OCO2_L1B_Science',
            'version': version,
            'temporal': f"{start_time.isoformat()}Z,{end_time.isoformat()}Z",
            'page_size': 100,
            'sort_key': '-start_date'
        }

        logger.info(f"Querying CMR for OCO-2 L1B data on {target_date.date()} "
                    f"(short_name=OCO2_L1B_Science, version={version})")
        
        try:
            response = self.session.get(self.CMR_SEARCH_URL, params=params, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Successfully retrieved CMR metadata ({len(response.text)} bytes)")
            return response.text
            
        except requests.RequestException as e:
            logger.error(f"Failed to retrieve metadata from CMR: {e}")
            return None
    
    def fetch_oco2_xml(self, target_date: datetime, 
                       orbit_str: Optional[str] = None,
                       viewing_mode: Optional[str] = None) -> List[str]:
        """
        Fetch OCO-2 L1B Science XML metadata for a specific date.
        
        Attempts to fetch from GES DISC directory first, falls back to CMR API.
        
        Args:
            target_date: The observation date to query
            orbit_str: Optional specific orbit string to filter
            viewing_mode: Optional viewing mode filter ("GL" or "ND")
        
        Returns:
            List of XML strings containing granule metadata
        """
        # Try GES DISC directory first
        xml_files = self.fetch_oco2_xml_from_directory(target_date)
        
        if not xml_files:
            # Fallback to CMR
            xml_content = self.fetch_oco2_xml_from_cmr(target_date)
            if xml_content:
                xml_files = [xml_content]
        
        return xml_files
    
    def parse_orbit_info(self, xml_contents: List[str]) -> List[OCO2Granule]:
        """
        Parse OCO-2 XML metadata to extract orbit information.
        
        Handles two formats:
        1. S4PA Granule Metadata Files (from GES DISC direct access)
        2. ATOM feed entries (from CMR API)
        
        Args:
            xml_contents: List of XML strings from GES DISC/CMR
        
        Returns:
            List of OCO2Granule objects with parsed metadata
        """
        granules = []
        filtered_granules = []  # Track filtered granules with reasons
        
        # Handle single XML string for backward compatibility
        if isinstance(xml_contents, str):
            xml_contents = [xml_contents]
        
        logger.info(f"Parsing {len(xml_contents)} XML file(s)")
        
        # Define namespaces for XML parsing (for CMR feeds)
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'gml': 'http://www.opengis.net/gml',
            'echo': 'http://www.echo.nasa.gov/esip'
        }
        
        # Process each XML file
        for xml_content in xml_contents:
            if not xml_content or not xml_content.strip():
                logger.warning("Skipping empty XML content")
                continue
            
            try:
                root = ET.fromstring(xml_content)
                
                # Detect format: Check if it's S4PA (individual granule metadata) or ATOM feed
                logger.debug(f"Parsed XML with root tag: {root.tag}")
                
                if root.tag == 'S4PAGranuleMetaDataFile':
                    # Format 1: GES DISC individual granule metadata file
                    logger.debug("Detected S4PA Granule Metadata Format")
                    granule = self._parse_s4pa_granule(root, xml_content)
                    if granule:
                        granules.append(granule)
                        logger.debug(f"Parsed: {granule}")
                    else:
                        # Track filtered granule
                        granule_id_elem = root.find('.//GranuleID')
                        granule_id = granule_id_elem.text if granule_id_elem is not None else "Unknown"
                        filtered_granules.append(('S4PA', granule_id))
                
                else:
                    # Format 2: CMR ATOM feed
                    logger.debug("Detected ATOM Feed Format (CMR)")
                    entries = root.findall('.//atom:entry', namespaces)
                    logger.info(f"Found {len(entries)} granule entries in XML file")
                    
                    for entry in entries:
                        try:
                            granule = self._parse_atom_entry(entry, namespaces)
                            if granule:
                                granules.append(granule)
                                logger.debug(f"Parsed: {granule}")
                        except Exception as e:
                            logger.warning(f"Failed to parse ATOM entry: {e}")
                            continue
                
            except ET.ParseError as e:
                logger.error(f"XML parsing error: {e}")
                continue
        
        logger.info(f"Successfully parsed {len(granules)} granules from all files")
        
        # Log filtered granules summary
        if filtered_granules:
            logger.warning(f"Filtered {len(filtered_granules)} granule(s) due to incomplete data:")
            for fmt, granule_id in filtered_granules:
                logger.warning(f"  - {granule_id} ({fmt})")
        
        return granules
    
    def _parse_s4pa_granule(self, root: ET.Element, xml_content: str) -> Optional[OCO2Granule]:
        """
        Parse S4PA Granule Metadata File (GES DISC individual granule format).
        
        Args:
            root: The root XML element (S4PAGranuleMetaDataFile)
            xml_content: The full XML content (for fallback granule ID extraction)
        
        Returns:
            OCO2Granule object or None if parsing failed
        """
        try:
            # Extract granule ID
            granule_id_elem = root.find('.//GranuleID')
            granule_id = granule_id_elem.text if granule_id_elem is not None else "Unknown"
            
            # Parse granule ID to extract orbit and mode
            orbit_str, view_mode, version = self._parse_granule_id(granule_id)
            
            logger.debug(f"Parsing S4PA granule: {granule_id}, orbit={orbit_str}, mode={view_mode}, version={version}")
            
            # Extract temporal bounds
            begin_date_elem = root.find('.//RangeDateTime/RangeBeginningDate')
            begin_time_elem = root.find('.//RangeDateTime/RangeBeginningTime')
            end_date_elem = root.find('.//RangeDateTime/RangeEndingDate')
            end_time_elem = root.find('.//RangeDateTime/RangeEndingTime')
            
            start_dt = None
            end_dt = None
            
            if begin_date_elem is not None and begin_time_elem is not None and end_date_elem is not None and end_time_elem is not None:
                try:
                    begin_date_str = begin_date_elem.text
                    begin_time_str = begin_time_elem.text
                    end_date_str = end_date_elem.text
                    end_time_str = end_time_elem.text
                    
                    # Construct ISO format datetime strings and parse
                    # Handle time strings that may end with 'Z' or be in plain format
                    start_iso = f"{begin_date_str}T{begin_time_str}".replace('Z', '+00:00')
                    end_iso = f"{end_date_str}T{end_time_str}".replace('Z', '+00:00')
                    
                    start_dt = datetime.fromisoformat(start_iso)
                    end_dt = datetime.fromisoformat(end_iso)
                    logger.debug(f"  Parsed temporal: {start_dt} to {end_dt}")
                except ValueError as e:
                    logger.warning(f"Failed to parse temporal data for {granule_id}: {e}")
                    start_dt = None
                    end_dt = None
            else:
                logger.debug(f"Missing temporal elements for {granule_id}: begin_date={begin_date_elem is not None}, begin_time={begin_time_elem is not None}, end_date={end_date_elem is not None}, end_time={end_time_elem is not None}")
            
            # For GES DISC direct files, construct download URL
            # Pattern: https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L1B_Science.{VERSION}/{YEAR}/{DOY}/{FILENAME}
            if start_dt and granule_id != "Unknown":
                year = start_dt.year
                doy = str(start_dt.timetuple().tm_yday).zfill(3)
                # Use only the date part (naive) for version comparison  
                target_date_naive = start_dt.replace(tzinfo=None) if start_dt.tzinfo is not None else start_dt
                version_str = self._get_collection_version(target_date_naive)
                download_url = f"https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L1B_Science.{version_str}/{year}/{doy}/{granule_id}"
            else:
                download_url = ""
            
            # Check all required fields
            missing_fields = []
            if not orbit_str:
                missing_fields.append("orbit_str")
            if not view_mode:
                missing_fields.append("viewing_mode")
            if not version:
                missing_fields.append("version")
            if not start_dt:
                missing_fields.append("start_time")
            if not end_dt:
                missing_fields.append("end_time")
            
            if missing_fields:
                logger.debug(f"Incomplete granule data for {granule_id}: missing {missing_fields}")
                return None
            
            # Extract spatial bounds (GPolygons) for MODIS matching
            spatial_bounds = self._extract_gpolygons(root)
            
            granule = OCO2Granule(
                granule_id=granule_id,
                orbit_str=orbit_str,
                viewing_mode=view_mode,
                version=version,
                start_time=start_dt,
                end_time=end_dt,
                download_url=download_url,
                spatial_bounds=spatial_bounds
            )
            return granule
                
        except Exception as e:
            logger.warning(f"Failed to parse S4PA granule: {e}")
            return None
    
    def _extract_gpolygons(self, root: ET.Element) -> Optional[List[List[Tuple[float, float]]]]:
        """
        Extract GPolygon spatial boundaries from S4PA granule metadata.
        
        GPolygons are stored in SpatialDomainContainer/HorizontalSpatialDomainContainer/GPolygon
        Each GPolygon contains a Boundary with multiple Point elements containing 
        PointLongitude and PointLatitude.
        
        Args:
            root: The root XML element (S4PAGranuleMetaDataFile)
        
        Returns:
            List of polygons, where each polygon is a list of (latitude, longitude) tuples,
            or None if no GPolygons found.
        """
        try:
            gpolygons = root.findall('.//SpatialDomainContainer/HorizontalSpatialDomainContainer/GPolygon')
            
            if not gpolygons:
                logger.debug("No GPolygons found in spatial domain")
                return None
            
            spatial_bounds = []
            
            for gpolyg_idx, gpoly in enumerate(gpolygons):
                boundary = gpoly.find('Boundary')
                if boundary is None:
                    logger.debug(f"GPolygon {gpolyg_idx} has no Boundary element, skipping")
                    continue
                
                points = boundary.findall('Point')
                if not points:
                    logger.debug(f"GPolygon {gpolyg_idx} Boundary has no Points, skipping")
                    continue
                
                polygon_coords = []
                for point in points:
                    lon_elem = point.find('PointLongitude')
                    lat_elem = point.find('PointLatitude')
                    
                    if lon_elem is not None and lat_elem is not None:
                        try:
                            lon = float(lon_elem.text)
                            lat = float(lat_elem.text)
                            polygon_coords.append((lat, lon))
                        except ValueError as e:
                            logger.warning(f"Failed to parse point coordinates: {e}")
                            continue
                
                if polygon_coords:
                    spatial_bounds.append(polygon_coords)
                    logger.debug(f"Extracted GPolygon {gpolyg_idx} with {len(polygon_coords)} points")
            
            return spatial_bounds if spatial_bounds else None
            
        except Exception as e:
            logger.warning(f"Failed to extract GPolygons: {e}")
            return None
    
    def _parse_atom_entry(self, entry: ET.Element, namespaces: Dict) -> Optional[OCO2Granule]:
        """
        Parse an ATOM feed entry (CMR format).
        
        Args:
            entry: The ATOM entry element
            namespaces: XML namespace mappings
        
        Returns:
            OCO2Granule object or None if parsing failed
        """
        try:
            # Extract granule ID
            title = entry.find('atom:title', namespaces)
            granule_id = title.text if title is not None else "Unknown"
            
            # Parse granule ID to extract orbit and mode
            orbit_str, view_mode, version = self._parse_granule_id(granule_id)
            
            # Extract temporal bounds
            time_start = entry.find('.//echo:temporal/echo:RangeDateTime/echo:BeginningDateTime', 
                                   namespaces)
            time_end = entry.find('.//echo:temporal/echo:RangeDateTime/echo:EndingDateTime', 
                                 namespaces)
            
            start_dt = datetime.fromisoformat(time_start.text.replace('Z', '+00:00')) if time_start is not None else None
            end_dt = datetime.fromisoformat(time_end.text.replace('Z', '+00:00')) if time_end is not None else None
            
            # Extract download URL
            links = entry.findall('atom:link', namespaces)
            download_url = None
            for link in links:
                if link.get('rel') == 'http://esipfed.org/ns/fedsearch/1.1/data#':
                    download_url = link.get('href')
                    break
            
            if all([orbit_str, view_mode, version, start_dt, end_dt]):
                granule = OCO2Granule(
                    granule_id=granule_id,
                    orbit_str=orbit_str,
                    viewing_mode=view_mode,
                    version=version,
                    start_time=start_dt,
                    end_time=end_dt,
                    download_url=download_url or "",
                    spatial_bounds=None  # CMR ATOM entries don't include GPolygon data
                )
                return granule
            else:
                logger.warning(f"Incomplete granule data for {granule_id}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to parse ATOM entry: {e}")
            return None
    
    def _parse_granule_id(self, granule_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse granule ID to extract orbit number, viewing mode, and version.
        
        Format examples:
        - oco2_L1bScGL_22845a_181018_B11006r_220921185957.h5 (Glint)
        - oco2_L1bScND_12345a_181018_B11006r_220921185957.h5 (Nadir)
        
        Structure: oco2_L1bSc[GL|ND]_[ORBITNUMBER][a-z]_[DATE]_[VERSION]_[TIMESTAMP].h5
        
        Args:
            granule_id: The granule filename
        
        Returns:
            Tuple of (orbit_str, viewing_mode, version)
        """
        try:
            parts = granule_id.split('_')
            
            # Extract viewing mode (GL or ND from L1bScGL or L1bScND)
            if 'GL' in parts[1]:
                view_mode = 'GL'
            elif 'ND' in parts[1]:
                view_mode = 'ND'
            elif 'TG' in parts[1]:  # Target mode (if applicable)
                view_mode = 'TG'
            else:
                view_mode = None
            
            # Extract orbit number (remove trailing 'a' or other letters)
            orbit_str = parts[2]
            
            # Extract version (e.g., B11006r) - it's in parts[4], after the date
            # parts[3] is the date (YYMMDD), parts[4] is the version string
            version = parts[4] if len(parts) > 4 else None
            
            return orbit_str, view_mode, version
            
        except (IndexError, ValueError) as e:
            logger.warning(f"Could not parse granule ID '{granule_id}': {e}")
            return None, None, None
    
    def extract_temporal_window(self, granules: List[OCO2Granule], 
                                orbit_str: Optional[str] = None,
                                viewing_mode: Optional[str] = None) -> Tuple[datetime, datetime]:
        """
        Extract the temporal window for specified orbit/mode.
        
        Args:
            granules: List of OCO2Granule objects
            orbit_str: Optional specific orbit to filter
            viewing_mode: Optional viewing mode to filter ("GL" or "ND")
        
        Returns:
            Tuple of (start_time, end_time) for the specified orbit
        
        Raises:
            ValueError: If no matching granules found
        """
        # Filter granules
        filtered = granules
        
        if orbit_str is not None:
            filtered = [g for g in filtered if g.orbit_str == orbit_str]
        
        if viewing_mode is not None:
            filtered = [g for g in filtered if g.viewing_mode == viewing_mode]
        
        if not filtered:
            raise ValueError(f"No granules found for orbit={orbit_str}, mode={viewing_mode}")
        
        # Get temporal bounds across all matching granules
        start_time = min(g.start_time for g in filtered)
        end_time = max(g.end_time for g in filtered)
        
        logger.info(f"Temporal window: {start_time.isoformat()} to {end_time.isoformat()}")
        logger.info(f"Duration: {(end_time - start_time).total_seconds() / 60:.1f} minutes")
        
        return start_time, end_time
    
    def extract_granule_temporal_windows(self, granules: List[OCO2Granule],
                                         orbit_str: Optional[str] = None,
                                         viewing_mode: Optional[str] = None) -> List[Dict]:
        """
        Extract separate start and end times for each granule.
        
        Args:
            granules: List of OCO2Granule objects
            orbit_str: Optional specific orbit to filter
            viewing_mode: Optional viewing mode to filter ("GL" or "ND")
        
        Returns:
            List of dictionaries with per-granule temporal info:
            [
                {
                    'granule_id': str,
                    'orbit_str': str,
                    'viewing_mode': str,
                    'start_time': datetime,
                    'end_time': datetime,
                    'duration_minutes': float
                },
                ...
            ]
        
        Raises:
            ValueError: If no matching granules found
        """
        # Filter granules
        filtered = granules
        
        if orbit_str is not None:
            filtered = [g for g in filtered if g.orbit_str == orbit_str]
        
        if viewing_mode is not None:
            filtered = [g for g in filtered if g.viewing_mode == viewing_mode]
        
        if not filtered:
            raise ValueError(f"No granules found for orbit={orbit_str}, mode={viewing_mode}")
        
        # Extract per-granule temporal windows
        windows = []
        for granule in filtered:
            duration = (granule.end_time - granule.start_time).total_seconds() / 60
            windows.append({
                'granule_id': granule.granule_id,
                'orbit_str': granule.orbit_str,
                'viewing_mode': granule.viewing_mode,
                'start_time': granule.start_time,
                'end_time': granule.end_time,
                'duration_minutes': duration
            })
        
        logger.info(f"Extracted temporal windows for {len(windows)} granule(s)")
        for w in windows:
            logger.info(f"  {w['granule_id']}: {w['start_time'].isoformat()} to {w['end_time'].isoformat()} ({w['duration_minutes']:.1f} min)")
        
        return windows
    
    def get_metadata_summary(self, target_date: datetime, 
                            orbit_str: Optional[str] = None,
                            viewing_mode: Optional[str] = None) -> Dict:
        """
        High-level method to fetch and summarize OCO-2 metadata for a date.
        
        Args:
            target_date: The observation date
            orbit_str: Optional orbit string filter
            viewing_mode: Optional viewing mode filter
        
        Returns:
            Dictionary containing granule information and temporal window
        """
        # Fetch XML files
        xml_contents = self.fetch_oco2_xml(target_date, orbit_str, viewing_mode)
        
        if not xml_contents:
            logger.warning("No XML content retrieved")
            return {
                'granules': [],
                'temporal_window': None,
                'orbit_count': 0,
                'xml_sources': 0,
                'xml_retrieved': False,
                'date': target_date.date()
            }
        
        # Parse granules
        granules = self.parse_orbit_info(xml_contents)
        
        if not granules:
            logger.warning("No granules found for the specified criteria")
            return {
                'granules': [],
                'temporal_window': None,
                'orbit_count': 0,
                'xml_sources': len(xml_contents),
                'xml_retrieved': True,
                'date': target_date.date()
            }
        
        # Extract temporal window
        try:
            start_time, end_time = self.extract_temporal_window(
                granules, orbit_str, viewing_mode
            )
            temporal_window = (start_time, end_time)
        except ValueError:
            temporal_window = None
        
        # Group granules by orbit and mode
        orbit_summary = {}
        for g in granules:
            key = (g.orbit_str, g.viewing_mode)
            if key not in orbit_summary:
                orbit_summary[key] = []
            orbit_summary[key].append(g)
        
        return {
            'granules': granules,
            'temporal_window': temporal_window,
            'orbit_count': len(orbit_summary),
            'orbit_summary': orbit_summary,
            'date': target_date.date(),
            'xml_sources': len(xml_contents),
            'xml_retrieved': True
        }


def main_example():
    """Example usage of Phase 1 metadata retrieval."""
    
    # Initialize retriever
    retriever = OCO2MetadataRetriever()
    
    # Example: Query for a specific date
    target_date = datetime(2018, 10, 18)
    
    logger.info(f"=== Phase 1: Metadata Acquisition ===")
    logger.info(f"Target date: {target_date.date()}")
    
    # Get metadata summary
    summary = retriever.get_metadata_summary(target_date)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"OCO-2 Metadata Summary for {summary['date']}")
    print(f"{'='*60}")
    print(f"XML files retrieved: {summary['xml_sources']}")
    print(f"Total orbits found: {summary['orbit_count']}")
    
    if summary['granules']:
        print(f"\nGranule Details:")
        for i, granule in enumerate(summary['granules'][:5], 1):  # Show first 5
            print(f"\n{i}. {granule.granule_id}")
            print(f"   Orbit: {granule.orbit_str}")
            print(f"   Mode: {granule.viewing_mode}")
            print(f"   Version: {granule.version}")
            print(f"   Time: {granule.start_time.strftime('%Y-%m-%d %H:%M:%S')} - "
                  f"{granule.end_time.strftime('%H:%M:%S')}")
            
            # Print GPolygon spatial bounds information
            if granule.spatial_bounds:
                print(f"   Spatial Bounds: {len(granule.spatial_bounds)} polygon(s)")
                
                # Show first polygon as example
                if granule.spatial_bounds:
                    first_polygon = granule.spatial_bounds[0]
                    print(f"     Polygon 0 corner points:")
                    for point_idx, (lat, lon) in enumerate(first_polygon, 1):
                        print(f"       Point {point_idx}: lat={lat:8.3f}, lon={lon:8.3f}")
                    
                    # Show remaining polygons count and lat/lon ranges
                    if len(granule.spatial_bounds) > 1:
                        print(f"     ... and {len(granule.spatial_bounds) - 1} more polygon(s)")
                    
                    # Calculate overall latitude and longitude ranges across all polygons
                    all_lats = []
                    all_lons = []
                    for polygon in granule.spatial_bounds:
                        for lat, lon in polygon:
                            all_lats.append(lat)
                            all_lons.append(lon)
                    
                    if all_lats and all_lons:
                        lat_min, lat_max = min(all_lats), max(all_lats)
                        lon_min, lon_max = min(all_lons), max(all_lons)
                        print(f"     Overall coverage - Lat: [{lat_min:8.3f}, {lat_max:8.3f}], Lon: [{lon_min:8.3f}, {lon_max:8.3f}]")
            else:
                print(f"   Spatial Bounds: None")
        
        if len(summary['granules']) > 5:
            print(f"\n   ... and {len(summary['granules']) - 5} more granules")
    
    if summary['temporal_window']:
        start, end = summary['temporal_window']
        duration = (end - start).total_seconds() / 60
        print(f"\nTemporal Window (±20 min for MODIS matching):")
        print(f"  Start: {start.isoformat()}")
        print(f"  End:   {end.isoformat()}")
        print(f"  Duration: {duration:.1f} minutes")
        
        # Add buffer for MODIS search
        modis_start = start - timedelta(minutes=20)
        modis_end = end + timedelta(minutes=20)
        print(f"\nMODIS Search Window:")
        print(f"  Start: {modis_start.isoformat()}")
        print(f"  End:   {modis_end.isoformat()}")


if __name__ == "__main__":
    main_example()
