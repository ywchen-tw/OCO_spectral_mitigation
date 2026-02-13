# Phase 1 Quick Reference

## Quick Start

```python
from src.phase_01_metadata import OCO2MetadataRetriever
from datetime import datetime

# Initialize
retriever = OCO2MetadataRetriever()

# Get metadata for a date
summary = retriever.get_metadata_summary(datetime(2023, 7, 15))

# Access results
for granule in summary['granules']:
    print(f"{granule.orbit_number} {granule.viewing_mode}")
```

## Key Classes

### OCO2Granule
```python
@dataclass
class OCO2Granule:
    granule_id: str          # Filename
    orbit_number: int        # Orbit number
    viewing_mode: str        # "GL" or "ND"
    version: str             # e.g., "B11"
    start_time: datetime     # Start of observation
    end_time: datetime       # End of observation
    download_url: str        # Download link
```

### OCO2MetadataRetriever
```python
retriever = OCO2MetadataRetriever(
    earthdata_username=None,  # Optional
    earthdata_password=None   # Optional
)

# Fetch XML from GES DISC
xml = retriever.fetch_oco2_xml(
    target_date=datetime(2023, 7, 15),
    orbit_number=None,   # Optional filter
    viewing_mode=None    # Optional: "GL" or "ND"
)

# Parse granules
granules = retriever.parse_orbit_info(xml)

# Extract temporal window
start, end = retriever.extract_temporal_window(
    granules,
    orbit_number=None,   # Optional filter
    viewing_mode=None    # Optional filter
)

# One-step summary
summary = retriever.get_metadata_summary(
    target_date=datetime(2023, 7, 15),
    orbit_number=None,
    viewing_mode=None
)
```

## Common Patterns

### Filter by orbit and mode
```python
summary = retriever.get_metadata_summary(
    target_date=datetime(2023, 7, 15),
    orbit_number=12345,
    viewing_mode="GL"
)
```

### Get MODIS search window
```python
from datetime import timedelta
from src.config import Config

start, end = retriever.extract_temporal_window(granules)

# Add ±20 minute buffer for MODIS
modis_start = start - timedelta(minutes=Config.MODIS_TEMPORAL_BUFFER_MINUTES)
modis_end = end + timedelta(minutes=Config.MODIS_TEMPORAL_BUFFER_MINUTES)
```

### Group granules by orbit
```python
orbit_groups = {}
for g in summary['granules']:
    key = (g.orbit_number, g.viewing_mode)
    if key not in orbit_groups:
        orbit_groups[key] = []
    orbit_groups[key].append(g)
```

## Expected Data Formats

### CMR XML Structure
```xml
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:echo="http://www.echo.nasa.gov/esip">
  <entry>
    <title>oco2_L1bScGL_12345a_B11_20230715T123456Z.h5</title>
    <echo:temporal>
      <echo:RangeDateTime>
        <echo:BeginningDateTime>2023-07-15T12:30:00Z</echo:BeginningDateTime>
        <echo:EndingDateTime>2023-07-15T13:45:00Z</echo:EndingDateTime>
      </echo:RangeDateTime>
    </echo:temporal>
    <link rel="http://esipfed.org/ns/fedsearch/1.1/data#" 
          href="https://..."/>
  </entry>
</feed>
```

### Granule ID Format
```
oco2_L1bScGL_12345a_B11_20230715T123456Z.h5
      ^^^^^^ ^^^^^^ ^^^
      |      |      └── Version (B11 = build 11)
      |      └── Orbit number (12345)
      └── Viewing mode (GL=Glint, ND=Nadir)
```

## Error Handling

```python
try:
    summary = retriever.get_metadata_summary(date)
except requests.RequestException as e:
    print(f"Network error: {e}")
except ET.ParseError as e:
    print(f"XML parsing error: {e}")
except ValueError as e:
    print(f"No matching granules: {e}")
```

## Configuration

From `src/config.py`:

```python
from src.config import Config

# CMR search URL
Config.CMR_GRANULE_SEARCH_URL

# OCO-2 L1B collection name
Config.OCO2_L1B_SCIENCE.short_name  # "OCO2_L1B_Science_11r"

# Temporal buffer for MODIS
Config.MODIS_TEMPORAL_BUFFER_MINUTES  # 20

# Coordinate transformations
from src.config import CoordinateSystem
X, Y, Z = CoordinateSystem.geodetic_to_ecef(lat, lon, alt)
```

## Utilities

From `src/utils.py`:

```python
from src.utils import (
    setup_logging,
    ensure_directory,
    format_bytes,
    haversine_distance,
    ecef_distance,
    parse_modis_cloud_mask
)

# Set up logging
setup_logging(log_level="INFO", log_file="app.log")

# Create directory
ensure_directory("./output")

# Format file size
print(format_bytes(1024 * 1024))  # "1.0 MB"

# Parse cloud mask
result = parse_modis_cloud_mask(0b11000000)
print(result['classification'])  # 'confident_cloudy'
```

## Testing

Run tests:
```bash
# All tests
pytest tests/test_phase_01.py -v

# Specific test
pytest tests/test_phase_01.py::TestOCO2MetadataRetriever::test_parse_granule_id_glint -v

# With coverage
pytest tests/test_phase_01.py --cov=src.phase_01_metadata
```

## Troubleshooting

### No granules found
- Check date format: `datetime(2023, 7, 15)` not string
- Verify date has OCO-2 observations
- Check orbit/mode filters

### Network errors
- Verify internet connection
- Check GES DISC server status
- For downloads (Phase 2+), will need Earthdata credentials

### XML parsing errors
- Check XML content validity
- Verify namespace handling
- Look for malformed responses

## Next Steps (Phase 2)

Once Phase 1 returns granules and temporal windows:
1. Use download URLs to fetch actual data files
2. Implement authentication for GES DISC
3. Download MODIS data with temporal matching
4. Cache downloaded files locally

See `prompts/Phase_02_Ingestion.md` for Phase 2 requirements.
