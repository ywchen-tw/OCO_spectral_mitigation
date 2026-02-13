````markdown
# Phase 3 Quick Reference

## Overview
Phase 3 extracts OCO-2 footprints and MODIS cloud masks, then matches them temporally for collocation.

## Core Classes

### SpatialProcessor
Main processor for all Phase 3 operations.

```python
from phase_03_processing import SpatialProcessor

processor = SpatialProcessor(data_dir="./data")
```

**Key Methods:**
- `extract_oco2_footprints(files)` → Dict[int, OCO2Footprint]
- `extract_modis_cloud_mask(files)` → Dict[str, List[MODISCloudPixel]]
- `match_temporal_windows(footprints, files)` → Dict[int, List[str]]

### Data Classes

**OCO2Footprint**
```python
@dataclass
class OCO2Footprint:
    sounding_id: int              # Unique identifier
    granule_id: str               # Source granule
    latitude: float               # Degrees
    longitude: float              # Degrees
    sounding_time: datetime       # Observation time
    viewing_mode: str             # 'GL' or 'ND'
```

**MODISCloudPixel**
```python
@dataclass
class MODISCloudPixel:
    pixel_x: int                  # Column in granule
    pixel_y: int                  # Row in granule
    granule_id: str               # Source granule
    latitude: float               # Will be updated in Phase 4
    longitude: float              # Will be updated in Phase 4
    cloud_flag: str               # 'Cloudy' or 'Uncertain'
    observation_time: datetime    # Parsed from filename
```

## Common Workflows

### 1. Extract All Data for a Date
```python
from phase_02_ingestion import DataIngestionManager
from phase_03_processing import SpatialProcessor

# Get Phase 2 files
ingestion = DataIngestionManager()
result = ingestion.download_all_for_date(
    datetime(2018, 10, 18),
    mode_filter='GL'
)

# Process with Phase 3
processor = SpatialProcessor()

oco2 = processor.extract_oco2_footprints(result['oco2_files'])
clouds = processor.extract_modis_cloud_mask(result['modis_files'])
matches = processor.match_temporal_windows(oco2, result['modis_files'])
```

### 2. Extract from Existing Files
```python
# If you already have downloaded files at disk locations
from phase_02_ingestion import DownloadedFile

files = [
    DownloadedFile(
        filepath=Path("./data/OCO2/2018/291/oco2_L2LiteGL_*.h5"),
        product_type="OCO2_L2_Lite",
        granule_id="g1",
        file_size_mb=10.0,
        download_time_seconds=5.0
    )
]

processor = SpatialProcessor()
footprints = processor.extract_oco2_footprints(files)
```

### 3. Match with Custom Buffer
```python
# Use 30-minute buffer instead of default 20 minutes
matches = processor.match_temporal_windows(
    oco2,
    result['modis_files'],
    buffer_minutes=30
)
```

## Key Algorithms

### TAI93 Epoch Conversion
OCO-2 times are stored as seconds since 1993-01-01 00:00:00:

```python
tai93_epoch = datetime(1993, 1, 1, 0, 0, 0)
oco2_time = tai93_epoch + timedelta(seconds=time_value)
```

### Cloud Mask Unpacking
MODIS 48-bit cloud mask (3 bytes per pixel):
- **Byte 1, bits 1-2** indicate cloud state:
  - `00` (0x0) = Cloudy ← **TARGET**
  - `01` (0x2) = Uncertain ← **TARGET**
  - `10` (0x4) = Probably Clear
  - `11` (0x6) = Clear

Extract bits 1-2:
```python
cloud_flags = (byte_value >> 1) & 0b11  # Shift right 1, mask lower 2 bits
```

### Temporal Matching
For each OCO-2 sounding, find MODIS granules within ±20 minute window:

1. Parse OCO-2 sounding time from metadata
2. Parse MODIS granule times from filenames (format: `AYYYYDDD.HHMM`)
3. Calculate time difference in minutes
4. Select granules where |difference| ≤ buffer_minutes
5. Sort by time proximity

## Output Structures

### Extract OCO2 Footprints
Returns: `Dict[sounding_id: int → OCO2Footprint]`

```python
{
    0: OCO2Footprint(sounding_id=0, lat=35.123, lon=-120.456, ...),
    1: OCO2Footprint(sounding_id=1, lat=35.234, lon=-120.567, ...),
    ...
}
```

### Extract MODIS Cloud Mask
Returns: `Dict[granule_id: str → List[MODISCloudPixel]]`

```python
{
    "MYD35_L2.A2018291.1200.061.test.hdf": [
        MODISCloudPixel(pixel_x=100, pixel_y=200, cloud_flag='Cloudy', ...),
        MODISCloudPixel(pixel_x=101, pixel_y=201, cloud_flag='Uncertain', ...),
        ...
    ],
    ...
}
```

### Temporal Matching
Returns: `Dict[sounding_id: int → List[granule_id: str]]`

```python
{
    0: ["MYD35_L2.A2018291.1200.061.test.hdf", "MYD03.A2018291.1200.061.test.hdf"],
    1: ["MYD35_L2.A2018291.1205.061.test.hdf"],
    ...
}
```

## Error Handling

**File Not Found**: Caught and logged; processing continues

**Time Conversion Errors**: Falls back to UTC now

**Missing Data Groups**: Tries common group names, logs warning

**Parsing Errors**: Graceful degradation; uses sensible defaults

## Performance Tips

1. **Batch Processing**: Process multiple dates in sequence
   - Uses same SpatialProcessor instance
   - Reuses module imports

2. **Memory Efficiency**: All data structures are small
   - Dict of OCO2Footprint ≈ 100 bytes each
   - Dict of MODISCloudPixel ≈ 150 bytes each
   - Full day of data ≈ 3-5 MB

3. **Parallelization Ready**: Each file can be processed independently
   - Phase 3 has no persistent state
   - Could use multiprocessing for multiple files

## Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Print Footprint Summary
```python
for sid, fp in list(footprints.items())[:5]:
    print(fp)
```

### Check Temporal Matches
```python
matched = sum(1 for m in matching.values() if len(m) > 0)
print(f"Matched: {matched}/{len(footprints)} soundings")
```

### Inspect Cloud Pixels
```python
for granule_id, pixels in modis_clouds.items():
    cloudy = sum(1 for p in pixels if p.cloud_flag == 'Cloudy')
    uncertain = sum(1 for p in pixels if p.cloud_flag == 'Uncertain')
    print(f"{granule_id}: {cloudy} cloudy, {uncertain} uncertain")
```

## Integration with Phases

**From Phase 2:**
- Downloaded file objects with paths and metadata
- Files organized in `./data/OCO2/` and `./data/MODIS/`

**To Phase 4:**
- OCO2Footprint dictionaries (indexed by sounding_id)
- MODISCloudPixel lists (indexed by granule_id)
- Temporal matching result (sounding_id → MODIS granules)
- All ready for ECEF conversion and KD-Tree construction

## Testing

Run unit tests:
```bash
pytest tests/test_phase_03.py -v
```

Run demo:
```bash
python workspace/demo_phase_03.py --date 2018-10-18 --mode GL
```

## Next: Phase 4 - Geometry

Phase 4 will take these outputs and:
1. Convert coordinates to ECEF
2. Parse MYD03 for 1 km geolocation
3. Build KD-Tree spatial index
4. Calculate nearest cloud distances

````
