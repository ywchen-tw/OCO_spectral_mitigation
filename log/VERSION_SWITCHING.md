# OCO-2 L1B Version Switching Documentation

## Overview
The Phase 1 metadata retrieval system (`phase_01_metadata.py`) now implements **dynamic version switching** for OCO-2 L1B Science data based on the acquisition date.

## Version Change Timeline
- **Before 2024-04-01 (DOY < 92)**: `OCO2_L1B_Science.11r`
- **On or After 2024-04-01 (DOY ≥ 92)**: `OCO2_L1B_Science.11.2r`

The version change date is set as: `VERSION_CHANGE_DATE = datetime(2024, 4, 1)` (2024-01-01 + 91 days)

## Implementation Details

### Class Constants
```python
VERSION_CHANGE_DATE = datetime(2024, 4, 1)  # 2024-04-01
L1B_VERSION_OLD = "11r"
L1B_VERSION_NEW = "11.2r"
CMR_COLLECTION_OLD = "OCO2_L1B_Science_11r"
CMR_COLLECTION_NEW = "OCO2_L1B_Science_11.2r"
```

### Helper Methods

#### `_get_collection_version(target_date: datetime) → str`
Returns the appropriate version string based on the target date.

**Example Usage:**
```python
retriever = OCO2MetadataRetriever(username, password)
version = retriever._get_collection_version(datetime(2024, 5, 1))  # Returns "11.2r"
version = retriever._get_collection_version(datetime(2024, 3, 1))  # Returns "11r"
```

#### `_get_cmr_collection(target_date: datetime) → str`
Returns the appropriate CMR collection name based on the target date.

**Example Usage:**
```python
collection = retriever._get_cmr_collection(datetime(2024, 5, 1))  # Returns "OCO2_L1B_Science_11.2r"
collection = retriever._get_cmr_collection(datetime(2024, 3, 1))  # Returns "OCO2_L1B_Science_11r"
```

### Data Source Integration

#### GES DISC Directory URL Construction
The `fetch_oco2_xml_from_directory()` method dynamically constructs URLs with the correct version:

```
Before 2024-04-01:
  https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L1B_Science.11r/2018/291/*xml

On/After 2024-04-01:
  https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L1B_Science.11.2r/2024/292/*xml
```

#### CMR API Fallback
The `fetch_oco2_xml_from_cmr()` method queries the correct CMR collection:

- Before 2024-04-01: Searches `OCO2_L1B_Science_11r`
- On/After 2024-04-01: Searches `OCO2_L1B_Science_11.2r`

## Date Boundary Behavior

### Key Dates
| Date | DOY | Version | Reason |
|------|-----|---------|--------|
| 2024-03-31 | 91 | 11r | Before change date |
| 2024-04-01 | 92 | 11.2r | On change date (inclusive) |
| 2024-04-02 | 93 | 11.2r | After change date |
| 2025-01-01 | 1 | 11.2r | Future dates use new version |
| 2018-10-18 | 291 | 11r | Historical data uses old version |

### Version Transition Algorithm
```python
if target_date >= VERSION_CHANGE_DATE:  # On or after 2024-04-01
    return L1B_VERSION_NEW  # "11.2r"
else:
    return L1B_VERSION_OLD  # "11r"
```

## Testing

### Test Coverage
Run `workspace/test_version_switching.py` to verify:

1. **Version Detection**: Validates correct version string returned for dates before/on/after change
2. **CMR Collection Naming**: Verifies CMR collection names match version
3. **Directory URL Construction**: Confirms version is correctly embedded in GES DISC directory paths
4. **Boundary Conditions**: Tests edge cases (day before, day of, day after)
5. **Future Dates**: Confirms all future dates use new version

### Test Output Example
```
Before (2024-03-31):
  Version: 11r
  CMR Collection: OCO2_L1B_Science_11r

On Change (2024-04-01 / DOY=92):
  Version: 11.2r
  CMR Collection: OCO2_L1B_Science_11.2r

Directory URL Construction:
  2018-10-18 (DOY=291):
    URL: https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L1B_Science.11r/2018/291/
  
  2024-10-18 (DOY=292):
    URL: https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L1B_Science.11.2r/2024/292/

✓ All version switching tests passed!
```

## Usage

### Seamless Integration
For end users, version switching is **automatic and transparent**. No special configuration needed:

```python
from src.phase_01_metadata import OCO2MetadataRetriever

retriever = OCO2MetadataRetriever(username="user", password="pass")

# Works for any date - version is handled automatically
metadata = retriever.get_metadata_summary(
    target_date=datetime(2024, 5, 15),
    orbit_number=3456,
    viewing_mode="GL"
)

# Internally uses version 11.2r and OCO2_L1B_Science_11.2r
print(metadata['xml_sources'])  # Shows which sources were used
print(metadata['granules'])     # Contains granules with correct version
```

### Manual Version Checking
If you need to know which version applies to a specific date:

```python
target_date = datetime(2024, 5, 15)
version = retriever._get_collection_version(target_date)
print(f"Data from {target_date.strftime('%Y-%m-%d')} uses version: {version}")
# Output: Data from 2024-05-15 uses version: 11.2r
```

## Migration Notes

### For Existing Code
Version switching is **backward compatible**. Existing calls to Phase 1 methods will:
- Automatically use version 11r for dates before 2024-04-01
- Automatically use version 11.2r for dates on/after 2024-04-01
- No code changes required

### For Phase 2+ Implementation
Downstream phases (ingestion, processing, geometry, synthesis) receive granule data with:
- `version` field already populated with correct version ("11r" or "11.2r")
- `download_url` pointing to correct GES DISC location
- No special handling needed - version information is already included

## Performance Impact
- **Zero overhead**: Version lookup is O(1) datetime comparison
- **No additional API calls**: Version determination happens before data retrieval
- **Cache friendly**: Version remained stable for years (11r → 11.2r once in 2024)

## Future Version Changes
To accommodate future version changes:

1. Add new version constants:
   ```python
   VERSION_CHANGE_DATE_2 = datetime(YYYY, M, D)
   L1B_VERSION_FUTURE = "11.3r"
   CMR_COLLECTION_FUTURE = "OCO2_L1B_Science_11.3r"
   ```

2. Update version detection logic:
   ```python
   def _get_collection_version(self, target_date):
       if target_date >= VERSION_CHANGE_DATE_2:
           return self.L1B_VERSION_FUTURE
       elif target_date >= self.VERSION_CHANGE_DATE:
           return self.L1B_VERSION_NEW
       else:
           return self.L1B_VERSION_OLD
   ```

3. Add corresponding CMR collection logic

4. Update tests to cover new boundary

## References
- **GES DISC URL Format**: `https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/OCO2_L1B_Science.{VERSION}/{YEAR}/{DOY}/`
- **CMR Collections**: 
  - Old: `OCO2_L1B_Science_11r`
  - New: `OCO2_L1B_Science_11.2r`
- **Change Date**: 2024-04-01 (Day of Year 92 in 2024)
- **Implementation File**: `src/phase_01_metadata.py`
- **Test File**: `workspace/test_version_switching.py`
