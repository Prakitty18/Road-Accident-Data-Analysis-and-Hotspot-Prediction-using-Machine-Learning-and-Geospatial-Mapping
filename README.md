# Road Accident Data Processing Pipeline

This document describes the complete data processing pipeline from the raw dataset (`accident_prediction_india.csv`) to the final processed dataset (`final_dataframe.csv`) used for machine learning model training.

## Overview

The data processing pipeline consists of three main stages:
1. **Geographic Data Enrichment** - Adding latitude and longitude coordinates
2. **Weather Data Integration** - Adding temperature and rainfall data
3. **Traffic Congestion Calculation** - Computing traffic congestion metrics

```
accident_prediction_india.csv
    ↓
processed_accident_data.csv (Step 1: Geographic Data)
    ↓
processed_accident_data_with_weather.csv (Step 2: Weather Data)
    ↓
final_dataframe.csv (Step 3: Traffic Congestion)
```

## Data Processing Steps

### Step 1: Geographic Data Enrichment

**Input:** `data/accident_prediction_india.csv`  
**Output:** `data/processed_accident_data.csv`  
**Script:** `Data-Analysis.ipynb` (Cell 4)

#### Process:
1. **City Name Processing**
   - Replaces "Unknown" city names with actual city names from the corresponding state
   - Uses a predefined mapping of Indian states to major cities
   - Randomly selects a city from the state if the original city is unknown

2. **Coordinate Generation**
   - Adds `Latitude` and `Longitude` columns
   - Generates random coordinates within a 5km radius of the city center
   - Uses approximate center coordinates for each city

#### Key Features:
- **Total Rows:** 3,000 accident records
- **Unique Cities:** 109 cities across India
- **Unique States:** 32 states and union territories
- **New Columns Added:** `Latitude`, `Longitude`

#### Data Quality Improvements:
- ✅ Eliminated "Unknown" city entries
- ✅ Added geospatial coordinates for location-based analysis
- ✅ Maintained data integrity with state-city relationships

---

### Step 2: Weather Data Integration

**Input:** `data/processed_accident_data.csv`  
**Output:** `data/processed_accident_data_with_weather.csv`  
**Script:** `add_weather_to_accidents.py`

#### Process:
1. **Date Construction**
   - Extracts Year and Month from each accident record
   - Constructs a representative date (15th of the month) for weather data retrieval
   - Handles edge cases (e.g., February 30th → uses last day of month)

2. **Weather Data Fetching**
   - Uses Google Earth Engine API with ECMWF/ERA5_LAND/HOURLY dataset
   - Fetches weather data for specific latitude/longitude points
   - Retrieves data for the exact date of each accident

3. **Data Extraction**
   - **Rainfall:** Daily total precipitation in millimeters (mm)
   - **Temperature:** Maximum and minimum temperature in Celsius (°C)
   - Uses point geometry extraction (not regional averages)

#### New Columns Added:
- `Rainfall_mm` - Daily rainfall in millimeters
- `Max_temp_celsius` - Maximum daily temperature in Celsius
- `Min_temp_celsius` - Minimum daily temperature in Celsius

#### Technical Details:
- **API:** Google Earth Engine (ECMWF/ERA5_LAND/HOURLY)
- **Scale:** 11132 meters (~1 degree at equator)
- **Error Handling:** Returns -9999.0 for missing/invalid data
- **Progress Tracking:** Saves progress every 100 rows

#### Data Quality:
- ✅ Point-specific weather data (not regional averages)
- ✅ Historical weather data matched to accident dates
- ✅ Handles missing data gracefully with default values

---

### Step 3: Traffic Congestion Calculation

**Input:** `data/processed_accident_data_with_weather.csv`  
**Output:** `final_dataframe.csv`  
**Script:** `Data-Analysis.ipynb` (Cell 6)

#### Process:
1. **Traffic Congestion Metric**
   - Calculates traffic congestion based on number of vehicles involved
   - Uses non-linear scaling (square root) for gentler scaling
   - Formula: `Traffic Congestion = √(Number of Vehicles) / √(Max Vehicles)`

2. **Normalization**
   - Scales values between 0.447 and 1.0
   - Provides diminishing returns for higher vehicle counts
   - More realistic representation of traffic density

#### New Column Added:
- `Traffic Congestion` - Normalized congestion score (0.447 - 1.0)

#### Formula:
```
Traffic Congestion = √(Number of Vehicles Involved) / √(Max Vehicles)
```

Where:
- Max Vehicles = Maximum number of vehicles in the dataset
- Result is clipped to ensure values between 0 and 1

#### Benefits:
- ✅ Non-linear scaling prevents extreme values
- ✅ Better represents real-world traffic patterns
- ✅ Normalized for machine learning compatibility

---

## Final Dataset Structure

### Columns in `final_dataframe.csv`:

#### Original Columns (23):
- `State Name`, `City Name`, `Year`, `Month`, `Day of Week`, `Time of Day`
- `Accident Severity`, `Number of Vehicles Involved`, `Vehicle Type Involved`
- `Number of Casualties`, `Number of Fatalities`
- `Weather Conditions`, `Road Type`, `Road Condition`, `Lighting Conditions`
- `Traffic Control Presence`, `Speed Limit (km/h)`, `Driver Age`
- `Driver Gender`, `Driver License Status`, `Alcohol Involvement`
- `Accident Location Details`

#### Added Columns (5):
- `Latitude` - Geographic latitude coordinate
- `Longitude` - Geographic longitude coordinate
- `Rainfall_mm` - Daily rainfall in millimeters
- `Max_temp_celsius` - Maximum daily temperature
- `Min_temp_celsius` - Minimum daily temperature
- `Traffic Congestion` - Normalized congestion score

**Total Columns:** 28  
**Total Rows:** 3,000

---

## Data Processing Pipeline Diagram

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA                                 │
│  accident_prediction_india.csv                              │
│  • 23 columns                                               │
│  • 3,000 rows                                               │
│  • Missing: Cities, Coordinates, Weather, Traffic          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ STEP 1: Geographic Enrichment
                        │ Script: Data-Analysis.ipynb (Cell 4)
                        │
                        ├─ Replace "Unknown" cities
                        ├─ Map state → city
                        └─ Generate coordinates
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              GEOGRAPHIC DATA ADDED                          │
│  processed_accident_data.csv                                │
│  • 25 columns (+2: Latitude, Longitude)                     │
│  • 3,000 rows                                                │
│  • All cities named                                         │
│  • All records geocoded                                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ STEP 2: Weather Data Integration
                        │ Script: add_weather_to_accidents.py
                        │ API: Google Earth Engine
                        │
                        ├─ Extract Year, Month
                        ├─ Construct date (15th of month)
                        ├─ Fetch rainfall (mm)
                        ├─ Fetch max temperature (°C)
                        └─ Fetch min temperature (°C)
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              WEATHER DATA ADDED                             │
│  processed_accident_data_with_weather.csv                   │
│  • 28 columns (+3: Rainfall_mm, Max_temp, Min_temp)          │
│  • 3,000 rows                                                │
│  • Weather data for each accident                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ STEP 3: Traffic Congestion
                        │ Script: Data-Analysis.ipynb (Cell 6)
                        │
                        ├─ Calculate from vehicles
                        ├─ Apply square root scaling
                        └─ Normalize (0.447 - 1.0)
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              FINAL PROCESSED DATASET                        │
│  final_dataframe.csv                                         │
│  • 28 columns                                                │
│  • 3,000 rows                                                │
│  • Complete and ready for ML                                │
│  ✅ All features enriched                                   │
└─────────────────────────────────────────────────────────────┘
```

### Column Evolution

```
Original Dataset (23 cols)
├─ State Name
├─ City Name (some "Unknown")
├─ Year, Month, Day of Week, Time of Day
├─ Accident details (Severity, Vehicles, Casualties, etc.)
├─ Weather Conditions (categorical only)
├─ Road conditions
├─ Driver information
└─ Location details

         │
         │ +2 columns (Step 1)
         ▼
Geographic Data (25 cols)
├─ [All original columns]
├─ Latitude ✨ NEW
└─ Longitude ✨ NEW

         │
         │ +3 columns (Step 2)
         ▼
Weather Data (28 cols)
├─ [All previous columns]
├─ Rainfall_mm ✨ NEW
├─ Max_temp_celsius ✨ NEW
└─ Min_temp_celsius ✨ NEW

         │
         │ +1 column (Step 3)
         ▼
Final Dataset (28 cols)
├─ [All previous columns]
└─ Traffic Congestion ✨ NEW (replaces calculation)
```

### Data Transformation Summary

| Stage | Input File | Output File | Columns Added | Key Operations |
|-------|-----------|------------|---------------|----------------|
| **Step 1** | `accident_prediction_india.csv` | `processed_accident_data.csv` | +2 | City mapping, Coordinate generation |
| **Step 2** | `processed_accident_data.csv` | `processed_accident_data_with_weather.csv` | +3 | Weather API calls, Point extraction |
| **Step 3** | `processed_accident_data_with_weather.csv` | `final_dataframe.csv` | +1 | Congestion calculation, Normalization |
| **Total** | 23 columns | 28 columns | **+5** | Complete enrichment |

---

## Data Quality Metrics

### Before Processing:
- ❌ Missing city names ("Unknown" entries)
- ❌ No geographic coordinates
- ❌ No weather/environmental data
- ❌ No traffic metrics

### After Processing:
- ✅ All cities properly named
- ✅ Geographic coordinates for all records
- ✅ Weather data (rainfall, temperature) for each accident
- ✅ Traffic congestion metrics calculated
- ✅ All 3,000 records complete and ready for analysis

---

## Usage

### Running the Processing Pipeline:

1. **Step 1: Geographic Data Enrichment**
   ```python
   # Run Cell 4 in Data-Analysis.ipynb
   # Or use the code from the notebook
   ```

2. **Step 2: Weather Data Integration**
   ```bash
   python add_weather_to_accidents.py
   ```
   ⚠️ **Note:** Requires Google Earth Engine authentication
   ```python
   ee.Authenticate()
   ```

3. **Step 3: Traffic Congestion Calculation**
   ```python
   # Run Cell 6 in Data-Analysis.ipynb
   # Then save to final_dataframe.csv (Cell 7)
   ```

### Complete Pipeline:
```bash
# 1. Process geographic data (from notebook)
# 2. Add weather data
python add_weather_to_accidents.py

# 3. Calculate traffic congestion (from notebook)
# 4. Save final dataframe
df.to_csv("final_dataframe.csv")
```

---

## Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `earthengine-api` - Google Earth Engine API
- `requests` - HTTP requests (for HERE API, if used)

---

## Notes

- Weather data fetching may take significant time (2-3 API calls per row)
- Progress is saved periodically during weather data enrichment
- Missing weather data is marked with -9999.0 (can be filtered or imputed)
- Traffic congestion uses non-linear scaling for better representation
- All processing maintains the original 3,000 accident records

---

## Output Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `accident_prediction_india.csv` | 3,000 | 23 | Raw input data |
| `processed_accident_data.csv` | 3,000 | 25 | With coordinates |
| `processed_accident_data_with_weather.csv` | 3,000 | 28 | With weather data |
| `final_dataframe.csv` | 3,000 | 28 | Final processed dataset |

---

## Next Steps

After completing the data processing pipeline, proceed to:
1. **Model Training** - See `MODEL.md` for details
2. **Model Evaluation** - Test accuracy and performance
3. **Deployment** - Use `streamlit_app.py` for predictions
