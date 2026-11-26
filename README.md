# Uber Fare Predictor

An interactive NYC Uber fare estimation app built with Streamlit, scikit-learn, and Folium. It predicts a base UberX fare from pickup and dropoff coordinates, trip date and time, distance, and passenger count, then displays estimated prices for other ride tiers.

## Features

- Predicts fares with a serialized HistGradientBoosting regression model
- Calculates trip distance using the Haversine formula
- Includes popular NYC route presets
- Supports pickup and dropoff selection by clicking an interactive map
- Shows distance, estimated travel time, traffic period, and ride schedule
- Estimates UberX, Uber Comfort, UberXL, and Uber Black prices
- Displays model validation metrics in the app

## Model

The machine learning workflow includes data cleaning, exploratory model comparison, and serialization of the best-performing model.

- Original dataset: 200,000 rows
- Training dataset referenced by the app: 195,096 trips
- Algorithm: HistGradientBoosting Regressor
- Test R2: 0.8402
- RMSE: $3.88
- MAE: $1.82

Model inputs:

- `distance_km`
- `pickup_longitude`
- `pickup_latitude`
- `dropoff_longitude`
- `dropoff_latitude`
- `passenger_count`
- `hour`
- `day_of_week`
- `month`
- `year`

## Project Structure

```text
uber-fare-predictor/
|-- app.py                 # Streamlit application entry point
|-- requirements.txt       # Python dependencies
|-- data_cleaning/
|   |-- data_cleaning.ipynb
|   |-- uber.csv
|   `-- uber_cleaned.csv
|-- data_visualisations/
|   |-- visualizations.ipynb
|   `-- uber_cleaned.csv
`-- models/
    |-- models.ipynb
    |-- best_model.joblib
    `-- uber_fare_model.joblib
```

The repository root also contains a launcher that starts the Streamlit app in the inner project directory.

## Getting Started

### Prerequisites

- Python 3.9 or newer
- pip

### Installation

Create and activate a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r uber-fare-predictor\requirements.txt
```

### Run the App

From the repository root:

```powershell
python app.py
```

Or run Streamlit directly:

```powershell
streamlit run uber-fare-predictor\app.py
```

Then open the local URL shown by Streamlit, typically `http://localhost:8501`.

## Usage

1. Choose a preset route or enter custom pickup and dropoff coordinates.
2. Select pickup or dropoff as the map click target.
3. Click the map to update coordinates.
4. Set the ride date, time, and passenger count.
5. Review the estimated fare and available ride tiers.

## Notes

- The ride-tier prices are presentation estimates based on fixed multipliers applied to the predicted UberX fare.
- The travel-time estimate is based on average assumed speeds and varies for rush-hour periods.
- Predictions are intended as estimates and are not live Uber quotes.
