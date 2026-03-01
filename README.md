# Used Car Price Prediction

Machine learning regression model to predict used car prices using ensemble methods. The project implements custom scikit-learn transformers, a complete ML pipeline with hyperparameter tuning, and achieves R² = 0.80 on test data.

## Project overview
This project builds an end-to-end machine learning system to predict used car prices based on features like mileage, brand, horsepower, accident history, and more. The model helps buyers and sellers estimate fair market values for used vehicles.

## Key results:

- XGBoost model achieves MAE of $6,269 (62.4% improvement over baseline)

- R² score: 0.7956 (explains ~80% of price variance)

- RMSE: $9,414

- Trained on 4,009 used cars with 26 engineered features

## Features

- Custom Transformers: Modular data cleaning, feature engineering, and target encoding

- Feature Engineering: 19 new features including age-mileage interactions, squared terms, log transforms

- Target Encoding: Smoothed brand encoding to capture price patterns by manufacturer

- Outlier Handling: IQR-based filtering of extreme price outliers

- Model Comparison: Automated training and evaluation of multiple models (XGBoost, Random Forest)

- Hyperparameter Tuning: RandomizedSearchCV with 40 iterations and 5-fold CV

- Production Ready: Serialized pipelines and metadata for inference

## Dataset

Source:[ used_cars.csv](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset) (4,009 records)

Features:

- brand, model, model_year

- milage, fuel_type, engine

- transmission, ext_col, int_col

- accident, clean_title, price (target)


## Tech stack

- Python 3.x

- Machine Learning: scikit-learn, XGBoost

- Data Processing: pandas, numpy

- Visualization: matplotlib

- Model Persistence: joblib

## Installation
```bash
# Clone repository
git clone https://github.com/Kaskra13/used_car_price_prediction_pipeline used-car-pipeline
cd used-car-pipeline

# Install dependencies
pip install pandas numpy matplotlib scikit-learn xgboost joblib

# Run training
python used_car_prediction2.py
```

## Usage
Training
```python
# Run full pipeline (data cleaning, training, evaluation)
python used_car_prediction.py
```

Inference
```python
import pandas as pd
import numpy as np
import joblib

# Load artifacts
preprocessing = joblib.load('preprocessing_pipeline.pkl')
target_encoder = joblib.load('target_encoder.pkl')
model = joblib.load('car_price_model.pkl')
metadata = joblib.load('model_metadata.pkl')

# Prepare new data
new_car = pd.DataFrame({
    'brand': ['Tesla'],
    'model': ['Model 3'],
    'model_year': [2021],
    'milage': ['25,000 mi.'],
    'fuel_type': ['Electric'],
    'engine': ['283HP Electric Motor'],
    'transmission': ['Automatic'],
    'ext_col': ['White'],
    'int_col': ['Black'],
    'accident': ['None reported'],
    'clean_title': ['Yes'],
    'price': ['$0']  # placeholder
})

# Transform and predict
processed = preprocessing.transform(new_car)
processed = processed.drop(columns=['price'])
encoded = target_encoder.transform(processed)
prediction_log = model.predict(encoded)
predicted_price = np.expm1(prediction_log)[0]

print(f"Predicted price: ${predicted_price:,.2f}")
```

## Model performance

| Model           | MAE ($) | RMSE ($) | R²     | Improvement |
| --------------- | ------- | -------- | ------ | ----------- |
| Baseline (Mean) | 16,669  | 20,846   | -0.002 | -           |
| XGBoost         | 6,269   | 9,414    | 0.796  | +62.4%      |
| Random Forest   | 7,009   | 10,262   | 0.757  | +58.0%      |



## Feature importance
Top features driving predictions:

| Feature                | Importance |
| ---------------------- | ---------- |
| age_milage_interaction | 61.2%      |
| horsepower             | 11.3%      |
| car_age                | 4.7%       |
| brand_encoded          | 4.3%       |
| is_luxury              | 2.3%       |


Top 3 features account for 77.2% of model decisions


## Pipeline components
1. DataCleaner
- Converts price and mileage to numeric format
- Creates binary accident indicators
- Handles missing values in accident and clean_title

2. FeatureEngineer
- Age Features: car_age, is_new, is_old
- Mileage Features: milage_per_year, low_milage, high_milage
- Engine Features: horsepower extraction, missing indicator
- Luxury Brand Indicator: 28 premium brands encoded
- Polynomial Features: age_squared, milage_squared
- Interactions: age_milage_interaction
- Log Transforms: log_milage, log_milage_per_year
- Fuel Type: One-hot encoding

3. TargetEncoder

- Smoothed brand encoding based on mean prices
- Prevents overfitting with regularization parameter
- Handles unseen brands with global mean
