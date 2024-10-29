# Trip-Duration-Prediction

## Table of Contents

- [Project Overview](#project-overview)
- [Installation and Usage](#installation)
- [Project Structure](#project-structure)
- [Data EDA](#data-eda)

## Project Overview

This project focuses on predicting taxi trip duration in New York City based on data collected from taxis. Accurately predicting trip duration is important for optimizing ride-hailing services and improving traffic management systems.

## Installation
To run this project locally, follow these steps:

```bash
# Install dependencies
pip install -r requirements.txt

# Clone the repository
git clone https://github.com/Mullereto/Trip-Duration-Prediction.git

# Navigate to the project directory
cd Trip-Duration-Prediction

#run load_test.py
python load_test.py
```
## Project Structure

```bash
Trip-Duration-Prediction/
│
├── mini_split/                    
│   ├── test.zip                 
│   ├── train.zip 
│   ├── val.zip 
│
├── model/               
│   ├── history.json     
│   ├── baseline_info.json    
│
├── prepared_data/                       
│   ├── 0/             
│       ├── data_prepartion_details.json      
│       ├── train.csv                
│       ├── val.csv      
│   ├── 1/             
│       ├── data_prepartion_details.json      
│       ├── train.csv                
│       ├── val.csv      
│
├── split/                     
│   ├── test.zip      
│   ├── train.zip          
│   ├── val.zip        
│
├── reports/                    
│   ├── figures/                
│   ├── Report.pdf              
│
├── load_test.py                     
├── preparing_data.py                     
├── saving_baseLine_model.py                     
├── train.py                                        
│            
├── requirements.txt            
├── README.md                                    
```
## DATA EDA

Exploratory Data Analysis (EDA)
2.1. Trip Duration (Target Variable)
- The trip duration distribution follows a Gaussian pattern (log-transformed for better visualization). The data shows outliers in the right tail, with most trips concentrated around 6 to 5 minutes.
2.2. Discrete Numeric Features
- Vendor ID: No clear pattern between trip duration and vendor ID.
- Passenger Count: Trips with 1–6 passengers have a more constant duration, while groups of 7–8 passengers generally take shorter trips. The passenger count of 0 may indicate data entry errors.
2.3. Geography Data
- Using the coordinates of pickup and drop-off locations, we calculated the haversine distance. Most trips are between 1–25 km, with the majority in the 1–5 km range.
2.4. Time Feature Analysis
- New features like month, time of day, and season were derived from the pickup_datetime. We observed a slight increase in trip duration - during the summer season, possibly due to vacation periods.
3. Feature Engineering
**3.1. Datetime-based Features**
**Several datetime-related features were extracted:**

- Hour of Day: To capture rush hour effects.
- Day of Week: Encoded as one of seven days.
- Month: To observe any monthly trends.
- Season: Grouped into winter, spring, summer, and fall.
- Time Period: Segmented into morning, afternoon, evening, and night.
**3.2. Distance and Direction**
**We calculated the following metrics:**

- Haversine Distance: The straight-line distance between pickup and dropoff.
- Manhattan Distance: A grid-based distance measurement, more suited for urban environments.
- Direction: The compass bearing from pickup to dropoff.
4. Correlation Analysis
- Positive correlations: Trip duration (log-transformed) positively correlates with pickup/dropoff longitudes and passenger count.
- Negative correlations: Weak negative correlations with pickup/dropoff latitudes.
**4.1. Handling Categorical Variables**
- Categorical features were one-hot encoded. Infrequent categories in the passenger_count were grouped into an “other” category.

**4.2. Transformations and Log Scaling**
-A log transformation was applied to address the right-skewed nature of trip duration, which improved the model's performance.

**4.3. Outlier Handling**
- Outliers in the trip_duration variable were identified using the IQR method and removed to prevent skewing the model.

5. Modeling
- Initial models included:

- Linear Regression: Provided a baseline R² score of ~0.55.
6. Final Model - Ridge Regression
- The final model chosen was Ridge Regression with an alpha value of 1. Ridge regression was selected to control overfitting by penalizing large coefficients.

7. Model Performance
- Train R² Score: 0.6997
- Validation R² Score: 0.6946
- The model explains about 69.9% of the variance in the trip duration.

6. Conclusion
- Key Features: Trip distance, hour of the day, and Manhattan distance were the most important predictors of trip duration.
- Model Effectiveness: Ridge regression with alpha=1 performed well, achieving a validation R² of 0.6946. The regularization technique - prevented overfitting.
- Feature Importance: Distance metrics and time-based features had the largest impact on the prediction of trip duration.
