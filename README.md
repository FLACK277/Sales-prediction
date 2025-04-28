# Sales-prediction
This project implements a machine learning model to predict car purchase amounts based on customer demographics and financial information. The model aims to help car dealerships and marketing teams identify high-value customer segments and optimize their marketing strategies.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data Description](#data-description)
- [Model Pipeline](#model-pipeline)
- [Visualization Outputs](#visualization-outputs)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features
- Data exploration and visualization of car purchase patterns
- Outlier detection and handling
- Feature engineering to create meaningful predictors
- Multiple regression models including Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, and XGBoost
- Model evaluation and comparison
- Feature importance analysis
- Customer segmentation for targeted marketing strategies
- Marketing optimization recommendations
- Model persistence for future predictions

## Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- pickle

## Installation
1. Clone this repository:
```bash
git clone https://github.com/yourusername/car-purchase-prediction.git
cd car-purchase-prediction
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Usage
1. Place your car purchase data CSV file in the project directory or update the file path in `main()`.
2. Run the script:
```python
python car_purchase_prediction.py
```

3. To use the saved model for making predictions:
```python
import pickle

# Load the model
with open('car_purchase_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make predictions on new data
predictions = model.predict(new_data)
```

## Data Description
The model expects a CSV file with the following columns:
- `customer name`: Name of customer (not used in modeling)
- `customer e-mail`: Email address (not used in modeling)
- `country`: Customer's country of residence
- `gender`: Customer's gender
- `age`: Customer's age
- `annual Salary`: Customer's annual salary
- `credit card debt`: Customer's credit card debt
- `net worth`: Customer's net worth
- `car purchase amount`: Target variable - amount spent on car purchase

## Model Pipeline
The prediction pipeline consists of the following steps:
1. **Data Loading and Exploration**: Loads dataset and displays basic statistics
2. **Exploratory Data Analysis**: Visualizes distributions and relationships between variables
3. **Outlier Handling**: Detects and caps outliers in numerical columns
4. **Data Preprocessing**: 
   - Drops unnecessary columns
   - Handles missing values
   - Creates new features (debt-to-income ratio, purchase-to-income ratio, etc.)
   - Encodes categorical variables
   - Scales numerical features
5. **Model Training**: Trains and compares multiple regression models
6. **Feature Importance Analysis**: Identifies the most influential factors in car purchase decisions
7. **Forecasting**: Makes predictions and analyzes forecast errors
8. **Marketing Strategy Optimization**: Generates insights for targeted marketing campaigns

## Visualization Outputs
The script generates several visualization files:
- `car_purchase_distribution.png`: Distribution of car purchase amounts
- `age_distribution.png`: Distribution of customer ages
- `age_vs_purchase.png`: Relationship between age and purchase amount
- `salary_vs_purchase.png`: Relationship between salary and purchase amount
- `purchase_by_gender.png`: Car purchase amounts by gender
- `correlation_matrix.png`: Correlation between different variables
- `purchase_by_country.png`: Car purchase amounts by country
- `debt_vs_purchase.png`: Relationship between credit card debt and purchase amount
- `networth_vs_purchase.png`: Relationship between net worth and purchase amount
- `feature_importance.png`: Importance of different features in prediction
- `actual_vs_predicted.png`: Comparison of actual vs predicted purchase amounts
- `error_distribution.png`: Distribution of prediction errors
- `percent_error.png`: Percent error by actual purchase amount
- `segment_purchase_amount.png`: Average purchase amount by customer segment
- `segment_sizes.png`: Customer count by segment
- `country_purchase_amount.png`: Average purchase amount by country

## Troubleshooting

### Common Issues

#### UnicodeDecodeError when reading CSV
If you encounter a UnicodeDecodeError when reading the CSV file:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc5 in position 5
```

Try specifying the encoding when reading the file:
```python
df = pd.read_csv(file_path, encoding='latin1')
```

If 'latin1' doesn't work, you can try other encodings such as 'ISO-8859-1' or 'cp1252'.

#### Missing dependencies
If you encounter module import errors, make sure all required libraries are installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

#### Memory errors
For large datasets, you might encounter memory issues. Try:
- Using a smaller subset of the data for initial testing
- Increasing your system's swap space
- Adding `low_memory=True` to the pandas read_csv function

## Contributing
Contributions to improve the model are welcome. Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
