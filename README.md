# Berkley Automobile-Data-Analysis

## Overview

This Python project focuses on predicting automobile prices based on various features such as engine specifications, fuel efficiency, and popularity. It also predicts if a car is a coupe or not based on the same independent variables. The Berkley automobile dataset was used for this specific program. The prediction is performed using linear regression, logistic regression, support vector machine, and random forest regressor models.

## Getting Started

### Prerequisites

- Python version 3.0 or greater
- Libraries: pandas, matplotlib, seaborn, numpy, scikit-learn
  (install with `pip install pandas matplotlib seaborn numpy scikit-learn`)

### Running the Script

1. Clone the repository to your local machine.
2. Ensure the required libraries are installed.
3. Open a terminal and navigate to the project directory.
4. Run the script by executing the command:
   ```bash
   python Automobile_Data_Analysis.py

### Features
Feature Selection and Binary Variable Creation: Engine HP, Engine Cylinders, Number of Doors, highway MPG, city mpg, MSRP, and Popularity features are selected. 
A binary variable 'Coupe' is created based on the number of doors.
Handling Missing Values: Missing values are filled with the mean of the respective columns.

### Data Preprocessing

### Feature Selection and Binary Variable Creation
features = ['Engine HP', 'Engine Cylinders', 'Number of Doors', 'highway MPG', 'city mpg', 'MSRP', 'Popularity']
automobile_features = automobile_general[features]
automobile_features['Coupe'] = np.where(automobile_features['Number of Doors'] == 2, 1, 0)

### Splitting the Dataset
The dataset is split into training and testing sets using a 50-50 split ratio.

Models and Analysis

### Handling Missing Values
automobile_features.fillna(automobile_features.mean(), inplace=True)

### Splitting the Dataset
continuous_doors = automobile_general['Number of Doors']
automobile_features.reset_index(drop=True, inplace=True)
train_data, test_data = train_test_split(automobile_features, test_size=0.5, random_state=42)

### Models and Analysis

### Linear Regression
linear_regression(X_train_scaled, y_train, X_test_scaled)
linear_regression_accuracy(X_train, y_train)

### Logistic Regression
logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)

### Support Vector Machine (SVM)
support_vector_machine(X_train_scaled, y_train, X_test_scaled, y_test)

### Random Forest Regressor
random_forest_regressor(X_train, y_train, X_test, y_test)
random_forest_regressor_accuracy(X_train, y_train, y_test, X_test)

### Conclusion
This project analyzes automobile price prediction using various machine learning models. It assesses model performance and provides insights into their effectiveness in predicting automobile prices based on selected features. It also does the same for predicting if the vehicle is a coupe or not.

Feel free to use or contribute to this project. If you encounter any issues or have suggestions for improvement, please submit an issue or pull request.
