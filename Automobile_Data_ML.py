import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None
automobile_general = pd.read_csv('car_features_msrp.csv')

features = ['Engine HP', 'Engine Cylinders', 'Number of Doors', 'highway MPG', 'city mpg', 'MSRP', 'Popularity']
automobile_features = automobile_general[features]
continous_doors = automobile_features['Coupe'] = np.where(automobile_features['Number of Doors'] == 2, 1, 0)
automobile_features.fillna(automobile_features.mean(), inplace=True)


automobile_features.reset_index(drop=True, inplace=True)
train_data, test_data = train_test_split(automobile_features, test_size=0.5, random_state=42)

# Separate features and target variable
X_train = train_data.drop(['MSRP', 'Coupe', 'Number of Doors'], axis=1)
X_train_df = pd.DataFrame(X_train)
y_train = train_data['MSRP']
z_train = train_data['Coupe']
z_test = test_data['Coupe']
y_train_df = pd.DataFrame(y_train)
X_test = test_data.drop(['MSRP', 'Coupe','Number of Doors'], axis=1)
y_test = test_data['MSRP']

# Perform feature scaling on the training and testing data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def automobile_describe(automobile_general):
    """automobile_describe function prints the statistical description of the utilized dataframe."""
    return print(automobile_general.describe())


automobile_describe(automobile_general)


def heatmap(automobile_general):
    """function creates a heatmap that maps the correlation between different data in the utilized dataframe."""
    plt.figure(figsize=(13, 6))
    sns.heatmap(automobile_general.corr(),
        cmap = 'BrBG',
        fmt = '.1f',
        linewidths = 4,
        annot = True)
    plt.title('Automobile Features Correlation Heatmap')
    plt.show()


heatmap(automobile_general)


def linear_regression(X_train, y_train, X_test, y_test):
    """
    linear_regression fits the training data frames and then predicts the test data frame

    Args:
        X_train (pd.DataFrame):  Training set data.
        y_train (pd.Series): Specifically indexed target variable.
        X_test (pd.DataFrame): Test set data.
        y_test (pd.Series): Specifically indexed target variable.

    Returns:
        Returns the linear regression figure based on the data frame inputs
    """
    linReg = LinearRegression()
    model = linReg.fit(X_train, y_train)
    coef_determine = model.score(X_test, y_test)
    print(f"The coefficient of determination for Linear Regression is {coef_determine}")

    feature_importance = pd.Series(model.coef_, index=X_train.columns)

    plt.figure(figsize=(15, 9))
    feature_importance.sort_values(ascending=True).plot(kind='barh')
    plt.title('Independent Variables Impact on MSRP')
    plt.xlabel('MSRP Impact')
    plt.ylabel('Indepedent Variables from dataset')
    plt.show()

    return coef_determine


linear_regression(X_train, y_train.values.ravel(), X_test, y_test.values.ravel())


def linear_regression_accuracy(X_train, y_train, X_test, y_test):
    """
    linear_regression fits the training data frames and then predicts the test data frame

    Args:
        X_train (pd.DataFrame):  Training set data.
        y_train (pd.Series): Specifically indexed target variable.

    Returns:
        Returns the linear regression figure based on the data frame inputs
    """
    linReg = LinearRegression()
    model = linReg.fit(X_train, y_train)
    coef_determine = model.score(X_test, y_test)

    if coef_determine <= 0.5: 
        print("The selected feature set doesn't have a good accuracy for Linear Regression.")
        print("The data has proven to provide poor results for predicting the MSRP.\n")
    else:
        print("The selected feature set has a usable coefficient of determination for Linear Regression.\n")


linear_regression_accuracy(X_train, y_train.values.ravel(), X_test, y_test.values.ravel())


def logistic_regression(X_train_scaled, X_test_scaled, z_test, z_train):
    """
    logistic_regression fits the training data frames and then predicts the validation data frame

    Args:
        X_train_scaled (pd.DataFrame): Training dataset.
        z_train (pd.Series): Target binary variable.
        X_test_scaled (pd.DataFrame): Validation dataset.
        z_test (pd.Series): Target binary variable for test set.

    Returns:
        Returns the logistic regression figure based on the data frame inputs
    """
    logistic_regression = LogisticRegression(max_iter=100)
    logistic_regression.fit(X_train_scaled, z_train)
    log_initial_result = logistic_regression.predict(X_test_scaled)
    result = np.mean(log_initial_result)
    accuracy = logistic_regression.score(X_test_scaled, z_test)
    print(f"The accuracy of Logistic Regression is {accuracy}")

    if accuracy <= 0.5: 
        print("The selected feature set doesn't have a good accuracy for Logistic Regression.")
        print("The data has proven to provide poor results for predicting the number of car doors.\n")
    else:
        print(f"The selected feature set has a coupe classification accuracy of {accuracy}.\n")
    return result


logistic_regression(X_train_scaled, X_test_scaled, z_test, z_train)


def support_vector_machine(X_train_scaled, z_train, X_test_scaled, z_test):
    """
    system_vector_machine fits the training data frames and then predicts the validation data frame

    Args:
        X_train_scaled (pd.DataFrame): Automobile features data
        z_train (pd.Series): Number of doors training data
        X_test_scaled (pd.DataFrame): Automobile features testing data
        z_test (pd.Series): Number of doors testing data

    Returns:
        Returns the support vector machine figure based on the provided training data
    """
    model_SVR = SVC(kernel='linear', C=1.0, random_state=42)
    model_SVR.fit(X_train_scaled, z_train)
    svm_initial_result = model_SVR.predict(X_test_scaled)
    accuracy = model_SVR.score(X_test_scaled, z_test)
    print(f"The accuracy of Support Vector Machine is {accuracy}")


    if accuracy <= 0.5: 
        print("The selected feature set doesn't have a good accuracy for the support vector machine.")
        print("The data has proven to provide poor results for predicting the MSRP.\n")
    else:
        print(f"The support vector machine model has a coupe classfication of {accuracy}.\n")
    return np.mean(svm_initial_result)


support_vector_machine(X_train_scaled, z_train, X_test_scaled, z_test)


def random_forest_regressor(X_train, y_train, X_test, y_test):
    """
    random_forest_regressor fits the training data frames and then predicts the validation data frame.

    Args:
        X_train (pd.DataFrame): Features training data
        y_train (pd.Series): Target variable training data
        X_test (pd.DataFrame): Features testing data
        y_test (pd.Series): Target variable testing data

    Returns:
        Returns the random forest regression model.
    """
    model_RFR = RandomForestRegressor(n_estimators=125, max_depth=5, random_state=42)
    model_RFR.fit(X_train, y_train)
    feature_importance = model_RFR.feature_importances_
    accuracy = model_RFR.score(X_test, y_test)
    print(f"The accuracy of Random Forest Regressor is {accuracy}")

    # Random Forest Regressor Feature Importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x=X_train.columns, y=feature_importance)
    plt.title('Feature importance percentages for the Random Forest Regressor model')
    plt.xlabel('Independent Variables')
    plt.ylabel('Percentage of total impact')
    plt.show()
    return model_RFR


random_forest_model = random_forest_regressor(X_train, y_train, X_test, y_test)


def random_forest_regressor_accuracy(X_train, y_train,y_test, X_test):
    """
    random_forest_regressor_accuracy fits the training data frames and then predicts the test data frame

    Args:
        X_train (pd.DataFrame):  Training set data.
        y_train (pd.Series): Training data for specifically indexed target variable.
        y_test (pd.Series): Test data for specifically indexed target variable.
        X_test (pd.DataFrame): Test data for the features dataframe.

    Returns:
        Returns the random forest regressor figure based on the data frame inputs
    """
    model_RFR = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model_RFR.fit(X_train, y_train)
    accuracy = model_RFR.score(X_test, y_test)
    
    if accuracy <= 0.5: 
        print("The selected feature set doesn't have a good accuracy for the random forest regressor model.\n")
        print("The data has proven to provide poor results for predicting the MSRP.")
    else:
        print("The selected feature set has a useable accuracy for the random forest regressor model.\n")
    return accuracy


random_forest_regressor_accuracy(X_train, y_train, y_test,X_test)