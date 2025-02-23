# Implementation of Regression KNN

## Entire Code (For Fruit Price Prediction)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset from the CSV file
df = pd.read_csv('fruit_prices.csv')  # Read a CSV file into a DataFrame

# Step 2: Prepare the features and target variable
X = df[["Size", "Sweetness", "Nutrients"]]  # Features: selected columns from the DataFrame
y = df["FruitPrice"]  # Target variable: fruit prices from the DataFrame

# Step 3: Scale the features for better performance
scaler = StandardScaler()  # Create a StandardScaler object
X_scaled = scaler.fit_transform(X)  # Fit and transform the features

# Step 4: Define the range of K values to test
k_values = range(1, 8)  # Testing K from 1 to 8
mse_scores = []  # List to store mean squared error scores for each K

# Step 5: Perform cross-validation for each K
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)  # Create KNN regressor with k neighbors
    mse = -cross_val_score(knn, X_scaled, y, cv=3, scoring='neg_mean_squared_error').mean()  
    # knn: the KNeighborsRegressor instance to evaluate
    # X_scaled: scaled feature set (Size, Sweetness, Nutrients) for the model
    # y: target variable (fruit prices) for the model
    # cv=3: number of folds for cross-validation (the data is split into 3 parts)
    # scoring='neg_mean_squared_error': metric used to evaluate the model's performance (negative mean squared error)
    # mean(): compute the mean of the negative MSE scores and negate to get actual MSE
    mse_values.append(mse)  # Store the mean squared error for the current k


# Step 6: Fit the model with the best K value on the entire dataset
print(mse_scores)  # Print the mean squared error scores for all k values

best_k = k_values[np.argmin(mse_scores)]  # Find the k value with the lowest mean squared error
print("BEST K: ", best_k)  # Print the best k value
knn_best = KNeighborsRegressor(n_neighbors=best_k)  # Create KNN regressor with the best k
knn_best.fit(X_scaled, y)  # Fit the model to the entire dataset

# Step 7: Test the KNN model on new data points
# Define new data points (example: Size, Sweetness, Nutrients)
new_data_points = np.array([[6, 3, 1],  # Example values for a new fruit
                             [6, 9, 4],  # Another fruit
                             [7, 6, 6]]) # Another fruit

# Scale the new data points
new_data_points_scaled = scaler.transform(new_data_points)  # Scale the new data points

# Predict the fruit prices for the new data points
predicted_prices = knn_best.predict(new_data_points_scaled)  # Predict prices based on the new data points

# Print the results
for i, point in enumerate(new_data_points):
    print(f"New Data Point {i + 1} (Size: {point[0]}, Sweetness: {point[1]}, Nutrients: {point[2]}) - Predicted Price: {predicted_prices[i]}")

# Step 8: Plot the results
plt.figure(figsize=(10, 6))  # Create a figure with a specified size
plt.plot(k_values, mse_scores, marker='o', linestyle='-', color='g')  # Plot k values against mean squared errors
plt.title("Mean Squared Error of KNN for Different K Values")  # Title of the plot
plt.xlabel("K Value")  # Label for x-axis
plt.ylabel("Mean Squared Error")  # Label for y-axis
plt.xticks(k_values)  # Set x-ticks to the k values
plt.grid(True)  # Enable grid for better readability
plt.show(block=True)  # Show the plot

# Step 9: Determine the best K and corresponding mean squared error
print(f"Best K: {best_k}, Best Mean Squared Error: {np.min(mse_scores)}")  # Print the best k and its corresponding mean squared error
```

## CSV File: fruit_prices.csv

```
Size,Sweetness,Nutrients,FruitPrice
3,7,5,1.5
5,6,4,2.0
4,8,7,1.8
6,1,6,2.2
7,7,8,3.0
2,5,3,1.2
4,6,6,2.1
5,7,5,1.9
6,6,6,2.5
6,6,6,2.4
6,6,6,2.3
5,8,7,2.7
```

## Breakdown of the Code


### Segment 1: Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
```

This segment imports necessary libraries for data manipulation, numerical operations, visualization, KNN regression, cross-validation, and feature scaling.

### Segment 2: Loading the Dataset

```python
df = pd.read_csv('fruit_prices.csv')  # Read a CSV file into a DataFrame
```

The dataset is loaded from a CSV file named fruit_prices.csv into a DataFrame called df.

### Segment 3: Preparing Features and Target Variable

```python
X = df[["Size", "Sweetness", "Nutrients"]]  # Features: selected columns from the DataFrame
y = df["FruitPrice"]  # Target variable: fruit prices from the DataFrame
```

Here, the features (input variables) are selected as columns Size, Sweetness, and Nutrients. The target variable (output label) is the FruitPrice column.

### Segment 4: Scaling the Features

```python
scaler = StandardScaler()  # Create a StandardScaler object
X_scaled = scaler.fit_transform(X)  # Fit and transform the features
```

A StandardScaler is initialized to standardize the features, which helps improve the performance of the KNN algorithm by ensuring that each feature contributes equally.

### Segment 5: Defining K Values and Performing Cross-Validation

```python
k_values = range(1, 8)  # Testing K from 1 to 8
mse_scores = []  # List to store mean squared error scores for each K

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)  # Create KNN regressor with k neighbors
    mse = -cross_val_score(knn, X_scaled, y, cv=3, scoring='neg_mean_squared_error').mean()  
    # knn: the KNeighborsRegressor instance to evaluate
    # X_scaled: scaled feature set (Size, Sweetness, Nutrients) for the model
    # y: target variable (fruit prices) for the model
    # cv=3: number of folds for cross-validation (the data is split into 3 parts)
    # scoring='neg_mean_squared_error': metric used to evaluate the model's performance (negative mean squared error)
    # mean(): compute the mean of the negative MSE scores and negate to get actual MSE
    mse_values.append(mse)  # Store the mean squared error for the current k

```

In this segment, for each K value, the KNN regressor is created. The cross_val_score function performs cross-validation to evaluate the model by computing the mean squared error for each K value.

### Segment 6: Finding the Best K Value

```python
print(mse_scores)  # Print the mean squared error scores for all k values

best_k = k_values[np.argmin(mse_scores)]  # Find the k value with the lowest mean squared error
print("BEST K: ", best_k)  # Print the best k value
knn_best = KNeighborsRegressor(n_neighbors=best_k)  # Create KNN regressor with the best k
knn_best.fit(X_scaled, y)  # Fit the model to the entire dataset
```

After printing the mean squared error scores, the best K value is found using np.argmin(mse_scores), which returns the index of the minimum error. The KNN regressor is then created with this best K value and fitted to the entire dataset.

### Segment 7: Testing the Model with New Data Points

```python
new_data_points = np.array([[6, 3, 1],  # Example values for a new fruit
                             [6, 9, 4],  # Another fruit
                             [7, 6, 6]]) # Another fruit

new_data_points_scaled = scaler.transform(new_data_points)  # Scale the new data points

predicted_prices = knn_best.predict(new_data_points_scaled)  # Predict prices based on the new data points

# Print the results
for i, point in enumerate(new_data_points):
    print(f"New Data Point {i + 1} (Size: {point[0]}, Sweetness: {point[1]}, Nutrients: {point[2]}) - Predicted Price: {predicted_prices[i]}")
```

In this segment, new data points are defined for prediction. The new data points are scaled using the same scaler. The predict method is called on the fitted KNN model to estimate the fruit prices for these points.

### Segment 8: Plotting Results

```python
plt.figure(figsize=(10, 6))  # Create a figure with a specified size
plt.plot(k_values, mse_scores, marker='o', linestyle='-', color='g')  # Plot k values against mean squared errors
plt.title("Mean Squared Error of KNN for Different K Values")  # Title of the plot
plt.xlabel("K Value")  # Label for x-axis
plt.ylabel("Mean Squared Error")  # Label for y-axis
plt.xticks(k_values)  # Set x-ticks to the k values
plt.grid(True)  # Enable grid for better readability
plt.show(block=True)  # Show the plot
```

This segment visualizes the mean squared error of the KNN model for different K values. The plot helps in understanding the impact of K on model performance.

### Segment 9: Printing the Best K and Mean Squared Error

```python
print(f"Best K: {best_k}, Best Mean Squared Error: {np.min(mse_scores)}")  # Print the best k and its corresponding mean squared error
```

The final segment prints the best K value and its corresponding minimum mean squared error from the list of scores.
