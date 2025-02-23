# Entire Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset from the CSV file
df = pd.read_csv('fruits.csv')  # Read a CSV file into a DataFrame

# Step 2: Prepare the features and labels
X = df[["Size", "Sweetness", "Nutrients"]]  # Features: selected columns from the DataFrame
y = df["FruitName"]  # Target variable: fruit names from the DataFrame

# Step 3: Encode the labels into numbers
label_encoder = LabelEncoder()  # Create a LabelEncoder object to convert labels to numeric form
y_encoded = label_encoder.fit_transform(y)  # Fit the encoder and transform the labels

# Step 4: Define the range of K values to test
k_values = range(1, 8)  # Testing K from 1 to 8
accuracies = []  # List to store accuracy scores for each K

# Step 5: Perform cross-validation for each K
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)  # Create KNN classifier with k neighbors
    # Perform cross-validation with 3 folds and return mean accuracy
    accuracy = cross_val_score(knn, X, y_encoded, cv=3, scoring='accuracy').mean()  
    # knn: the model to evaluate
    # X: input features for the model
    # y_encoded: target variable (labels) for the model
    # cv=3: number of folds for cross-validation (the data is split into 3 parts)
    # scoring='accuracy': metric used to evaluate the model's performance (accuracy)
    accuracies.append(accuracy)  # Store the mean accuracy for the current k

# Step 6: Fit the model with the best K value on the entire dataset
print(accuracies)  # Print the accuracy scores for all k values

best_k = k_values[np.argmax(accuracies)]  # Find the k value with the highest accuracy
print("BEST K: ", best_k)  # Print the best k value
knn_best = KNeighborsClassifier(n_neighbors=best_k)  # Create KNN classifier with the best k
knn_best.fit(X, y_encoded)  # Fit the model to the entire dataset

# Step 7: Test the KNN model on new data points
# Define new data points (example: Size, Sweetness, Nutrients)
new_data_points = np.array([[6, 3, 1],  # Example values for a new fruit
                             [6, 9, 4],  # Another fruit
                             [7, 6, 6]]) # Another fruit

# Predict the fruit names for the new data points
predicted_indices = knn_best.predict(new_data_points)  # Predict indices of the nearest neighbors
predicted_fruit_names = label_encoder.inverse_transform(predicted_indices)  # Convert indices back to fruit names

# Print the results
for i, point in enumerate(new_data_points):
    print(f"New Data Point {i + 1} (Size: {point[0]}, Sweetness: {point[1]}, Nutrients: {point[2]}) - Predicted Fruit: {predicted_fruit_names[i]}")

# Step 8: Plot the results
plt.figure(figsize=(10, 6))  # Create a figure with a specified size
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')  # Plot k values against accuracies
plt.title("Accuracy of KNN for Different K Values")  # Title of the plot
plt.xlabel("K Value")  # Label for x-axis
plt.ylabel("Accuracy")  # Label for y-axis
plt.xticks(k_values)  # Set x-ticks to the k values
plt.grid(True)  # Enable grid for better readability
plt.show(block=True)  # Show the plot

# Step 9: Determine the best K and corresponding accuracy
print(f"Best K: {best_k}, Best Accuracy: {np.max(accuracies)}")  # Print the best k and its corresponding accuracy
```

<br>
<br>




### Segment 1: Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
```
This segment imports necessary libraries for data manipulation, numerical operations, visualization, KNN classification, cross-validation, and label encoding.

### Segment 2: Loading the Dataset
```python
df = pd.read_csv('fruits.csv')  # Read a CSV file into a DataFrame
```
The dataset is loaded from a CSV file named `fruits.csv` into a DataFrame called `df`.

### Segment 3: Preparing Features and Labels
```python
X = df[["Size", "Sweetness", "Nutrients"]]  # Features: selected columns from the DataFrame
y = df["FruitName"]  # Target variable: fruit names from the DataFrame
```
Here, the features (input variables) are selected as columns `Size`, `Sweetness`, and `Nutrients`. The target variable (output label) is the `FruitName` column.

### Segment 4: Encoding Labels
```python
label_encoder = LabelEncoder()  # Create a LabelEncoder object to convert labels to numeric form
y_encoded = label_encoder.fit_transform(y)  # Fit the encoder and transform the labels
```
A `LabelEncoder` is initialized to convert categorical fruit names into numeric values. The `fit_transform` method encodes the labels.

### Segment 5: Defining K Values and Performing Cross-Validation
```python
k_values = range(1, 8)  # Testing K from 1 to 8
accuracies = []  # List to store accuracy scores for each K

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)  # Create KNN classifier with k neighbors
    accuracy = cross_val_score(knn, X, y_encoded, cv=3, scoring='accuracy').mean()  
    # knn: the model to evaluate
    # X: input features for the model
    # y_encoded: target variable (labels) for the model
    # cv=3: number of folds for cross-validation (the data is split into 3 parts)
    # scoring='accuracy': metric used to evaluate the model's performance (accuracy)
    accuracies.append(accuracy)  # Store the mean accuracy for the current k
```
In this segment, for each K value, the KNN classifier is created. The `cross_val_score` function performs cross-validation to evaluate the model:

- `knn`: The KNeighborsClassifier instance to evaluate.
- `X`: The input features for the model (the independent variables).
- `y_encoded`: The target variable (the labels or dependent variable).
- `cv=3`: Specifies the number of folds for cross-validation; the dataset is split into three parts, and the model is trained on two parts and tested on one part for each fold.
- `scoring='accuracy'`: Specifies the metric to evaluate the model's performance, which is accuracy in this case.

The mean accuracy from the cross-validation scores is appended to the `accuracies` list.

### Segment 6: Finding the Best K Value
```python
print(accuracies)  # Print the accuracy scores for all k values

best_k = k_values[np.argmax(accuracies)]  # Find the k value with the highest accuracy
print("BEST K: ", best_k)  # Print the best k value
knn_best = KNeighborsClassifier(n_neighbors=best_k)  # Create KNN classifier with the best k
knn_best.fit(X, y_encoded)  # Fit the model to the entire dataset
```
After printing the accuracy scores, the best K value is found using `np.argmax(accuracies)`, which returns the index of the maximum accuracy. The KNN classifier is then created with this best K value and fitted to the entire dataset.

### Segment 7: Testing the Model with New Data Points
```python
new_data_points = np.array([[6, 3, 1],  # Example values for a new fruit
                             [6, 9, 4],  # Another fruit
                             [7, 6, 6]]) # Another fruit

predicted_indices = knn_best.predict(new_data_points)  # Predict indices of the nearest neighbors
predicted_fruit_names = label_encoder.inverse_transform(predicted_indices)  # Convert indices back to fruit names

# Print the results
for i, point in enumerate(new_data_points):
    print(f"New Data Point {i + 1} (Size: {point[0]}, Sweetness: {point[1]}, Nutrients: {point[2]}) - Predicted Fruit: {predicted_fruit_names[i]}")
```
In this segment, new data points are defined for prediction. The `predict` method is called on the fitted KNN model:

- `new_data_points`: The new data for which predictions are to be made.
- `predicted_indices`: The indices of the predicted nearest neighbors based on the new data points.
- `label_encoder.inverse_transform(predicted_indices)`: Converts the predicted numeric indices back to the original fruit names.

Finally, the results are printed.

### Segment 8: Plotting Results
```python
plt.figure(figsize=(10, 6))  # Create a figure with a specified size
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')  # Plot k values against accuracies
plt.title("Accuracy of KNN for Different K Values")  # Title of the plot
plt.xlabel("K Value")  # Label for x-axis
plt.ylabel("Accuracy")  # Label for y-axis
plt.xticks(k_values)  # Set x-ticks to the k values
plt.grid(True)  # Enable grid for better readability
plt.show(block=True)  # Show the plot
```
This segment visualizes the accuracy of the KNN model for different K values. The `plt.plot` function creates a line plot:

- `figsize`: Sets the dimensions of the plot.
- `marker`, `linestyle`, `color`: Customize the appearance of the plot (markers for points, line style, and color).

### Segment 9: Printing the Best K and Accuracy
```python
print(f"Best K: {best_k}, Best Accuracy: {np.max(accuracies)}")  # Print the best k and its corresponding accuracy
```
The final segment prints the best K value and its corresponding maximum accuracy from the list of accuracies.

This breakdown provides clear explanations of the function parameters and their roles in the code.
