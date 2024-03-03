# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
datasets = pd.read_csv('csvData.csv')  # Load your dataset containing population, confirmed cases, and target variable
X = datasets.iloc[:, [1, 2]].values  # Selecting features (e.g., population and confirmed cases)
Y = datasets.iloc[:, 3].values  # Target variable indicating whether the disease occurred or not

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the classifier into the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
classifier.fit(X_Train, Y_Train)

# Save the trained model to a file
with open('kip.pkl', 'wb') as file:
    pickle.dump(classifier, file)

# Predicting the test set results
Y_Pred = classifier.predict(X_Test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Train, Y_Train
X1, X2 = np.meshgrid(np.arange(start=X_Set[:, 0].min() - 1, stop=X_Set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_Set[:, 1].min() - 1, stop=X_Set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Random Forest Classifier (Training set)')
plt.xlabel('Population')
plt.ylabel('Confirmed Cases')
plt.legend()
plt.show()

# Visualising the Test set results
X_Set, Y_Set = X_Test, Y_Test
X1, X2 = np.meshgrid(np.arange(start=X_Set[:, 0].min() - 1, stop=X_Set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_Set[:, 1].min() - 1, stop=X_Set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Random Forest Classifier (Test set)')
plt.xlabel('Population')
plt.ylabel('Confirmed Cases')
plt.legend()
plt.show()

# Testing the model with new input data
# Prepare input data (replace the values with your actual data)
input_data = np.array([[29000, 20]])  # Replace population_value and confirmed_cases_value with your actual values

# Scale the input data
scaled_input_data = sc_X.transform(input_data)

# Predict whether the disease will occur based on the input data
prediction = classifier.predict(scaled_input_data)

# Output the prediction result
if prediction[0] == 1:
    print("Based on the input data, the disease is likely to occur.")
else:
    print("Based on the input data, the disease is unlikely to occur.")
