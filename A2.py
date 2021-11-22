from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

#import dataset
jobs = pd.read_csv('Employee.csv')
print(jobs.head())

# Replace linguistic variables with numerical values
# Key :
# Gender - Male = 0 Female = 1
# Degree - Bachelors = 0 Masters = 1 PHd = 2
# Out of Projects - No = 0 Yes = 1
# City - Bangalore = 0 Pune = 1 New Delhi = 2
jobs['Gender'] = jobs['Gender'].replace(to_replace= "Male", value= 0)
jobs['Gender'] = jobs['Gender'].replace(to_replace= "Female", value= 1)
jobs['Education'] = jobs['Education'].replace(to_replace= "Bachelors", value= 0)
jobs['Education'] = jobs['Education'].replace(to_replace= "Masters", value= 1)
jobs['Education'] = jobs['Education'].replace(to_replace= "PHD", value= 2)
jobs['EverBenched'] = jobs['EverBenched'].replace(to_replace= "No", value= 0)
jobs['EverBenched'] = jobs['EverBenched'].replace(to_replace= "Yes", value= 1)
jobs['City'] = jobs['City'].replace(to_replace= "Bangalore", value= 0)
jobs['City'] = jobs['City'].replace(to_replace= "Pune", value= 1)
jobs['City'] = jobs['City'].replace(to_replace= "New Delhi", value= 2)
print(jobs.head())

# Optimize for Leave or Not
x = jobs.drop(columns=['LeaveOrNot'], axis=1)
y = jobs["LeaveOrNot"]

#Data partition
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

#Scaling data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Classification modelling with 3 hidden layers and 8 neurons in each hidden layer
mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), max_iter=1000)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

#Performance metric of the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)
print("Model Accuracy: ", accuracy)
print("Classification report: \n", report)
print("Confusion Matrix: \n", matrix)
