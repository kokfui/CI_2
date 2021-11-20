import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import vis
jobs = pd.read_csv('Assignment2/Employee.csv')
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
x = jobs[jobs.columns[3::6]]
y = jobs["LeaveOrNot"]

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

print(X_train)
print(X_test)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(2), max_iter=1000)
mlp.fit(X_train, y_train)

#Visualization 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

vis.vis2d(ax, mlp, X_train, y_train, X_test, y_test)

# activation_functions = ['identity','logistic','tanh','relu']
# fig = plt.figure()
# hidden_layers = [(3), (3,3), (3,3,3)]

# activation_functions = ['identity', 'logistic', 'tanh', 'relu']
# hidden_layers = [(3), (3,3), (3,3,3)]
# fig = plt.figure()
# for i,actfcn in enumerate(activation_functions):
#   for j,hlyr in enumerate(hidden_layers):
#     mlp = MLPClassifier(hidden_layer_sizes=hlyr, activation=actfcn, max_iter=1000)
#     mlp.fit(X_train, y_train)
#     ax = fig.add_subplot(len(hidden_layers), len(activation_functions), j*len(activation_functions)+i+1)
#     ax.set_title('{},{},{}'.format(actfcn,str(hlyr),round(mlp.score(X_test,y_test),2)))
#     vis.vis2d(ax, mlp, X_train, y_train, X_test, y_test)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig = plt.figure()
# axes = vis.vis3d(fig, mlp, X_train, y_train, X_test, y_test)
# for i,a in enumerate(axes):
#   a.set_title("LeaveOrNot")
#   a.set_xticklabels([])
#   a.get_yaxis().set_visible(False)
# axes[-1].set_xticklabels("Variables")
plt.show()
