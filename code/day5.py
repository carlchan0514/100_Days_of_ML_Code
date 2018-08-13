# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Importing the dataset
dataset = pd.read_csv('../datasets/Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# excersize
# if "python setup.py develop" installed the package
from pylib.plot import plot_decision_boundary

# generate some data from sklearn
from sklearn import datasets, linear_model, tree, ensemble
# matplotlib inline
X, Y = datasets.make_blobs(centers=6, random_state=11)
# print(X)
# print(Y)

# # Now let's start visualizing logistic regression
# model = linear_model.LogisticRegression()
# model.fit(X, Y)
# plot_decision_boundary(model, X=X, Y=Y)
# plt.show()
#
# # take a look at decision tree
# model = tree.DecisionTreeClassifier()
# model.fit(X, Y)
# plot_decision_boundary(model, X=X, Y=Y)
# plt.show()
#
# # how about random forest?
# model = ensemble.RandomForestClassifier(n_estimators=21, random_state=1)
# model.fit(X, Y)
# plot_decision_boundary(model, X=X, Y=Y)
# plt.show()
#
# # higher dimension case
# data = datasets.load_iris()
# X = data.data
# Y = data.target
#
# model = linear_model.LogisticRegression()
# model.fit(X, Y)
# plot_decision_boundary(model, X=X, Y=Y)
# plt.show()

# Visualization
# Visualize the Training Set results
x_set, y_set = X_train, Y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min()-1, stop=x_set[:, 0].max()+1, step=0.01),
                     np.arange(start=x_set[:, 1].min()-1, stop=x_set[:, 1].max()+1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'blue')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x1.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()

# Visualize the Test Set Results
x_set, y_set = X_test, Y_test
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min()-1, stop=x_set[:, 0].max()+1, step=0.01),
                     np.arange(start=x_set[:, 1].min()-1, stop=x_set[:, 1].max()+1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x1.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (Testing set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()

# # Now let's visualize logistic regression
# plot_decision_boundary(classifier, X=X_train, Y=Y_train)
# plot_decision_boundary(classifier, X=X_test, Y=Y_test)
# plt.show()

print(Y_pred)