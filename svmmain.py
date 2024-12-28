import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import seaborn as sns

# load all the csv files
subdata0 = pd.read_csv(r"C:\Users\jonoz\Downloads\0.csv", header = None)
subdata1 = pd.read_csv(r"C:\Users\jonoz\Downloads\1.csv", header = None)
subdata2 = pd.read_csv(r"C:\Users\jonoz\Downloads\2.csv", header = None)
subdata3 = pd.read_csv(r"C:\Users\jonoz\Downloads\3.csv", header = None)

# consolidate all the data into one singular variable
data = pd.concat([subdata0, subdata1, subdata2, subdata3], axis = 0)

# extract the feature columns (EMG channels) and target column (labels)
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# normalize the data
X_mean = np.mean(X, axis = 0)
X_std = np.std(X, axis = 0)
X = (X - X_mean) / X_std

# t-SNE
if len(X) == 0:
    raise ValueError("No data available for t-SNE.")
X_embedded = TSNE(n_components = 3, perplexity = 30, learning_rate = 50, n_iter = 1000).fit_transform(X.reshape(X.shape[0], -1))

# split the data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = SVC(verbose = 1)
model.fit(X_train, y_train)

# create a prediction variable where the predicted values are stored
y_pred = model.predict(X_test)

# map predicted labels to hand movements
hand_movements = {0: 'rock', 1: 'scissors', 2: 'paper', 3: 'ok'}
y_pred_labels = [hand_movements[label] for label in y_pred]

# print predicted hand movement and sample number
for i in range(len(y_pred_labels)):
    print('Sample', i, ':', y_pred_labels[i])

print(classification_report(y_test, y_pred))

# calculate the confusion matrix
conf = confusion_matrix(y_test, y_pred)

# plot the confusion matrix
plt.figure(figsize = (8, 6))
sns.heatmap(conf, annot = True, fmt = 'd', cmap = 'Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Mapping movements to colors
colors = {'rock': 'purple', 'scissors': 'green', 'paper': 'blue', 'ok': 'yellow'}

# Convert integer labels to string labels
y_labels = [hand_movements[label] for label in y]

# Graphing the manifold
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c = [colors[label] for label in y_labels])
ax.set_xlabel('D-1')
ax.set_ylabel('D-2')
ax.set_zlabel('D-3')
plt.show()
