import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE

# Load EEG data from CSV files and consolidate
data_paths = [r"C:\Users\jonoz\Downloads\0.csv", r"C:\Users\jonoz\Downloads\1.csv", r"C:\Users\jonoz\Downloads\2.csv", r"C:\Users\jonoz\Downloads\3.csv"]

data_list = [pd.read_csv(path, header=None) for path in data_paths]
data = pd.concat(data_list, axis=0)

# Extract feature columns (EEG channels) and target column (labels)
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Normalize the data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Check the dimensionality of the data
input_dimension = X.shape[1]

# Decide whether to use t-SNE based on input dimensionality
if input_dimension <= 65: 
    X_embedded = X  # No need for t-SNE if dimension is 3 or less
else:
    X_embedded = TSNE(n_components=3, perplexity=30, learning_rate=50, n_iter=1000).fit_transform(X)

# Split the data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X_embedded, y, test_size=0.2)

# Create and train the SVM model
model = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Map predicted labels to hand movements
hand_movements = {0: 'rock', 1: 'scissors', 2: 'paper', 3: 'ok'}
y_pred_labels = [hand_movements[label] for label in y_pred]

# Print predicted hand movement and sample number
for i in range(len(y_pred_labels)):
    print('Sample', i, ':', y_pred_labels[i])

print(classification_report(y_test, y_pred))

# Calculate the confusion matrix
conf = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues')
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
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=[colors[label] for label in y_labels])
ax.set_xlabel('D-1')
ax.set_ylabel('D-2')
ax.set_zlabel('D-3')
plt.show()