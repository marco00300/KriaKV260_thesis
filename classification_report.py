'''classification_report'''

import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

# Step 1: Load y_true and y_pred from pickle files
with open('y_true.pickle', 'rb') as f:
    y_true = pickle.load(f)

with open('y_pred.pickle', 'rb') as f:
    y_pred = pickle.load(f)

# Step 2: Generate the classification report
report = classification_report(y_true, y_pred)

# Step 3: Print the report
print(report)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Assuming you know the class labels
labels = np.unique(y_true)
print("Confusion Matrix:")
print("Labels: ", labels)
print("   " + "  ".join([str(label) for label in labels]))
for i, row in enumerate(conf_matrix):
    print(f"{labels[i]} {row}")
