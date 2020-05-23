from keras_preprocessing.sequence import pad_sequences
from scipy.io import wavfile
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics, svm
import glob
import numpy as np
import os
import csv
import time
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import winsound

# Actual number train_number = 8000
train_number = 8000
# Actual number train_number = 3000
test_number = 3000
# Actual number train_number = 3000
valid_number = 5000

# Train Read
train_names = []
train_data = []
for filepath in glob.glob(
    './ml-fmi-23-2020/train/train/*'
)[:train_number]:
    fs, data = wavfile.read(filepath)
    train_names.append(os.path.basename(filepath))
    train_data.append(data)
train_data = np.array(train_data)

train_labels_file = open(
    './ml-fmi-23-2020/train.txt', 'r'
)
train_labels = [0] * train_number
for line in train_labels_file.readlines():
    name = line.split(',')[0]
    if name in train_names:
        train_labels[train_names.index(name)] = (
            int(line.split(',')[1])
        )
train_labels = np.array(train_labels)

# Test Read
test_names = []
test_data = []
for filepath in glob.glob(
    './ml-fmi-23-2020/test/test/*'
)[:test_number]:
    fs, data = wavfile.read(filepath)
    test_names.append(os.path.basename(filepath))
    test_data.append(data)
test_data = np.array(test_data)

test_labels_file = open(
    './ml-fmi-23-2020/test.txt', 'r'
)
test_labels = [0] * test_number
for line in test_labels_file.readlines():
    name = line.split(',')[0]
    if name in test_names:
        test_labels[test_names.index(name)] = (
            int(line.split(',')[1])
        )
test_labels = np.array(test_labels)

# Validation Read
valid_names = []
valid_data = []
for filepath in glob.glob(
    './ml-fmi-23-2020/validation/validation/*'
)[:valid_number]:
    fs, data = wavfile.read(filepath)
    valid_names.append(os.path.basename(filepath))
    valid_data.append(data)
valid_data = np.array(valid_data)

valid_labels_file = open(
    './ml-fmi-23-2020/validation.txt', 'r'
)
valid_labels = [0] * valid_number
for line in valid_labels_file.readlines():
    name = line.split(',')[0]
    if name in valid_names:
        valid_labels[valid_names.index(name)] = (
            int(line.split(',')[1])
        )
valid_labels = np.array(valid_labels)

# Model

# C_grid = [1e-11,1e-08, 1e-07, 0.000001, 0.00001, 0.0001, 0.001]
# gamma_grid = [1e-11, 1e-07, 0.000001, 0.00001, 0.0001, 0.001]
# # param_grid = {'C': C_grid, 'gamma' : gamma_grid, 'kernel': ('linear', 'rbf')}
# param_grid = {'C': C_grid}

# grid = GridSearchCV(LinearSVC(), param_grid, cv = 3, scoring = "accuracy")
# grid.fit(train_data, train_labels)
#
# # Find the best model
# print(grid.best_score_)
#
# print(grid.best_params_)
#
# print(grid.best_estimator_)

# means = grid.cv_results_['mean_test_score']
# stds = grid.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, grid.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

model = LinearSVC(C=1e-11)
start = time.time()
# model = RandomForestClassifier(min_samples_leaf=20)

model.fit(train_data, train_labels)
end = time.time()
print("Time:", end - start)
# scores = cross_val_score(model, train_data, train_labels, cv=5)
predictions = model.predict(test_data)
print(predictions)
# print(np.mean(predictions == test_labels))
print("Precizie:", metrics.accuracy_score(predictions, test_labels)*100, "%")

# Scriere in CSV

with open('submission.csv', mode='w', newline='') as rez_file:
    rez_writer = csv.writer(rez_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    rez_writer.writerow(['name', 'label'])
    for i,j in zip(test_names, predictions):
        rez_writer.writerow([i, j])
#
# plt.scatter(test_labels, predictions)
# plt.xlabel("True Values")
# plt.ylabel("Predictions")
# plt.show()

# duration = 1000  # milliseconds
# freq = 440  # Hz
# winsound.Beep(freq, duration)