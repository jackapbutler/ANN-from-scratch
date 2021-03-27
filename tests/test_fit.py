import pickle
from sklearn.datasets import make_regression
import json

model = pickle.load(open("../models/sklearn_neuralnet.pkl", "rb"))

# Generate some data for validation
X_test, y = make_regression(n_samples= 1000, n_features= 9, n_classes= 2)

# Test that the model can predict
y_hat = model.predict(X_test)