from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# As this dataset is non-linear we need to incorporate a hidden layer
feature_set, labels = datasets.make_moons(100, noise=0.20)
