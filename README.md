# ANN-from-Scratch + MLOps Fundamentals
Building an artificial neural network using only the NumPy Python library along with MLOps fundamentals.<br>

## Perceptron
![Image of network](images/perceptron.PNG)

## MLP
![Image of network](images/mlp.PNG)

## Sklearn's MLP and MLOps Workflow
- Used Sklearn's MLPClassifier to perform classification on a diabetes dataset.
- Automated the generation of reports and accuracy statistics using CML and GitHub Actions.
- Used DVC to automate a pipeline for getting data, processing data, training a model and storing metrics.
- Used a further DVC process to automate the pipeline, reports and comparison to other branches.
- Integrated model unit testing on a dummy dataset with a certain number of features. 