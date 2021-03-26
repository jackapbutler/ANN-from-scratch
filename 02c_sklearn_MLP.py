# Packages
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

# set seed
seed = 32

# Data
X = pd.read_csv('./data/processed_data.csv')
y = X.pop('Outcome') # ejects quality column as labels

# Train / Test Split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed)

# Fit model
mlp = MLPClassifier(hidden_layer_sizes=[40, 75, 125, 75, 40], random_state=seed)
mlp.fit(X_tr, y_tr)

# Make predictions
train_preds = mlp.predict(X_tr)
test_preds = mlp.predict(X_te)

# Results
train_score = accuracy_score(y_tr, train_preds)*100
test_score = accuracy_score(y_te, test_preds)*100

# Write this to 
with open('sklearn_metrics.txt', 'w') as outfile:
    outfile.write('Training accuracy: '+str(round(train_score, 4))+'%.')
    outfile.write(' ')
    outfile.write('Testing accuracy: '+str(round(test_score, 4))+'%.')

# Plot loss curve 
plt.plot(mlp.loss_curve_)
plt.title('MLP Error')
plt.savefig("sklearn_mlp_loss_curve.png")
plt.close()
