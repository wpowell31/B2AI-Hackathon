from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, log_loss, confusion_matrix
import shap
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
random.seed(31)


# load data
phq9_df = pd.read_csv("phq9_df.csv")


# split data into features and target
X = phq9_df.drop(columns=['Unnamed: 0', 'record_id', "depression_x", "insomnia_x",
                                'gad_x', 'panic_x', 'soc_anx_dis_x', 'Mood_psychiatric disorders',
                                'other_psych_x', 'panic_x', 'ptsd_x', 'schizophrenia_x',
                                'phq9_score', 'Severity'])
y = phq9_df["Severity"].map({'None-minimal': 0,
                             'Mild': 1,
                             'Moderate': 1,
                             'Moderately Severe/Severe': 2})

print(X.columns[:10])
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)


# define model
model = RandomForestClassifier()

# define grid search
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
}

# Define K-Fold cross-validation
kfold = KFold(n_splits=3, shuffle=True, random_state=42)

# perform grid search
grid_search = GridSearchCV(model, param_grid, cv=kfold, n_jobs=-1, scoring='neg_log_loss')
grid_search.fit(X_train, y_train)

# get best parameters
rf_best_model = grid_search.best_estimator_
print(grid_search.best_params_)

# make predictions
y_pred = rf_best_model.predict(X_test)
y_pred_proba = rf_best_model.predict_proba(X_test)

# evaluate model
# Calculate accuracy
accuracy = rf_best_model.score(X_test, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Calculate log loss
ll = log_loss(y_test, y_pred_proba)
print(f'log loss: {ll}')


# save
with open("rf_classification_metrics.txt", "w") as f:
    f.write(f"Score: {accuracy}\n")
    f.write(f'log loss: {ll}')

# Print classification report
print(classification_report(y_test, y_pred))

print("Confusion matrix")
print(confusion_matrix(y_test, y_pred))

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Save the confusion matrix to a text file
np.savetxt('rf_confusion_matrix.txt', conf_matrix, fmt='%d')

