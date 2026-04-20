"""
train.py
Trains the XGBoost model using GridSearchCV for hyperparameter tuning.
Saves the best model to the models/ directory.
Run this script to retrain the model from scratch.

Usage:
    python train.py
"""

import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import preprocess


def train(models_dir='models'):
    # Step 1: Preprocess data
    X_train, X_test, y_train_enc, y_test_enc, le = preprocess(
        filepath='data/UNI_DATASET.csv',
        models_dir=models_dir
    )

    # Step 2: Initialize XGBoost with multiclass settings
    xgb_model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )

    # Step 3: GridSearchCV for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    print("Running GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train_enc)

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Accuracy:", grid_search.best_score_)

    # Step 4: Evaluate best model on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

    # Step 5: Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test_enc, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    print("Confusion matrix saved as confusion_matrix.png")

    # Step 6: Save the best model
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, f'{models_dir}/xxgb_learning_style_model.joblib')
    print(f"\nModel saved to {models_dir}/xxgb_learning_style_model.joblib")


if __name__ == '__main__':
    train()
