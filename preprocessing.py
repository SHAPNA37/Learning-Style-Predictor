"""
preprocessing.py
Handles data loading, cleaning, imputation, feature selection,
encoding, scaling, and train/test split.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

warnings.filterwarnings('ignore')


def load_data(filepath='data/UNI_DATASET.csv'):
    df1 = pd.read_csv(filepath)
    df = df1.copy()
    df.columns = df.columns.str.strip()
    return df


def impute_missing(df):
    # Impute mean for normally distributed columns
    mean_cols = ['Age', 'Attendance', 'Class_Size', 'Sleep_Patterns', 'Screen_Time', 'Time_Wasted_on_Social_Media']
    for col in mean_cols:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)

    # Impute median for skewed column(s)
    median_cols = ['Study_Hours']
    for col in median_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Ordinal mappings
    ordinal_mappings = {
        'Class_Participation': {'Low': 1, 'Medium': 2, 'High': 3},
        'Financial_Status': {'Low': 1, 'Medium': 2, 'High': 3},
        'Parental_Involvement': {'Low': 1, 'Medium': 2, 'High': 3},
        'Motivation': {'Low': 1, 'Medium': 2, 'High': 3},
        'Self_Esteem': {'Low': 1, 'Medium': 2, 'High': 3},
        'Stress_Levels': {'Low': 1, 'Medium': 2, 'High': 3},
        'School_Environment': {'Negative': 1, 'Neutral': 2, 'Positive': 3},
        'Professor_Quality': {'Low': 1, 'Medium': 2, 'High': 3},
        'Physical_Activity': {'Low': 1, 'Medium': 2, 'High': 3},
        'Lack_of_Interest': {'Low': 1, 'Medium': 2, 'High': 3},
        'Sports_Participation': {'Low': 1, 'Medium': 2, 'High': 3},
        'Previous_Grades': {'C': 1, 'B': 2, 'A': 3},
        'Grades': {'C': 1, 'B': 2, 'A': 3}
    }

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    ordinal_cols = list(ordinal_mappings.keys())
    nominal_cols = [col for col in cat_cols if col not in ordinal_cols]

    # Apply ordinal median imputation
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            temp_col = col + '_num'
            df[temp_col] = df[col].map(mapping)
            median_val = df[temp_col].median()
            df[temp_col].fillna(median_val, inplace=True)
            reverse_map = {v: k for k, v in mapping.items()}
            df[col] = df[temp_col].round().astype(int).map(reverse_map)
            df.drop(columns=[temp_col], inplace=True)

    # Apply nominal mode imputation
    for col in nominal_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def remove_invalid_rows(df):
    df = df[
        (df['Study_Hours'] >= 0) &
        (df['Screen_Time'] >= 0) &
        (df['Time_Wasted_on_Social_Media'] >= 0)
    ]
    return df


def select_features(df):
    anova_kruskal_remove = ['Age', 'Attendance', 'Class_Size', 'Time_Wasted_on_Social_Media']

    chi_square_remove = ['Gender', 'Parental_Education', 'Family_Income', 'Previous_Grades', 'Major',
                         'School_Type', 'Financial_Status', 'Educational_Resources', 'Stress_Levels',
                         'Professor_Quality', 'Nutrition', 'Bullying']

    domain_remove = ['Age', 'Gender', 'Parental_Education', 'Family_Income', 'Previous_Grades', 'Major',
                     'School_Type', 'Financial_Status', 'Attendance', 'Class_Size', 'Bullying',
                     'Nutrition', 'Professor_Quality', 'Stress_Levels', 'Parental_Involvement',
                     'Self_Esteem', 'School_Environment', 'Extracurricular_Activities',
                     'Study_Space', 'Tutoring', 'Mentoring']

    columns_to_remove = set(anova_kruskal_remove + chi_square_remove + domain_remove)
    df.drop(columns=[c for c in columns_to_remove if c in df.columns], inplace=True)
    return df


def preprocess(filepath='data/UNI_DATASET.csv', models_dir='models'):
    df = load_data(filepath)
    df = impute_missing(df)
    df = remove_invalid_rows(df)
    df = select_features(df)

    # Features and target
    X = df.drop(columns=['Learning_Style'])
    y = df['Learning_Style']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # One-hot encode categorical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Encode target
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Scale numerical columns
    numerical_cols = ['Study_Hours', 'Sleep_Patterns', 'Screen_Time']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Save scaler, label encoder, and feature names
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, f'{models_dir}/scaler_learning_style.joblib')
    joblib.dump(le, f'{models_dir}/label_encoder_learning_style.joblib')
    joblib.dump(list(X_train.columns), f'{models_dir}/feature_names.joblib')

    print("Preprocessing complete.")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    return X_train, X_test, y_train_enc, y_test_enc, le
