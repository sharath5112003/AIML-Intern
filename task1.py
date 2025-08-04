# titanic_preprocessing_with_notes.py

# --------------- STEP 1: Import Libraries & Dataset ---------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Explore basic info
print("Basic Info:\n", df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe())

# --------------- STEP 2: Handle Missing Values ---------------
# Fill numeric missing values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill categorical missing values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Optional: Drop columns with too many missing values
df.drop(columns=['Cabin'], inplace=True)  # Cabin has many missing values

# --------------- STEP 3: Encode Categorical Variables ---------------
# One-hot encoding for nominal categories
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Optional: Label encoding for ordinal categories (if applicable)
# Pclass is ordinal, but already numerical. No need here.

# --------------- STEP 4: Normalize/Standardize Numerical Features ---------------
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# --------------- STEP 5: Detect and Remove Outliers (IQR Method) ---------------
# Boxplot before removing outliers
sns.boxplot(df['Fare'])
plt.title("Boxplot Before Outlier Removal")
plt.show()

# Remove outliers using IQR
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5*IQR) & (df['Fare'] <= Q3 + 1.5*IQR)]

# Boxplot after removing outliers
sns.boxplot(df['Fare'])
plt.title("Boxplot After Outlier Removal")
plt.show()

# Final cleaned data
print("\nCleaned Data Sample:\n", df.head())

# --------------- INTERVIEW QUESTIONS AND ANSWERS ---------------

print("\nINTERVIEW PREP: Common Questions\n")

questions_answers = {
    "1. Types of missing data:": 
        " - MCAR (Missing Completely at Random)\n - MAR (Missing at Random)\n - MNAR (Missing Not at Random)",
    
    "2. How to handle categorical variables?":
        " - One-hot encoding (for nominal)\n - Label encoding (for ordinal)\n - Target encoding, binary encoding (advanced)",

    "3. Normalization vs Standardization:":
        " - Normalization: Scales to [0,1] range\n - Standardization: Mean 0, Std 1 (z-score)",

    "4. How to detect outliers?":
        " - Visual: Boxplots, scatter plots\n - Statistical: Z-score, IQR\n - Clustering-based: DBSCAN",

    "5. Why is preprocessing important in ML?":
        " - Improves accuracy, reduces noise\n - Handles missing/outlier data\n - Speeds up model convergence",

    "6. One-hot vs Label encoding:":
        " - One-hot: Binary columns for each category (unordered)\n - Label: Assigns integers (ordered)",

    "7. How to handle data imbalance?":
        " - Resampling (SMOTE, undersampling)\n - Use weighted models\n - Change evaluation metrics",

    "8. Can preprocessing affect accuracy?":
        " - Yes, bad preprocessing leads to poor generalization.\n - Proper cleaning helps in better predictions."
}

for q, a in questions_answers.items():
    print(f"{q}\n{a}\n")
