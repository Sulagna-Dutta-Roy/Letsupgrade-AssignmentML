# ===import library ===========
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

# =====loading the data ======
train = pd.read_csv('train.csv')
train.head()

# =====checking the null values =======
train.isnull.sum()

# ====exploratory data analysis =========

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)

# ========================================
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train, palette='Rainbow')

# =======================================
train['Fare'].hist(color='green', bins=40, figsize=(8, 4))

# ============== Data cleaning:============

# ===we want to fill missing age data instead of just dropping the missing age data rows ==================
plt.figure(figsize=(12, 7))
sns.boxplot(X='Pclass', y ='Age', a=train, palette = 'winter')

# ===============Building Logistic model==========================
train.drop('Survived',axis=1).head()
train['Survived'].head()

from sklearn.model_regression import train_test_split

X_train, X_test, y_train, y_test = train_test_split((train.drop('Survived'),axis=1),
train['Survived'], test_size = 0.30, random_state = 101)

# =================Training and predicting==============================
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix

accuracy = confusion_matrix(y_test, predictions)

# ===========================================================
import sklearn.metrics import classication_report

print(classication_report(y_test, predictions))