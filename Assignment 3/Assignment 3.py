import numpy as np
import pandas as pd

# ===========loading the data==============
train_df = pd.read_csv('train.csv')
train_df.columns()

# ===========5 rows in the dataset=========
train_df.head(10)

train_df.tail()

# =============maximum function used to find the total==========================
train_df.describe()

# ==========how many nulls======================================
print(train_df.isnull().sum())

# ===============To remove the null values=======================
train_df.Cabin = train_df.Cabin.fillna("Unknown")
print(train_df.isnull.sum())

# ======fillna method for data cleaning ===================

# =======total data===========
print(train_df.shape)
print("\n")
print(train_df.dtypes)

train_df.info()
print('_'*30)
