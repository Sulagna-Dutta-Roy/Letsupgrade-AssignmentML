from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# ===============svm classifier====================
iris_data = read_csv('iris.csv')

D = iris_data.values
x = D[:, 0:4]
y = D(:,4)

x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.20)

model = SVC()
model.fit(x_tr, y_tr)

predict_flower = model.predict(x_ts)
print('Accuracy', accuracy_score(y_ts, predict_flower))

# ================Forest Classifier====================================
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

digits = load_digits()

dir(digits)

plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])

df = pd.dataframe(digits.data)
df.head()

df['target'] = digits.target
df[0:12]

x = df.drop('target', axis='columns')
y = df.target
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)

model.score(X_test, y_test)
y_predicted = model.predict(X_test)

cm = confusion matrix(y_test, y_predicted)

# =========================================
plt.figure(figsize=(10,8))
sn.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')